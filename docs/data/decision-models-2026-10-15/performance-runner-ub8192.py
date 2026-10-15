"""Interleaved System One latency and process-memory comparison.

Input profiles are frozen JSON cases previously checked for token counts and
answers. No model weights or activation formats are changed by this runner.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 96)]


def footprint(pid):
    library = ctypes.CDLL(ctypes.util.find_library("proc"))
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    usage = Usage()
    if library.proc_pid_rusage(pid, 4, ctypes.byref(usage)):
        return None
    return {
        "rss_bytes": usage.values[6],
        "physical_footprint_bytes": usage.values[7],
        "lifetime_peak_physical_footprint_bytes": usage.values[28],
    }


def children(pid):
    listing = subprocess.check_output(["ps", "-Ao", "pid=,ppid="], text=True)
    rows = [
        tuple(map(int, line.split()))
        for line in listing.splitlines()
        if len(line.split()) == 2
    ]
    selected = {pid}
    while True:
        found = {child for child, parent in rows if parent in selected}
        if found <= selected:
            return sorted(selected)
        selected |= found


def call(port, path, body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def check_response(response, expected_tokens, body):
    if response["usage"] != {"input_tokens": expected_tokens, "output_tokens": 0}:
        raise RuntimeError(f"unexpected decision token usage: {response['usage']}")
    if set(response["answers"]) != set(body["questions"]):
        raise RuntimeError("decision result omitted or added a question")
    for qid, answer in response["answers"].items():
        if answer.get("type") != body["questions"][qid]["type"]:
            raise RuntimeError("decision question type changed")
        if answer["type"] == "noul":
            value = answer["noul"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError("invalid Noul probability")
            continue
        probabilities = answer["probabilities"]
        question = body["questions"][qid]
        expected = (
            set(question["criteria"])
            if answer["type"] == "choice"
            else {str(i) for i in range(len(question["criteria"]))}
        )
        if set(probabilities) != expected:
            raise RuntimeError("decision candidate mapping changed")
        if not probabilities or any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities.values()
        ):
            raise RuntimeError("invalid decision probability")
        if abs(sum(probabilities.values()) - 1) > 1e-6:
            raise RuntimeError("invalid decision distribution")
        if answer["type"] == "choice" and (
            answer.get("choice") not in probabilities
            or probabilities[answer["choice"]] != max(probabilities.values())
        ):
            raise RuntimeError("decision does not select its maximum probability")
        if answer["type"] == "score" and not math.isfinite(answer["score"]):
            raise RuntimeError("invalid ordinal score")


class MemorySampler:
    def __init__(self, server):
        self.server = server
        self.peak = 0
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stopped.is_set():
            try:
                self.peak = max(
                    self.peak, self.server.memory()["total_physical_footprint_bytes"]
                )
            except (OSError, RuntimeError) as error:
                self.error = str(error)
            self.stopped.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()


def environment(exclude):
    # Import only when running from a checkout; the established isolation
    # probe detects active inference and benchmark binaries without stopping them.
    from dev.benchmarks.qwen4b_ffn_cache import environment as existing_probe

    return existing_probe(exclude=exclude)


def wait_idle(exclude, samples):
    stable = 0
    while stable < 3:
        snapshot = environment(exclude)
        samples.append({"at": time.time(), **snapshot})
        gpu = snapshot["gpu_utilization_percent"]
        if not snapshot["busy_pids"] and gpu is not None and gpu <= 50:
            stable += 1
        else:
            stable = 0
            print(
                "WAIT idle",
                {"gpu": gpu, "busy_pids": snapshot["busy_pids"]},
                flush=True,
            )
        time.sleep(1)


class Server:
    def __init__(self, args, profile, engine, output):
        self.port = 18144 if engine == "llama" else 18145
        self.engine = engine
        if engine == "llama":
            command = [
                str(args.reference_binary),
                "-m",
                profile["model"],
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "-c",
                "8193",
                "-np",
                "1",
                "-ngl",
                "99",
                "-b",
                "8192",
                "-ub",
                "8192",
                "--no-warmup",
                "--embd-normalize",
                "-1",
            ]
            if profile["name"] == "Kev-0.8B":
                command += ["-ctk", "bf16", "-ctv", "bf16"]
        else:
            command = [
                "sh",
                profile["launcher"],
                "--binary",
                str(args.binary),
                "--port",
                str(self.port),
                "--queue-size",
                "1",
            ]
        self.command = command
        self.log = (output / f"{profile['name']}-{engine}.log").open("w")
        self.process = subprocess.Popen(
            command,
            cwd=args.repo,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"{engine} startup exited {self.process.returncode}"
                    )
                if time.monotonic() >= deadline:
                    self.close()
                    raise TimeoutError(f"{engine} startup timeout")
                try:
                    call(self.port, "/health" if engine == "llama" else "/ready")
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup_error:
                original.add_note(f"startup cleanup: {cleanup_error}")
            raise
        self.pids = children(self.process.pid)

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
                raise RuntimeError("benchmark service required forced termination")
        self.log.close()
        if self.process.returncode:
            raise RuntimeError(
                f"{self.engine} cleanup exited {self.process.returncode}"
            )

    def memory(self):
        parts = {str(pid): footprint(pid) for pid in self.pids}
        if any(value is None for value in parts.values()):
            raise RuntimeError("benchmark process memory sample unavailable")
        return {
            "processes": parts,
            "total_rss_bytes": sum(x["rss_bytes"] for x in parts.values()),
            "total_physical_footprint_bytes": sum(
                x["physical_footprint_bytes"] for x in parts.values()
            ),
            "sum_process_lifetime_peaks_bytes": sum(
                x["lifetime_peak_physical_footprint_bytes"] for x in parts.values()
            ),
        }


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "performance_test": True,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
        ).strip(),
        "rounds": args.rounds,
        "requests_per_round": args.samples,
        "context": {"splash": 8192, "llama": 8193, "reference_reserved_slot": 1},
        "order": "AB BA AB (alternating round order)",
        "isolation": [],
        "models": {},
    }
    from dev.benchmarks.qwen4b_ffn_cache import DEVICE_LOCK

    lock = DEVICE_LOCK.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )

    try:
        for name in args.models.split(","):
            profile = json.loads(
                (args.cases_dir / f"{name}-benchmark-cases.json").read_text()
            )
            for length in args.lengths:
                rounds = profile["bodies"][str(length)]
                if len(rounds) != args.rounds or any(
                    len(items) != args.samples for items in rounds
                ):
                    raise RuntimeError(
                        "frozen input counts do not match requested rounds/samples"
                    )
            servers = {}
            model = {"profile": profile, "samples": [], "commands": {}, "cleanup": {}}
            result["models"][name] = model
            try:
                for engine in ("llama", "splash"):
                    servers[engine] = Server(args, profile, engine, args.output)
                    model["commands"][engine] = servers[engine].command
                owned = [
                    os.getpid(),
                    *(pid for server in servers.values() for pid in server.pids),
                ]
                for length in args.lengths:
                    # Warm both paths at the actual shape. Timed inputs have
                    # distinct prefixes, so neither path can reuse the document.
                    warm = profile["warmup"][str(length)]
                    for server in servers.values():
                        check_response(
                            call(server.port, "/v1/systemone", warm), length, warm
                        )
                    for round_number in range(args.rounds):
                        order = (
                            ("llama", "splash")
                            if round_number % 2 == 0
                            else ("splash", "llama")
                        )
                        bodies = profile["bodies"][str(length)][round_number]
                        comparisons = {}
                        for engine in order:
                            wait_idle(owned, result["isolation"])
                            server = servers[engine]
                            print(
                                "START",
                                name,
                                length,
                                round_number + 1,
                                engine,
                                flush=True,
                            )
                            memory_before = server.memory()
                            responses, durations = [], []
                            with MemorySampler(server) as sampler:
                                for body in bodies:
                                    started = time.perf_counter()
                                    response = call(server.port, "/v1/systemone", body)
                                    elapsed = time.perf_counter() - started
                                    check_response(response, length, body)
                                    responses.append(response)
                                    durations.append(elapsed)
                            comparisons[engine] = responses
                            row = {
                                "input_tokens": length,
                                "round": round_number + 1,
                                "engine": engine,
                                "seconds": durations,
                                "mean_seconds": statistics.mean(durations),
                                "median_seconds": statistics.median(durations),
                                "p95_seconds": percentile(durations, 0.95),
                                "input_tokens_per_second": length
                                / statistics.mean(durations),
                                "questions_per_second": 1 / statistics.mean(durations),
                                "observed_peak_physical_footprint_bytes": max(
                                    sampler.peak,
                                    memory_before["total_physical_footprint_bytes"],
                                ),
                                "memory_sampling_error": sampler.error,
                                "memory_before": memory_before,
                                "memory_after": server.memory(),
                                "responses": responses,
                            }
                            if engine == "splash":
                                row["native_status"] = call(server.port, "/status")
                            after = environment(owned)
                            row["isolation_after"] = after
                            if after["busy_pids"] or sampler.error:
                                raise RuntimeError(
                                    "performance block was interrupted by external inference or memory sampling failed"
                                )
                            model["samples"].append(row)
                            save()
                            print(
                                "END",
                                name,
                                length,
                                round_number + 1,
                                engine,
                                round(row["mean_seconds"] * 1000, 3),
                                "ms",
                                flush=True,
                            )
                        for actual, expected in zip(
                            comparisons["splash"], comparisons["llama"]
                        ):
                            for qid, answer in actual["answers"].items():
                                if answer.get("choice") != expected["answers"][qid].get(
                                    "choice"
                                ):
                                    raise RuntimeError(
                                        "performance sample changed the reference decision"
                                    )
            except BaseException as error:
                model["error"] = repr(error)
                result["status"] = "failed"
                save()
                raise
            finally:
                primary_error = sys.exc_info()[1]
                cleanup_errors = []
                for engine, server in reversed(list(servers.items())):
                    try:
                        server.close()
                    except BaseException as error:
                        cleanup_errors.append(f"{engine}: {error}")
                    model["cleanup"][engine] = server.process.returncode
                    save()
                if cleanup_errors:
                    model["cleanup_errors"] = cleanup_errors
                    result["status"] = "failed"
                    save()
                    if primary_error is None:
                        raise RuntimeError("; ".join(cleanup_errors))
                    primary_error.add_note("; ".join(cleanup_errors))
        result["status"] = "passed"
        save()
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="Kev-0.8B,Julia-1,Laya")
    parser.add_argument(
        "--lengths",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[128, 512, 2048, 8192],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1 or args.samples < 1:
        parser.error("rounds and samples must be positive")
    run(args)


if __name__ == "__main__":
    main()
