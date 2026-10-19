"""D3 correctness and interleaved LM Studio/Splash time and footprint comparison."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev.benchmarks.decision_models import (
    MemorySampler,
    check_response,
    children,
    footprint,
    wait_idle,
)
from dev.benchmarks.qwen4b_ffn_cache import DEVICE_LOCK, environment
from install import models
from server import judgments
from server.d3 import D3Scorer


def call(url, path, body=None, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        url + path,
        headers=headers,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from None


class Splash:
    def __init__(self, args, name, output, selected=True):
        self.command = [
            "sh",
            f"d3{name}splash.sh",
            "--port",
            "18145",
            "--binary",
            str(args.binary),
            "--max-context",
            "262144",
            "--max-memory",
            "40G",
            "--kv-format",
            "q4h",
            "--queue-size",
            "4" if args.phase == "correctness" else "1",
            "--encoder-option-queries",
            "on" if selected else "off",
        ]
        self.log = (
            output / f"{name}-splash-{'selected' if selected else 'full'}.log"
        ).open("w")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            cwd=args.repo,
            start_new_session=True,
        )
        self.url = "http://127.0.0.1:18145"
        deadline = time.monotonic() + 240
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Splash startup exited {self.process.returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Splash startup timeout")
                try:
                    call(self.url, "/ready")
                    break
                except OSError:
                    time.sleep(0.25)
            self.pids = children(self.process.pid)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
                raise RuntimeError("Splash required forced cleanup") from None
        self.log.close()
        if self.process.returncode:
            raise RuntimeError(f"Splash exited {self.process.returncode}")

    def memory(self):
        return memory(self.pids)


def memory(pids):
    values = {str(pid): footprint(pid) for pid in pids}
    if not values or any(value is None for value in values.values()):
        raise RuntimeError("process footprint unavailable")
    return {
        "processes": values,
        "total_physical_footprint_bytes": sum(
            v["physical_footprint_bytes"] for v in values.values()
        ),
        "total_rss_bytes": sum(v["rss_bytes"] for v in values.values()),
    }


def host_memory():
    return {
        "vm_stat": subprocess.check_output(["vm_stat"], text=True),
        "swap": subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True
        ).strip(),
    }


def wait_for_device(exclude, samples, accept_desktop_load):
    if not accept_desktop_load:
        return wait_idle(exclude, samples)
    stable = 0
    while stable < 3:
        snapshot = environment(exclude)
        samples.append({"at": time.time(), **snapshot})
        if snapshot["busy_pids"]:
            stable = 0
            print("WAIT external inference", snapshot["busy_pids"], flush=True)
        else:
            stable += 1
        time.sleep(1)


class Reference:
    def __init__(self, args, name):
        # Read only the owned model's native listener, never unrelated processes.
        port = args.lmstudio_lite_port if name == "lite" else args.lmstudio_mini_port
        if port is None:
            raise ValueError(
                "performance requires explicit LM Studio native listener ports"
            )
        found = subprocess.check_output(
            ["lsof", "-t", "-iTCP:" + str(port), "-sTCP:LISTEN"], text=True
        )
        self.pids = sorted({int(v) for v in found.split()})

    def memory(self):
        return memory(self.pids)


def llama_command(args, name):
    directory = Path.home() / f".lmstudio/models/ggml-org/vllm-d3-{name}-GGUF"
    variant = "BF16" if name == "lite" else "Q4_K_M"
    return [
        str(args.reference_binary),
        "--model",
        str(directory / f"vllm-d3-{name}-{variant}.gguf"),
        "--mmproj",
        str(directory / f"mmproj-vllm-d3-{name}-BF16.gguf"),
        "--host",
        "127.0.0.1",
        "--port",
        "18144",
        "--alias",
        "vllm-d3-" + name,
        "--ctx-size",
        "262144",
        "--parallel",
        "4",
        "--batch-size",
        "32768",
        "--ubatch-size",
        "32768",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--flash-attn",
        "on",
        "--n-gpu-layers",
        "999999",
        "--threads",
        "13",
    ]


class Llama:
    def __init__(self, args, name, output):
        self.command = llama_command(args, name)
        self.log = (output / f"{name}-llama.log").open("w")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            cwd=args.repo,
            start_new_session=True,
        )
        self.url = "http://127.0.0.1:18144"
        deadline = time.monotonic() + 240
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"llama.cpp startup exited {self.process.returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("llama.cpp startup timeout")
                try:
                    call(self.url, "/health")
                    break
                except (OSError, RuntimeError):
                    time.sleep(0.25)
            self.pids = children(self.process.pid)
            self.properties = call(self.url, "/props")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
                raise RuntimeError("llama.cpp required forced cleanup") from None
        self.log.close()
        if self.process.returncode:
            raise RuntimeError(f"llama.cpp exited {self.process.returncode}")

    def memory(self):
        return memory(self.pids)


def frontend_hash(args):
    digest = hashlib.sha256()
    for path in sorted((args.repo / "server").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def scorer(args, name):
    from transformers import AutoTokenizer

    variant = "BF16" if name == "lite" else "Q4_K_M"
    selection = models.Selection.of(
        args.repo / "install/models-local",
        f"ggml-org/vllm-d3-{name}-GGUF:{variant}",
        language_only=True,
    )
    root = selection.link.resolve()
    config = json.loads((root / "config.json").read_text())["decision_config"]
    tokenizer = AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True, fix_mistral_regex=False
    )
    return D3Scorer(tokenizer, config)


def count_tokens(model_scorer, body):
    state, specs, errors = judgments.validate_systemone(body)
    if errors:
        raise ValueError(errors)
    return sum(
        len(model_scorer.encode(state, spec, body["questions"][qid])[0])
        for qid, spec in specs
        if not spec.deterministic
    )


def case(model_scorer, length, number):
    body = {
        "state": "",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Choose the responsible team.",
                "criteria": {
                    "billing": "Refunds",
                    "shipping": "Delivery",
                },
            }
        },
    }
    prefix = f"Run {number}. Archive:"
    tail = "\nCurrent request: charged twice; please refund the extra charge."
    count = 0
    for _ in range(12):
        body["state"] = prefix + " note" * count + tail
        actual = count_tokens(model_scorer, body)
        if actual == length:
            return body
        count += length - actual
        if count < 0:
            raise ValueError("requested input too short")
    raise RuntimeError("could not freeze exact token count")


def same_decision(a, b):
    for qid, left in a["answers"].items():
        right = b["answers"][qid]
        if left["type"] == "choice" and left["choice"] != right["choice"]:
            raise AssertionError("Splash and LM Studio selected different choices")
        if left["type"] == "noul" and (left["noul"] >= 0.5) != (right["noul"] >= 0.5):
            raise AssertionError("Splash and LM Studio disagree on Noul")
        if left["type"] == "score":
            if max(left["probabilities"], key=left["probabilities"].get) != max(
                right["probabilities"], key=right["probabilities"].get
            ):
                raise AssertionError("Splash and LM Studio disagree on score mode")


def request(endpoint, key, model, body, expected):
    payload = copy.deepcopy(body)
    payload["model"] = model
    response = call(endpoint, "/v1/systemone", payload, key)
    check_response(response, expected, body)
    return response


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    engine_hash = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    application_hash = frontend_hash(args)
    if args.phase == "performance":
        if args.correctness_report is None:
            raise ValueError("performance requires --correctness-report")
        prior = json.loads(args.correctness_report.read_text())
        if (
            prior.get("status") != "passed"
            or prior.get("phase") != "correctness"
            or prior.get("engine_sha256") != engine_hash
            or prior.get("frontend_sha256") != application_hash
            or not set(args.models) <= set(prior.get("models", {}))
            or not set(args.lengths) <= set(prior.get("validated_lengths", []))
            or prior.get("reference_mode") != args.reference_mode
        ):
            raise ValueError("matching-engine correctness report did not pass")
    key = args.lmstudio_key_file.read_text().strip() if args.lmstudio_key_file else None
    compare = args.reference_mode != "none"
    reference_name = "llama" if args.reference_mode == "llama" else "lmstudio"
    if args.reference_mode == "lmstudio" and not key:
        raise ValueError("LM Studio comparison requires --lmstudio-key-file")
    report = {
        "phase": args.phase,
        "reference_mode": args.reference_mode,
        "engine_sha256": engine_hash,
        "frontend_sha256": application_hash,
        "reference_binary_sha256": hashlib.sha256(
            args.reference_binary.read_bytes()
        ).hexdigest()
        if args.reference_mode == "llama"
        else None,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "context_capacity": 262144,
        "validated_lengths": args.lengths,
        "splash_kv": "q4h",
        "reference_kv": "q4_0" if compare else None,
        "lmstudio_kv": "q4_0" if args.reference_mode == "lmstudio" else None,
        "concurrent_requests": 1,
        "accept_desktop_load": args.accept_desktop_load,
        "models": {},
        "status": "running",
        "isolation": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )

    report["lmstudio_models"] = (
        [
            m
            for m in call(args.lmstudio_url, "/api/v1/models", key=key)["models"]
            if m["key"] in {"vllm-d3-lite", "vllm-d3-mini"}
        ]
        if args.reference_mode == "lmstudio"
        else []
    )
    lock = DEVICE_LOCK.open("a") if args.phase == "performance" else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for name in args.models:
            model_scorer = scorer(args, name)
            variant = "BF16" if name == "lite" else "Q4_K_M"
            target = f"ggml-org/vllm-d3-{name}-GGUF:{variant}"
            result = {"cases": [], "samples": []}
            report["models"][name] = result
            server = Splash(args, name, args.output)
            result["command"] = server.command
            standalone = None
            try:
                if args.reference_mode == "llama":
                    standalone = Llama(args, name, args.output)
                    result["reference_command"] = standalone.command
                    result["reference_properties"] = standalone.properties
                reference_url = standalone.url if standalone else args.lmstudio_url
                result["status_before"] = call(server.url, "/status")
                if result["status_before"]["identity"]["kv"]["format"] != "q4h":
                    raise AssertionError("Q4H was not active")
                if args.phase == "correctness":
                    for length in sorted({128, 129, 2048, 2049, *args.lengths}):
                        body = case(model_scorer, length, length)
                        ref = (
                            request(reference_url, key, "vllm-d3-" + name, body, length)
                            if compare
                            else None
                        )
                        actual = request(server.url, None, target, body, length)
                        repeated = request(server.url, None, target, body, length)
                        if ref is not None:
                            same_decision(actual, ref)
                        for qid, answer in actual["answers"].items():
                            for label, p in answer["probabilities"].items():
                                if abs(
                                    p - repeated["answers"][qid]["probabilities"][label]
                                ) > 1e-3 * max(1, abs(p)):
                                    raise AssertionError(
                                        "repeated request exceeded existing 1e-3 bound"
                                    )
                        result["cases"].append(
                            {
                                "input_tokens": length,
                                "body": body,
                                reference_name: ref,
                                "splash": actual,
                                "repeated": repeated,
                            }
                        )
                        save()
                        print("CORRECT", name, length, flush=True)
                    bodies = [case(model_scorer, 4096, 100 + i) for i in range(4)]
                    serial = [
                        request(server.url, None, target, body, 4096) for body in bodies
                    ]
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        parallel = list(
                            pool.map(
                                lambda body: request(
                                    server.url, None, target, body, 4096
                                ),
                                bodies,
                            )
                        )
                    for a, b in zip(serial, parallel):
                        for qid in a["answers"]:
                            for label, p in a["answers"][qid]["probabilities"].items():
                                if abs(
                                    p - b["answers"][qid]["probabilities"][label]
                                ) > 1e-3 * max(1, abs(p)):
                                    raise AssertionError(
                                        "concurrent requests exceeded existing 1e-3 bound"
                                    )
                    result["concurrency_four"] = {
                        "serial": serial,
                        "parallel": parallel,
                    }
                    body = {
                        "state": {
                            "request": "charged twice; refund the extra charge",
                            "备注": "请退款 <已付款>",
                        },
                        "questions": {
                            "team": {
                                "type": "choice",
                                "instructions": "Choose the responsible team.",
                                "criteria": {
                                    "shipping": "Delivery",
                                    "billing": "Refunds",
                                },
                            },
                            "refund": {
                                "type": "noul",
                                "instructions": "Is a refund requested?",
                            },
                            "priority": {
                                "type": "score",
                                "instructions": "Rate the urgency.",
                                "criteria": ["routine", "urgent", "emergency"],
                            },
                        },
                    }
                    tokens = count_tokens(model_scorer, body)
                    actual = request(server.url, None, target, body, tokens)
                    ref = (
                        request(reference_url, key, "vllm-d3-" + name, body, tokens)
                        if compare
                        else None
                    )
                    if ref is not None:
                        same_decision(actual, ref)
                    result["question_types"] = {
                        "body": body,
                        "splash": actual,
                        reference_name: ref,
                    }
                    print("TYPES", name, "choice/noul/score", flush=True)
                else:
                    reference = standalone or (
                        Reference(args, name) if compare else None
                    )
                    other = (
                        Reference(args, "mini" if name == "lite" else "lite")
                        if args.reference_mode == "lmstudio"
                        else None
                    )
                    owned = [os.getpid(), *server.pids]
                    if compare:
                        owned.extend(reference.pids)
                        if other:
                            owned.extend(other.pids)
                    for length in args.lengths:
                        warm = case(model_scorer, length, -1)
                        for _ in range(2):
                            if compare:
                                request(
                                    reference_url,
                                    key,
                                    "vllm-d3-" + name,
                                    warm,
                                    length,
                                )
                            request(server.url, None, target, warm, length)
                        for round_number in range(args.rounds):
                            body = case(model_scorer, length, round_number)
                            pair = {}
                            for engine in (
                                ["splash"]
                                if not compare
                                else [reference_name, "splash"]
                                if round_number % 2 == 0
                                else ["splash", reference_name]
                            ):
                                measured = (
                                    reference if engine == reference_name else server
                                )
                                wait_for_device(
                                    owned, report["isolation"], args.accept_desktop_load
                                )
                                before = measured.memory()
                                host_before = host_memory()
                                with MemorySampler(measured) as sampler:
                                    start = time.perf_counter()
                                    response = request(
                                        reference_url
                                        if engine == reference_name
                                        else server.url,
                                        key if engine == reference_name else None,
                                        "vllm-d3-" + name
                                        if engine == reference_name
                                        else target,
                                        body,
                                        length,
                                    )
                                    seconds = time.perf_counter() - start
                                if sampler.error:
                                    raise RuntimeError(sampler.error)
                                row = {
                                    "engine": engine,
                                    "round": round_number + 1,
                                    "input_tokens": length,
                                    "seconds": seconds,
                                    "input_tokens_per_second": length / seconds,
                                    "memory_before": before,
                                    "memory_after": measured.memory(),
                                    "host_memory_before": host_before,
                                    "host_memory_after": host_memory(),
                                    "sampled_peak_footprint_bytes": max(
                                        sampler.peak,
                                        before["total_physical_footprint_bytes"],
                                    ),
                                    "response": response,
                                    "isolation_after": environment(owned),
                                }
                                if row["isolation_after"]["busy_pids"]:
                                    raise RuntimeError(
                                        "external inference overlapped the performance sample"
                                    )
                                result["samples"].append(row)
                                pair[engine] = response
                                save()
                                print(
                                    "SAMPLE",
                                    name,
                                    length,
                                    round_number + 1,
                                    engine,
                                    round(seconds, 4),
                                    flush=True,
                                )
                            if compare:
                                same_decision(pair["splash"], pair[reference_name])
                    result["summary"] = [
                        {
                            "engine": engine,
                            "input_tokens": length,
                            "mean_seconds": statistics.mean(
                                r["seconds"]
                                for r in result["samples"]
                                if r["engine"] == engine and r["input_tokens"] == length
                            ),
                            "mean_tokens_per_second": statistics.mean(
                                r["input_tokens_per_second"]
                                for r in result["samples"]
                                if r["engine"] == engine and r["input_tokens"] == length
                            ),
                        }
                        for length in args.lengths
                        for engine in (
                            [reference_name, "splash"] if compare else ["splash"]
                        )
                    ]
                result["status_after"] = call(server.url, "/status")
            except BaseException as error:
                result["request_error"] = repr(error)
                raise
            finally:
                try:
                    if standalone:
                        try:
                            standalone.close()
                        except BaseException as error:
                            result["llama_cleanup_error"] = repr(error)
                            raise
                        finally:
                            result["llama_exit"] = standalone.process.returncode
                finally:
                    server.close()
                    result["splash_exit"] = server.process.returncode
                    save()
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()
        if lock:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--binary", type=Path, default=Path("build/splash").absolute())
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--lmstudio-lite-port", type=int)
    parser.add_argument("--lmstudio-mini-port", type=int)
    parser.add_argument("--lmstudio-key-file", type=Path)
    parser.add_argument(
        "--reference-binary",
        type=Path,
        default=Path.home()
        / ".lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.60.0/llama-server",
    )
    parser.add_argument(
        "--reference-mode", choices=["lmstudio", "llama", "none"], default="lmstudio"
    )
    parser.add_argument("--correctness-report", type=Path)
    parser.add_argument(
        "--phase", choices=["correctness", "performance"], required=True
    )
    parser.add_argument(
        "--models", nargs="+", choices=["lite", "mini"], default=["lite", "mini"]
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--accept-desktop-load", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
