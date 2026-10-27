"""Cold, fixed-quota expert/PLE trials; never changes startup defaults.

Use an exact 32768-token chat request with 512 actual output tokens. Each
quota starts a fresh engine, runs the same native warmup, and makes one cold
request. Repeat selected quotas in ABBA order with a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dev.benchmarks import abba
from dev.benchmarks import qwen4b_ffn_cache as isolation
from dev.tests import smoke_real as smoke

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024**2
POOL = 10 * 1024**3


def write(path, value):
    isolation.write(path, value)


def frozen_inputs(args, request_path):
    files = {Path(__file__).resolve(), request_path, args.binary}
    for directory in ("runtime", "server", "install"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in (
                ".py",
                ".cpp",
                ".hpp",
                ".h",
                ".mm",
                ".metal",
            ):
                files.add(path)
    for name in ("splash.metallib", "host-step.metallib", "build-identity.json"):
        files.add(args.binary.parent / name)
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def log(*values):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *values, flush=True)


def allocation(status, ple):
    smoke.validate_status(status, "q4h")
    pool = status["memory_plan"]["budget"]["expert_ple_pool"]
    expected = dict(
        limit_bytes=POOL, expert_planned_bytes=POOL - ple, ple_planned_bytes=ple
    )
    if pool != expected:
        raise ValueError(f"requested quota was changed: {expected!r} -> {pool!r}")
    actual = status["expert_ple_adaptive"]
    if actual["enabled"] or actual["state"] != "off" or actual["failure"]:
        raise ValueError("online quota adjustment must be disabled")
    expert, table = status["streamed_experts"], status["streamed_table"]
    if (
        not expert
        or expert["failure"]
        or not table
        or table["poisoned"]
        or table["read_errors"]
    ):
        raise ValueError("streamed store failed")
    e, p = actual["expert_allocated_bytes"], actual["ple_committed_bytes"]
    if not (0 < e <= POOL - ple and 0 < p <= ple and e + p <= POOL):
        raise ValueError("actual store allocations exceed their quotas")
    if not status["memory_audit"]["valid"]:
        raise ValueError("runtime memory audit failed")
    features = status["startup_features"]
    if (
        features["drafter"] != "dflash2"
        or features["draft_block_size"] != 8
        or features["prefill_token_budget"] != 2048
        or features["ffn_cache_bytes"] != 0
        or any(
            features["optional"][key]
            for key in ("prompt_lookup", "fast_topk", "compact_draft_head")
        )
    ):
        raise ValueError("fixed feature settings changed")
    return dict(
        expert_quota_bytes=POOL - ple,
        ple_quota_bytes=ple,
        expert_allocated_bytes=e,
        expert_arena_bytes=expert["arena_bytes"],
        expert_staging_bytes=expert["staging_bytes"],
        expert_slots=expert["slots"],
        ple_committed_bytes=p,
        ple_capacity_rows=table["cache_capacity_rows"],
        pool_actual_bytes=e + p,
    )


def counters(before, after):
    result = {}
    for name, key in (("experts", "streamed_experts"), ("ple", "streamed_table")):
        a, b = before[key], after[key]
        phases = {}
        for phase in ("prefill", "decode"):
            keys = (
                ("routed", "read", "prefetched", "read_ahead", "ghost_hits")
                if name == "experts"
                else (
                    "requested_rows",
                    "unique_rows",
                    "hit_rows",
                    "read_rows",
                    "reused_evicted_rows",
                    "consumed_prefetched_rows",
                )
            )
            keys += ("read_wait_ms",)
            if name == "experts":
                keys += ("read_bytes",)
            delta = {k: b[phase][k] - a[phase][k] for k in keys}
            if any(v < 0 for v in delta.values()):
                raise ValueError("store counters reset during a fixed-quota request")
            if name == "experts":
                delta["hits"] = delta["routed"] - delta["read"]
            phases[phase] = delta
        result[name] = phases
        if name == "ple":
            result[name].update(
                {k: b[k] - a[k] for k in ("evictions", "read_bytes", "read_errors")}
            )
    # The public expert telemetry exposes bounded ghost-hit rereads, not a
    # complete eviction/reread counter. Do not invent it from misses.
    result["experts"]["evictions"] = None
    result["experts"]["all_rereads"] = None
    result["experts"]["counter_limit"] = "ghost_hits covers bounded eviction history"
    return result


def validate_response(response, after, reference=None):
    usage = response["usage"]
    if usage["prompt_tokens"] != 32768 or usage["completion_tokens"] != 512:
        raise ValueError("actual input/output token count differs")
    if response["metrics"]["cache"]["matched_tokens"] != 0:
        raise ValueError("prefix reuse skipped prefill")
    message = response["choices"][0]["message"]
    if not message.get("content"):
        raise ValueError("empty output")
    if reference is not None and message != reference:
        raise ValueError("output differs from the fixed-quota reference")
    if after["state"]["active_lanes"] or after["kv"]["pages_active"]:
        raise ValueError("request resources remained active")
    if any(after["scheduler"][k] for k in ("queued", "prefilling", "decoding")):
        raise ValueError("scheduler did not become idle")
    return message


def paired_verdict(rows):
    """Four equal-work trials in A/B/B/A order; no win inside the noise."""
    quotas = [r["ple_mib"] for r in rows]
    if len(rows) != 4 or quotas[0] != quotas[3] or quotas[1] != quotas[2]:
        raise ValueError("expected four ABBA quota trials")
    if quotas[0] == quotas[1] or any(r["interference"] for r in rows):
        raise ValueError("ABBA needs distinct quotas and uncontaminated trials")
    verdict = abba.compare(*(r["wall_seconds"] for r in rows))
    if verdict["noise"] > abba.NOISE_LIMIT:
        winner = None
    elif abs(verdict["regression"]) <= verdict["bound"]:
        winner = None
    else:
        winner = quotas[1] if verdict["regression"] < 0 else quotas[0]
    return dict(verdict, winner_ple_mib=winner, keep_baseline=winner is None)


def owned_pids(pid):
    lines = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid="], text=True, timeout=5
    ).splitlines()
    pairs = [tuple(map(int, line.split())) for line in lines]
    result = {pid}
    while children := {child for child, parent in pairs if parent in result} - result:
        result |= children
    return result


def environment(pid=None):
    return isolation.environment(exclude=() if pid is None else owned_pids(pid))


def wait_idle(folder, pid=None):
    consecutive = 0
    while consecutive < 3:
        sample = environment(pid)
        with (folder / "idle.jsonl").open("a") as output:
            output.write(json.dumps(sample) + "\n")
        idle = not sample["busy_pids"] and sample["gpu_utilization_percent"] is not None
        idle = idle and sample["gpu_utilization_percent"] <= 50
        consecutive = consecutive + 1 if idle else 0
        log("IDLE" if idle else "WAIT_IDLE", sample)
        if consecutive < 3:
            time.sleep(5)


def file_cache_snapshot():
    return {
        "vm_stat": subprocess.check_output(["vm_stat"], text=True, timeout=5),
        "swap": subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True, timeout=5
        ),
        "note": "No global cache purge; streamed weight readers use F_NOCACHE.",
    }


def command(args, port, ple):
    return [
        sys.executable,
        "-u",
        "-m",
        "server.server",
        str(args.model_root),
        "--binary",
        str(args.binary),
        "--model",
        args.request["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "--max-context",
        "262144",
        "--max-memory",
        "auto",
        "--sparse-kv-memory",
        "4G",
        "--max-cache-disk",
        "50G",
        "--queue-size",
        "1",
        "--kv-format",
        "q4h",
        "--verify-rows",
        "all",
        "--disable-ane",
        "--prompt-lookup",
        "off",
        "--fast-topk",
        "off",
        "--compact-draft-head",
        "off",
        "--experimental-quantized-ffn",
        "off",
        "--expert-ple-memory",
        str(POOL),
        "--expert-ple-policy",
        "static",
        "--expert-memory",
        str(POOL - ple),
        "--ple-memory",
        str(ple),
    ]


def run_trial(args, index, ple_mib, reference):
    folder = args.output / f"{index:02d}-ple{ple_mib}"
    folder.mkdir()
    wait_idle(folder)
    ple = ple_mib * MIB
    port = smoke.available_port()
    argv = command(args, port, ple)
    write(folder / "command.json", argv)
    write(folder / "file-cache-before.json", file_cache_snapshot())
    env = dict(os.environ)
    env.pop("SPLASH_API_KEY", None)
    env.update(MTL_SHADER_VALIDATION="0", MTL_DEBUG_LAYER="1")
    halt = threading.Event()
    samples, errors = [], []
    with (folder / "server.log").open("w") as output:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        write(folder / "process.json", {"pid": process.pid, "port": port})
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"startup exited {process.returncode}: {folder}")
                try:
                    code, before = smoke.request(port, "GET", "/status", timeout=3)
                    if code == 200 and before.get("ready"):
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("startup exceeded existing 1800-second limit")
            allocated = allocation(before, ple)
            write(folder / "ready.json", before)
            if not all(before["warmup"][k] for k in ("prefill_2048", "decode_b1")):
                raise ValueError("native warmup incomplete")
            wait_idle(folder, process.pid)
            # Native warmup is the only warmup. No preceding HTTP request can
            # populate the engine prefix cache or alter the streamed caches.
            code, before = smoke.request(port, "GET", "/status", timeout=20)
            if code != 200:
                raise ValueError("status unavailable before measurement")
            write(folder / "before.json", before)
            allocation(before, ple)
            write(folder / "request.json", args.request)

            def observe():
                while not halt.wait(3):
                    try:
                        code, status = smoke.request(port, "GET", "/status", timeout=20)
                        if code != 200 or status.get("transport", {}).get(
                            "status_stale"
                        ):
                            raise ValueError("observer status unavailable/stale")
                        if allocation(status, ple) != allocated:
                            raise ValueError("fixed physical allocation changed")
                        sample = dict(
                            environment=environment(process.pid),
                            pressure=status["memory_governor"]["system_pressure"],
                            thermal=status["thermal_state"],
                            memory=status["memory_actual"],
                        )
                        samples.append(sample)
                        with (folder / "samples.jsonl").open("a") as stream:
                            stream.write(json.dumps(sample) + "\n")
                        log(
                            "SAMPLE",
                            index,
                            ple_mib,
                            sample["pressure"],
                            sample["memory"],
                        )
                    except Exception as error:
                        errors.append(repr(error))
                        return

            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            log("RUN", index, "PLE_MiB", ple_mib, allocated)
            started = time.monotonic()
            try:
                code, response = smoke.request(
                    port, "POST", "/v1/chat/completions", args.request, timeout=3600
                )
                elapsed = time.monotonic() - started
            finally:
                halt.set()
                observer.join(30)
            write(folder / "response.json", response)
            if code != 200:
                raise ValueError(f"generation HTTP {code}")
            code, after = smoke.request(port, "GET", "/status", timeout=20)
            write(folder / "after.json", after)
            if code != 200 or errors or observer.is_alive():
                raise ValueError(f"status/observer failure: {errors}")
            if allocation(after, ple) != allocated:
                raise ValueError("fixed physical allocation changed")
            message = validate_response(response, after, reference)
            if after["identity"] != before["identity"]:
                raise ValueError("runtime identity changed")
            if after["memory_plan"]["budget"] != before["memory_plan"]["budget"]:
                raise ValueError("fixed plan/KV budget changed")
            metrics = after["metrics"]
            delta = {
                k: metrics[k] - before["metrics"][k]
                for k in (
                    "prefill_input_tokens",
                    "prefill_wall_ms",
                    "decode_output_tokens",
                    "decode_cycle_ms",
                    "decode_wall_ms",
                )
            }
            if delta["prefill_input_tokens"] != 32768:
                raise ValueError("native prefill did not process the entire prompt")
            interference = [s for s in samples if s["environment"]["busy_pids"]]
            row = dict(
                index=index,
                ple_mib=ple_mib,
                allocation=allocated,
                prefill_tok_s=1000
                * delta["prefill_input_tokens"]
                / delta["prefill_wall_ms"],
                decode_tok_s=1000
                * delta["decode_output_tokens"]
                / delta["decode_cycle_ms"],
                decode_stream_tok_s=response["metrics"]["request_latency"][
                    "stream_tokens_per_second"
                ],
                wall_seconds=elapsed,
                native_delta=delta,
                store_delta=counters(before, after),
                interference=interference,
                pressures=sorted({s["pressure"] for s in samples}),
                metal_peak_bytes=after["memory_actual"]["peak_bytes"],
                prefill_partitions=dict(
                    batches=after["scheduler"]["prefill_batches"]
                    - before["scheduler"]["prefill_batches"],
                    cuts=after["cache"]["capture_cuts"]
                    - before["cache"]["capture_cuts"],
                ),
                identity=after["identity"],
                correctness="exact_message_match",
                message_sha256=hashlib.sha256(
                    json.dumps(message, sort_keys=True).encode()
                ).hexdigest(),
            )
            write(folder / "result.json", row)
            write(folder / "file-cache-after.json", file_cache_snapshot())
            log("RESULT", json.dumps(row))
            return row, message
        finally:
            failed = sys.exc_info()[0] is not None
            halt.set()
            forced = False
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(60)
                except subprocess.TimeoutExpired:
                    forced = True
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(10)
            write(
                folder / "cleanup.json",
                dict(exit_code=process.returncode, forced=forced),
            )
            if (process.returncode != 0 or forced) and not failed:
                raise RuntimeError(f"unclean engine shutdown: {process.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ple-mib", nargs="+", type=int, default=[512, 256, 1024])
    parser.add_argument(
        "--reference",
        type=Path,
        help="retained response.json for exact output comparison",
    )
    parser.add_argument("--run", action="store_true", help="actually launch engines")
    args = parser.parse_args()
    if any(not 0 < p * MIB < POOL for p in args.ple_mib):
        parser.error("PLE quota must be positive and below 10 GiB")
    request_path = args.request.resolve()
    args.request = json.loads(request_path.read_text())
    if not (
        args.request["max_completion_tokens"] == 512
        and args.request.get("ignore_eos") is True
    ):
        parser.error("request must fix 512 actual output tokens")
    args.binary = args.binary.resolve()
    args.model_root = args.model_root.resolve()
    plan = dict(
        pool_bytes=POOL,
        ple_mib=args.ple_mib,
        input_tokens=32768,
        request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dynamic=False,
        startup_integration=False,
        validation=dict(MTL_SHADER_VALIDATION="0", MTL_DEBUG_LAYER="1"),
    )
    frozen = frozen_inputs(args, request_path)
    plan["frozen_inputs"] = frozen
    plan["model_binding"] = json.loads((args.model_root / "model.json").read_text())
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    reference = (
        None
        if args.reference is None
        else json.loads(args.reference.read_text())["choices"][0]["message"]
    )
    rows = []
    with isolation.device_lock():
        for i, ple in enumerate(args.ple_mib, 1):
            if frozen_inputs(args, request_path) != frozen:
                raise ValueError("source, request or build changed")
            row, message = run_trial(args, i, ple, reference)
            if frozen_inputs(args, request_path) != frozen:
                raise ValueError("source, request or build changed during trial")
            reference = message if reference is None else reference
            if rows and (
                row["identity"] != rows[0]["identity"]
                or row["prefill_partitions"] != rows[0]["prefill_partitions"]
            ):
                raise ValueError("model/build identity or prefill partitions changed")
            rows.append(row)
            write(args.output / "results.json", rows)
        if len(rows) == 4 and args.ple_mib[0] == args.ple_mib[-1]:
            write(args.output / "abba.json", paired_verdict(rows))
    write(args.output / "exit.json", {"exit_code": 0})


if __name__ == "__main__":
    main()
