"""Isolated Qwen3.5-4B FFN cache experiment; default only prints a plan.

Prepare, correctness and performance are separate explicit actions. The native
operator check precedes end-to-end checks; a token divergence blocks timing.
Raw stdout/stderr and results are retained, including invalid attempts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import re
import signal
import statistics
import subprocess
import tempfile
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

from dev.benchmarks import qwen4b_matrix as matrix
from dev.benchmarks import qwen4b_matrix_adapter as adapter

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ROOT / "build/ffn-cache-variants"
DEVICE_LOCK = Path(tempfile.gettempdir()) / "splash-qwen4b-ffn-device.lock"
MODES = ("full_fp16", "quantized", "limited_fp16")
PROFILES = ("cache", "m64n32", "fused_gate_up", "q4k_group_scale", "limited_cache")
CATEGORIES = (
    "target_source_weights_bytes",
    "ffn_fp16_cache_bytes",
    "draft_weights_bytes",
    "vision_weights_bytes",
    "state_bytes",
    "prefill_workspace_bytes",
    "decode_workspace_bytes",
    "kv_bytes",
    "state_staging_bytes",
    "ane_bytes",
    "unclassified_bytes",
)


def validate_memory(report, mode, budget=0):
    matrix.require(mode in MODES, "invalid FFN mode")
    for key in (*CATEGORIES, "backend_current_bytes", "backend_peak_bytes"):
        matrix.require(
            type(report.get(key)) is int and report[key] >= 0,
            f"missing memory category: {key}",
        )
    matrix.require(
        sum(report[k] for k in CATEGORIES) == report["backend_current_bytes"],
        "memory ledger does not reconcile",
    )
    matrix.require(
        0 < report["backend_current_bytes"] <= report["backend_peak_bytes"],
        "invalid memory peak",
    )
    if mode == "quantized":
        matrix.require(
            report["ffn_fp16_cache_bytes"] == 0
            and report.get("ffn_cached_matrices") == 0,
            "quantized mode retained resident FP16 FFN weights",
        )
    elif mode == "limited_fp16":
        matrix.require(
            type(budget) is int and budget > 0, "limited cache requires a budget"
        )
        matrix.require(
            report["ffn_fp16_cache_bytes"] <= budget
            and 0 <= report.get("ffn_cached_matrices", -1) <= 96,
            "limited cache exceeded budget",
        )
        matrix.require(
            (report["ffn_fp16_cache_bytes"] == 0)
            == (report["ffn_cached_matrices"] == 0),
            "cache byte/count disagreement",
        )
    else:
        matrix.require(
            report["ffn_fp16_cache_bytes"] > 0
            and report.get("ffn_cached_matrices") == 96,
            "full mode did not cache all 96 FFN matrices",
        )


def first_divergence(full, quantized):
    for i in range(max(len(full), len(quantized))):
        a = full[i] if i < len(full) else None
        b = quantized[i] if i < len(quantized) else None
        if a != b:
            return {"index": i, "full_fp16": a, "quantized": b}
    return None


def parameters(mode, output=512, input_tokens=32768):
    return next(matrix.groups(matrix.read_json(matrix.MANIFEST)))["parameters"] | {
        "kv_format": "q4h",
        "ane_requested": True,
        "ane_share": 0.5,
        "ane_minimum_rows": 512,
        "drafter_enabled": True,
        "drafter_block_size": 8,
        "tail_prefill_splits": 16,
        "concurrency": 1,
        "input_tokens": input_tokens,
        "output_tokens": output,
        "ffn_weight_mode": mode,
        "ffn_prefill_variant": "baseline",
        "ffn_cache_budget_bytes": 0,
    }


def configurations(profile="cache", budget=1 << 30):
    matrix.require(profile in PROFILES, "unknown FFN profile")
    a, b = parameters("quantized"), parameters("quantized")
    if profile == "cache":
        a = parameters("full_fp16")
    elif profile == "limited_cache":
        matrix.require(
            type(budget) is int and budget > 0, "cache budget must be positive"
        )
        b |= {"ffn_weight_mode": "limited_fp16", "ffn_cache_budget_bytes": budget}
    elif profile == "fused_gate_up":
        a["ffn_prefill_variant"] = "m64n32"
        b["ffn_prefill_variant"] = "fused_m64n32"
    else:
        b["ffn_prefill_variant"] = profile
    return {"A": a, "B": b}


def jobs(profile="cache", budget=1 << 30):
    configs = configurations(profile, budget)
    result = []
    for output in (512, 2048):
        for prompt in range(3):
            group = f"out-{output}/prompt-{prompt}"
            for position, arm in enumerate("ABBA"):
                p = configs[arm] | {"output_tokens": output}
                result.append(
                    {
                        "group_id": matrix.digest(p),
                        "repeat": prompt,
                        "position": position,
                        "arm": arm,
                        "abba_group": group,
                        "sample_id": f"{group}/position-{position}",
                        "parameters": p,
                    }
                )
    return result


def write(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def device_lock():
    # Keep the inode: unlinking a held lock would allow a second owner.
    with DEVICE_LOCK.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise matrix.MatrixError(
                "experiment device lock is held by another profile/worktree"
            ) from error
        yield


def environment(
    exclude=(), *, allow_inference_pid=None, allow_concurrent_inference=False
):
    """Benchmarks are busy while alive; services also check request activity."""
    records = []
    gpu = None
    accelerator = plistlib.loads(
        subprocess.check_output(["ioreg", "-r", "-c", "IOAccelerator", "-a"])
    )
    for device in accelerator:
        utilization = device.get("PerformanceStatistics", {}).get(
            "Device Utilization %"
        )
        if type(utilization) in (int, float):
            gpu = max(gpu or 0, utilization)
    output = subprocess.check_output(["ps", "-Ao", "pid=,pcpu=,comm="], text=True)
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid, cpu, command = int(parts[0]), float(parts[1]), parts[2]
        if pid == os.getpid() or pid in exclude:
            continue
        name = Path(command).name
        inference = "llama-server" in name
        native = name in {
            "splash",
            "backend-benchmark",
            "model-runtime-oracle",
            "qwen4b-matrix-native",
            "qwen4b-ffn-check",
            "gguf-projection-benchmark",
            "gguf-moe-benchmark",
            "decode-profile",
            "attention-sweep",
            "qwen4b-operator-candidates",
            "drafter-fc-precision-benchmark",
        }
        if not inference and not native:
            continue
        processing = None
        if inference:
            args = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            port = re.search(r"--port\s+(\d+)", args)
            if port:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port[1]}/slots", timeout=0.5
                    ) as response:
                        slots = json.load(response)
                    if isinstance(slots, list):
                        processing = any(
                            isinstance(s, dict) and s.get("is_processing") is True
                            for s in slots
                        )
                except (OSError, ValueError, TypeError):
                    pass
        records.append(
            {
                "pid": pid,
                "cpu_percent": cpu,
                "processing": processing,
                "authorized_concurrent_inference": inference
                and (allow_concurrent_inference or pid == allow_inference_pid),
                "busy": (native and name != "splash") or processing is True or cpu >= 5,
            }
        )
    return {
        "gpu_utilization_percent": gpu,
        "servers": records,
        "busy_pids": [
            r["pid"]
            for r in records
            if r["busy"] and not r["authorized_concurrent_inference"]
        ],
    }


def require_idle(
    *,
    ignore_gpu_utilization=False,
    allow_inference_pid=None,
    allow_concurrent_inference=False,
):
    samples = []
    for i in range(2):
        snapshot = (
            environment()
            if allow_inference_pid is None and not allow_concurrent_inference
            else environment(
                allow_inference_pid=allow_inference_pid,
                allow_concurrent_inference=allow_concurrent_inference,
            )
        )
        matrix.require(
            not snapshot["busy_pids"],
            f"external inference active: {snapshot['busy_pids']}",
        )
        matrix.require(
            ignore_gpu_utilization
            or (
                snapshot["gpu_utilization_percent"] is not None
                and snapshot["gpu_utilization_percent"] <= 50
            ),
            "GPU not idle enough for an isolated experiment",
        )
        samples.append(snapshot)
        if i == 0:
            time.sleep(2)
    return samples


def checked_process(
    command,
    input_text,
    stdout,
    stderr,
    timeout_seconds=1800,
    poll_seconds=2,
    correctness=False,
    allow_inference_pid=None,
    allow_concurrent_inference=False,
):
    before, observed, authorized_observed, process = [], [], [], None
    status, error, started = "failed", None, time.monotonic()
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminated(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminated)
    try:
        with device_lock():
            matrix.require(
                correctness
                or (allow_inference_pid is None and not allow_concurrent_inference),
                "parallel inference is correctness-only",
            )
            before = require_idle(
                ignore_gpu_utilization=correctness,
                allow_inference_pid=allow_inference_pid,
                allow_concurrent_inference=allow_concurrent_inference,
            )
            with stdout.open("x") as out, stderr.open("x") as err:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE
                    if input_text is not None
                    else subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    text=True,
                    start_new_session=True,
                )
                write(
                    stdout.parent / "process.json",
                    {
                        "pid": process.pid,
                        "process_group": process.pid,
                        "command": command,
                    },
                )
                if input_text is not None:
                    process.stdin.write(input_text)
                    process.stdin.close()
                while process.poll() is None:
                    snapshot = (
                        environment((process.pid,))
                        if allow_inference_pid is None
                        and not allow_concurrent_inference
                        else environment(
                            (process.pid,),
                            allow_inference_pid=allow_inference_pid,
                            allow_concurrent_inference=allow_concurrent_inference,
                        )
                    )
                    if any(
                        r.get("authorized_concurrent_inference") and r["busy"]
                        for r in snapshot.get("servers", [])
                    ):
                        authorized_observed.append(snapshot)
                    if snapshot["busy_pids"]:
                        observed.append(snapshot)
                    if time.monotonic() - started > timeout_seconds:
                        status = "timed_out"
                        raise TimeoutError("experiment process timeout")
                    time.sleep(poll_seconds)
                status = (
                    "complete"
                    if process.returncode == 0 and not observed
                    else "invalid"
                )
    except (Exception, KeyboardInterrupt) as failure:
        error = str(failure) or type(failure).__name__
        if isinstance(failure, KeyboardInterrupt):
            status = "cancelled"
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return {
        "command": command,
        "status": status,
        "error": error,
        "exit_code": process.returncode if process is not None else None,
        "environment_before": before,
        "external_workload_observed": observed,
        "authorized_inference_pid": allow_inference_pid,
        "authorized_all_inference_services": allow_concurrent_inference,
        "authorized_concurrent_inference_observed": authorized_observed,
        "process_wall_seconds": time.monotonic() - started,
        "performance_usable": status == "complete" and not correctness,
    }


def prepared(args, rehash=False):
    document = matrix.read_json(args.output / "prepared.json")
    matrix.require(
        document["code_sha256"] == adapter.code_sha256(ROOT),
        "experiment source changed; prepare again",
    )
    for record in document["artifacts"].values():
        adapter.verify_record(record, rehash=rehash)
    for record in document["model_records"].values():
        adapter.verify_record(record)
    matrix.require(
        document.get("profile", "cache") == getattr(args, "profile", "cache"),
        "profile changed; prepare a new directory",
    )
    matrix.require(
        document.get("schedule_sha256", matrix.digest(jobs()))
        == matrix.digest(
            jobs(
                getattr(args, "profile", "cache"),
                getattr(args, "cache_budget_bytes", 1 << 30),
            )
        ),
        "schedule changed",
    )
    return document


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = adapter.stamp_variants(
        ROOT, VARIANTS, args.model_root, args.off_model_root
    )
    matrix.require(8 in result["stamped_block_sizes"], f"block8 not built: {result}")
    tool = VARIANTS / "block-8/engine-tests/qwen4b-ffn-check"
    matrix.require(tool.is_file(), "build qwen4b-ffn-check first")
    stamp = matrix.read_json(VARIANTS / "block-8/matrix-identity.json")
    matrix.require(
        Path(str(tool) + ".config").read_text().strip()
        == stamp["configuration"]["test"],
        "numerical tool build flags differ",
    )
    dependencies = adapter.source_inputs(ROOT)
    matrix.require(
        tool.stat().st_mtime_ns
        >= max(
            (ROOT / p).stat().st_mtime_ns
            for p in dependencies
            if p.startswith("runtime/")
            or p
            in (
                "dev/benchmarks/qwen4b_ffn_check.mm",
                "dev/tests/engine/GgufFormatReference.hpp",
                "dev/tests/engine/LinearNumerics.hpp",
            )
        ),
        "numerical tool predates source/header",
    )
    adapter.prepare_inputs(args.model_root, args.output / "inputs.json")
    document = {
        "code_sha256": adapter.code_sha256(ROOT),
        "profile": args.profile,
        "schedule_sha256": matrix.digest(jobs(args.profile, args.cache_budget_bytes)),
        "identity": {
            "code_sha256": stamp["code_sha256"],
            "weight_sha256": adapter.role_hashes(stamp["models"]["standard"]),
        },
        "model_records": stamp["models"]["standard"],
        "artifacts": stamp["artifact_sha256"]
        | {
            "numerical_check": adapter.file_record(tool),
            "inputs": adapter.file_record(args.output / "inputs.json"),
        }
        | {
            name: adapter.file_record(args.model_root / name)
            for name in (
                "model.json",
                "config.json",
                "draft/config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            )
        },
    }
    write(args.output / "prepared.json", document)
    write(
        args.output / "plan.json",
        {
            "jobs": jobs(args.profile, args.cache_budget_bytes),
            "samples": 24,
            "ordering": "ABBA for each input/output pair",
            "prompts": 3,
            "true_repeats_per_mode_per_prompt": 2,
            "timings_exclude": "model load, ANE setup and warmup",
            "memory_peak_scope": "backend-owned Metal resources since startup; category ledger is current allocation",
        },
    )


def validate_native(raw, args, document, job, prompts, mode):
    p = job["parameters"]
    variant = VARIANTS / "block-8"
    matrix.require(
        raw.get("compiled_source_sha256")
        == adapter.build_identity.build_id(
            ROOT,
            constants=matrix.read_json(variant / "matrix-identity.json")[
                "configuration"
            ]["constants"],
        ).removeprefix("src-"),
        "embedded production identity mismatch",
    )
    matrix.require(
        raw.get("actual_ffn_weight_mode") == p["ffn_weight_mode"],
        "actual FFN mode differs",
    )
    matrix.require(
        raw.get("actual_ffn_prefill_variant") == p["ffn_prefill_variant"],
        "FFN variant differs",
    )
    validate_memory(
        raw["memory_report"], p["ffn_weight_mode"], p["ffn_cache_budget_bytes"]
    )
    matrix.require(
        raw["memory_report"]["backend_peak_bytes"] == raw["memory_peak_bytes"],
        "memory peak differs between reports",
    )
    if mode == "correctness":
        adapter.validate_short(
            raw,
            {
                "job": job,
                "prompts": prompts,
                "ffn_long_correctness": p["input_tokens"] == 32768,
            },
        )
    checked = matrix.validate_batch(
        raw,
        job,
        prompts,
        document["identity"],
        adapter.eos_ids(args.model_root),
        performance=mode == "performance",
    )
    matrix.require(
        (
            checked["status"] == "complete"
            or (mode == "correctness" and p["input_tokens"] < 512)
        )
        and checked["facts_pass"],
        "batch correctness/ANE did not pass",
    )
    matrix.require(
        isinstance(raw.get("ffn_paths"), list) and raw["ffn_paths"],
        "submitted FFN execution paths missing",
    )
    return checked


def native(args, document, job, prompts, mode, label, *, long_correctness=False):
    variant = VARIANTS / "block-8"
    attempt = args.output / "attempts" / (label + "-" + uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True)
    payload = {
        "protocol": matrix.PROTOCOL,
        "operation": "run_batch",
        "mode": mode,
        "identity": document["identity"],
        "job": job,
        "prompts": prompts,
        "model_root": str(args.model_root.resolve()),
        "metallib": str((variant / "splash.metallib").resolve()),
    }
    if long_correctness:
        matrix.require(
            mode == "correctness", "long correctness cannot use performance mode"
        )
        payload["ffn_long_correctness"] = True
    write(attempt / "payload.json", payload)
    evidence = {
        "attempt_directory": str(attempt.resolve()),
        "sample_id": label,
        "job": job,
        "identity": document["identity"],
        "performance_usable": False,
    }
    raw = None
    try:
        prepared(args)
        evidence |= checked_process(
            [str(variant / "engine-tests/qwen4b-matrix-native")],
            json.dumps(payload),
            attempt / "stdout.json",
            attempt / "stderr.log",
            correctness=mode == "correctness",
            allow_inference_pid=getattr(args, "allow_inference_pid", None)
            if mode == "correctness"
            else None,
            allow_concurrent_inference=getattr(
                args, "allow_concurrent_inference", False
            )
            if mode == "correctness"
            else False,
        )
        matrix.require(
            evidence["status"] == "complete",
            f"invalid attempt retained: {label}: {evidence.get('error')}",
        )
        raw = matrix.read_json(attempt / "stdout.json")
        raw["commands"] = [evidence["command"]]
        raw["loaded_weight_sha256"] = document["identity"]["weight_sha256"]
        codec = adapter.tokenizer(args.model_root)
        for response in raw["requests"]:
            response["output_text"] = codec.decode(
                response["output_token_ids"], skip_special_tokens=False
            )
        raw["attempt_directory"] = str(attempt.resolve())
        raw["attempt_id"] = attempt.name
        evidence["raw_response"] = raw
        checked = validate_native(raw, args, document, job, prompts, mode)
        prepared(args)
        evidence["validation_status"] = "pass"
    except Exception as error:
        evidence |= {
            "validation_status": "failed",
            "error": str(error),
            "performance_usable": False,
        }
        raise
    finally:
        write(attempt / "evidence.json", evidence)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return raw, checked


def correctness(args):
    document = prepared(args, rehash=True)
    write(
        args.output / "correctness.json",
        {
            "status": "running",
            "identity": document["identity"],
            "profile": args.profile,
        },
    )
    try:
        tool = document["artifacts"]["numerical_check"]["path"]
        attempt = args.output / "attempts" / ("operators-" + uuid.uuid4().hex[:12])
        attempt.mkdir(parents=True)
        evidence = checked_process(
            [
                tool,
                document["artifacts"]["metallib"]["path"],
                str(args.model_root.resolve()),
                configurations(args.profile, args.cache_budget_bytes)["B"][
                    "ffn_prefill_variant"
                ],
            ],
            None,
            attempt / "operators.jsonl",
            attempt / "stderr.log",
            correctness=True,
            allow_inference_pid=getattr(args, "allow_inference_pid", None),
            allow_concurrent_inference=getattr(
                args, "allow_concurrent_inference", False
            ),
        )
        evidence |= {
            "attempt_directory": str(attempt.resolve()),
            "identity": document["identity"],
            "profile": args.profile,
        }
        write(attempt / "evidence.json", evidence)
        write(args.output / "operators-run.json", evidence)
        with (args.output / "operators-attempts.jsonl").open("a") as ledger:
            ledger.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
        if evidence["status"] != "complete":
            status = "failed"
            output = attempt / "operators.jsonl"
            if evidence["exit_code"] == 2 and output.is_file():
                try:
                    last = json.loads(output.read_text().splitlines()[-1])
                    if last.get("status") == "needs_weight_rounding_analysis":
                        status = last["status"]
                except (ValueError, IndexError):
                    pass
            write(
                args.output / "correctness.json",
                {
                    "status": status,
                    "identity": document["identity"],
                    "profile": args.profile,
                    "operator_attempt": str(attempt.resolve()),
                    "error": evidence.get("error"),
                },
            )
        matrix.require(
            evidence["status"] == "complete",
            "operator check failed/interfered; raw output retained",
        )
        operators = [
            json.loads(line)
            for line in (attempt / "operators.jsonl").read_text().splitlines()
        ]
        matrix.require(
            len(operators) == operators[-1].get("comparisons", 0) + 1
            and operators[-1].get("comparisons", 0) > 384
            and operators[-1].get("real_decode_covered") is True
            and operators[-1].get("ane_gpu_views_covered") is True
            and operators[-1].get("original_half_contract_pass") is True
            and operators[-1].get("status") == "complete",
            "operator check incomplete",
        )
        comparisons = []
        codec = adapter.tokenizer(args.model_root)
        configs = configurations(args.profile, args.cache_budget_bytes)
        for length in (64, 2048):
            base = adapter.short_payload(
                args.model_root, 8, document["identity"], concurrency=1, tail_splits=16
            )
            prompt = base["prompts"][0]
            ids = prompt["token_ids"]
            ids[ids.index(248046) : ids.index(248046)] = codec.encode(
                " ", add_special_tokens=False
            ) * (length - len(ids))
            prompt["input_sha256"] = matrix.digest(ids)
            outputs = {}
            for arm, config in configs.items():
                p = config | {"output_tokens": 32, "input_tokens": length}
                job = {"group_id": matrix.digest(p), "repeat": 0, "parameters": p}
                raw, _ = native(
                    args,
                    document,
                    job,
                    [prompt],
                    "correctness",
                    f"correctness-{length}-{arm}",
                )
                outputs[arm] = raw["requests"][0]["output_token_ids"]
                if length == 64:
                    matrix.require(
                        raw["ane_actual"]["enabled"] is False,
                        "ANE unexpectedly used below 512 rows",
                    )
            divergence = first_divergence(outputs["A"], outputs["B"])
            comparisons.append(
                {
                    "input_tokens": length,
                    "first_divergence": divergence,
                    "output_token_ids": outputs,
                }
            )
        passed = all(row["first_divergence"] is None for row in comparisons)
        write(
            args.output / "correctness.json",
            {
                "status": "pass" if passed else "needs_divergence_analysis",
                "identity": document["identity"],
                "profile": args.profile,
                "operator_comparisons": operators[-1]["comparisons"],
                "end_to_end": comparisons,
            },
        )
        matrix.require(
            passed, "first token divergence recorded; analyze it before performance"
        )
    except Exception as error:
        gate = matrix.read_json(args.output / "correctness.json")
        if gate["status"] == "running":
            write(
                args.output / "correctness.json",
                gate | {"status": "failed", "error": str(error)},
            )
        raise


def long_correctness(args):
    document = prepared(args, rehash=True)
    gate = matrix.read_json(args.output / "correctness.json")
    matrix.require(
        gate["status"] == "pass"
        and gate["identity"] == document["identity"]
        and gate.get("profile") == args.profile,
        "matching operator/short correctness gate required",
    )
    prompt = matrix.read_json(args.output / "inputs.json")["prompts"][0]
    configs = configurations(args.profile, args.cache_budget_bytes)
    comparisons, attempts = [], []
    path = args.output / "long-correctness.json"
    base = {
        "identity": document["identity"],
        "profile": args.profile,
        "comparisons": comparisons,
        "attempts": attempts,
        "performance_measured": False,
    }
    write(path, base | {"status": "running"})
    try:
        for output in (512, 2048):
            outputs = {}
            for arm, config in configs.items():
                p = config | {"input_tokens": 32768, "output_tokens": output}
                job = {"group_id": matrix.digest(p), "repeat": 0, "parameters": p}
                label = f"long-correctness-{output}-{arm}"
                write(path, base | {"status": "running", "current": label})
                print(f"START {label} profile={args.profile}", flush=True)
                raw, checked = native(
                    args,
                    document,
                    job,
                    [prompt],
                    "correctness",
                    label,
                    long_correctness=True,
                )
                outputs[arm] = raw["requests"][0]["output_token_ids"]
                matrix.require(
                    len(outputs[arm]) == output, "actual long output length differs"
                )
                attempts.append(
                    {
                        "label": label,
                        "directory": raw["attempt_directory"],
                        "validated": checked,
                    }
                )
                print(f"END {label} output_tokens={len(outputs[arm])}", flush=True)
            delta = first_divergence(outputs["A"], outputs["B"])
            comparisons.append(
                {
                    "input_tokens": 32768,
                    "output_tokens": output,
                    "first_divergence": None
                    if delta is None
                    else {
                        "index": delta["index"],
                        "A": delta["full_fp16"],
                        "B": delta["quantized"],
                    },
                    "output_token_ids": outputs,
                }
            )
            write(
                path,
                base
                | {
                    "status": "running"
                    if delta is None
                    else "needs_divergence_analysis"
                },
            )
            matrix.require(
                delta is None,
                "long token divergence recorded; subsequent length blocked",
            )
        write(path, base | {"status": "pass"})
    except Exception as error:
        current = matrix.read_json(path)
        if current["status"] == "running":
            write(path, current | {"status": "failed", "error": str(error)})
        raise


def output_gate(prior, record):
    tokens = record["output_token_ids"]
    # Same-configuration repeat is checked first to identify nondeterminism.
    ordered = sorted(prior, key=lambda r: r["job"]["arm"] != record["job"]["arm"])
    for old in ordered:
        delta = first_divergence(old["output_token_ids"], tokens)
        if delta is not None:
            return {
                "kind": "same_mode_repeat"
                if old["job"]["arm"] == record["job"]["arm"]
                else "cross_mode",
                "index": delta["index"],
                "previous_token": delta["full_fp16"],
                "current_token": delta["quantized"],
                "previous_sample": old["job"]["sample_id"],
                "current_sample": record["job"]["sample_id"],
            }
    return None


def summarize(records, identity, profile):
    rows = []

    def metric(record, key):
        request = record["validated"]["requests"][0]
        return (
            request[key]
            if key in ("prefill_tok_s", "decode_tok_s")
            else request["timings"][key]
        )

    keys = (
        "prefill_tok_s",
        "decode_tok_s",
        "prefill_wall_seconds",
        "decode_wall_seconds",
        "ttft_seconds",
        "total_wall_seconds",
    )
    for output in (512, 2048):
        subset = [
            r for r in records if r["job"]["parameters"]["output_tokens"] == output
        ]
        arms, paired, stability = {}, {}, []
        for arm in "AB":
            selected = [r for r in subset if r["job"]["arm"] == arm]
            arms[arm] = {
                key: statistics.median(
                    statistics.median(
                        metric(r, key) for r in selected if r["job"]["repeat"] == p
                    )
                    for p in range(3)
                )
                for key in keys
            }
            arms[arm]["memory_peak_bytes"] = max(
                r["validated"]["memory_peak_bytes"] for r in selected
            )
            arms[arm]["memory_categories_current"] = selected[-1]["memory_report"]
        for prompt in range(3):
            group = [r for r in subset if r["job"]["repeat"] == prompt]
            baseline = [r for r in group if r["job"]["arm"] == "A"]
            changes = {}
            for key in keys:
                ratios = [
                    (metric(group[b], key) / metric(group[a], key) - 1) * 100
                    for a, b in ((0, 1), (3, 2))
                ]
                changes[key] = {
                    "pair_changes_percent": ratios,
                    "median_change_percent": statistics.median(ratios),
                }
                values = [metric(r, key) for r in baseline]
                stability.append(
                    {
                        "prompt": prompt,
                        "metric": key,
                        "reference_spread_percent": (max(values) - min(values))
                        / statistics.median(values)
                        * 100,
                    }
                )
            paired[str(prompt)] = changes
        rows.append(
            {
                "output_tokens": output,
                "arms": arms,
                "paired_by_prompt": paired,
                "paired_change_percent": {
                    key: statistics.median(
                        paired[str(p)][key]["median_change_percent"] for p in range(3)
                    )
                    for key in keys
                },
                "reference_stability": stability,
                "reference_within_5_percent": all(
                    r["reference_spread_percent"] <= 5 for r in stability
                ),
                "peak_saved_bytes": arms["A"]["memory_peak_bytes"]
                - arms["B"]["memory_peak_bytes"],
            }
        )
    return {
        "status": "complete",
        "identity": identity,
        "profile": profile,
        "samples": len(records),
        "rows": rows,
    }


def performance(args):
    document = prepared(args, rehash=True)
    gate = matrix.read_json(args.output / "correctness.json")
    profile = getattr(args, "profile", "cache")
    budget = getattr(args, "cache_budget_bytes", 1 << 30)
    matrix.require(
        gate["status"] == "pass"
        and gate["identity"] == document["identity"]
        and gate.get("profile", "cache") == profile,
        "matching numerical/end-to-end correctness gate required",
    )
    prompts = matrix.read_json(args.output / "inputs.json")["prompts"]
    schedule, records = jobs(profile, budget), []
    for first in range(0, len(schedule), 4):
        group_jobs = schedule[first : first + 4]
        group_key = group_jobs[0]["abba_group"].replace("/", "-")
        checkpoint = args.output / "groups" / (group_key + ".json")
        checkpoint.parent.mkdir(exist_ok=True)
        if checkpoint.exists() and getattr(args, "resume", False):
            saved = matrix.read_json(checkpoint)
            matrix.require(
                saved["identity"] == document["identity"]
                and saved["profile"] == profile,
                "resume identity/profile differs",
            )
            if saved["status"] == "complete":
                matrix.require(
                    [r["job"] for r in saved["records"]] == group_jobs,
                    "resume schedule differs",
                )
                for record in saved["records"]:
                    attempt = matrix.read_json(
                        Path(record["raw_response"]["attempt_directory"])
                        / "evidence.json"
                    )
                    matrix.require(
                        attempt["performance_usable"]
                        and attempt["validation_status"] == "pass"
                        and attempt["identity"] == document["identity"]
                        and not attempt["external_workload_observed"],
                        "saved attempt was interfered or invalid",
                    )
                    matrix.require(
                        attempt["raw_response"] == record["raw_response"],
                        "saved attempt raw response changed",
                    )
                    checked = validate_native(
                        record["raw_response"],
                        args,
                        document,
                        record["job"],
                        [prompts[record["job"]["repeat"]]],
                        "performance",
                    )
                    matrix.require(
                        checked == record["validated"], "saved validation changed"
                    )
                records.extend(saved["records"])
                continue
            matrix.require(
                saved["status"] != "needs_divergence_analysis",
                "resolve divergence before resuming",
            )
        elif checkpoint.exists():
            raise matrix.MatrixError(
                "existing performance group; use --resume to preserve it"
            )
        if checkpoint.exists():
            old = matrix.read_json(checkpoint)
            archive = checkpoint.parent / "history"
            archive.mkdir(exist_ok=True)
            write(archive / (group_key + "-" + old["group_run"] + ".json"), old)
        prior = []
        group_run = uuid.uuid4().hex[:12]
        base = {
            "identity": document["identity"],
            "profile": profile,
            "group_run": group_run,
            "status": "running",
            "records": prior,
        }
        write(checkpoint, base)
        try:
            for job in group_jobs:
                prepared(args)
                label = job["sample_id"].replace("/", "-") + "-" + group_run
                print(
                    f"START {first + len(prior) + 1}/24 {label} arm={job['arm']}",
                    flush=True,
                )
                write(
                    args.output / "performance-run.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "profile": profile,
                        "active_group": group_key,
                        "group_run": group_run,
                        "active_sample": job["sample_id"],
                        "completed_groups": len(records) // 4,
                    },
                )
                raw, checked = native(
                    args, document, job, [prompts[job["repeat"]]], "performance", label
                )
                record = {
                    "job": job,
                    "raw_response": raw,
                    "validated": checked,
                    "memory_report": raw["memory_report"],
                    "output_token_ids": raw["requests"][0]["output_token_ids"],
                }
                divergence = output_gate(prior, record)
                prior.append(record)
                write(checkpoint, base)
                if divergence:
                    base |= {
                        "status": "needs_divergence_analysis",
                        "divergence": divergence,
                    }
                    write(checkpoint, base)
                    write(
                        args.output / "summary.json",
                        {
                            "status": "needs_divergence_analysis",
                            "identity": document["identity"],
                            "profile": profile,
                            "completed_groups": len(records) // 4,
                            "divergence": divergence,
                        },
                    )
                    raise matrix.MatrixError(
                        "token divergence recorded; subsequent batches blocked"
                    )
                print(
                    f"END {first + len(prior)}/24 valid, output gate passed", flush=True
                )
        except Exception as error:
            if base["status"] != "needs_divergence_analysis":
                base |= {"status": "incomplete", "error": str(error)}
                write(checkpoint, base)
            raise
        base["status"] = "complete"
        write(checkpoint, base)
        records.extend(prior)
    write(
        args.output / "summary.json", summarize(records, document["identity"], profile)
    )


def stage_profile(args):
    document = prepared(args, rehash=True)
    gate = matrix.read_json(args.output / "correctness.json")
    matrix.require(
        gate["status"] == "pass"
        and gate["identity"] == document["identity"]
        and gate["profile"] == args.profile,
        "stage profile requires matching correctness",
    )
    prompt = matrix.read_json(args.output / "inputs.json")["prompts"][0]
    result = []
    prior = []
    for arm, config in configurations(args.profile, args.cache_budget_bytes).items():
        p = config | {"output_tokens": 512, "ffn_stage_profile": True}
        job = {
            "group_id": matrix.digest(p),
            "repeat": 0,
            "arm": arm,
            "sample_id": "stage-" + arm,
            "parameters": p,
        }
        raw, checked = native(
            args, document, job, [prompt], "performance", "stage-profile-" + arm
        )
        matrix.require(
            raw.get("ffn_stage_profile", {}).get("diagnostic_only") is True,
            "stage diagnostic missing",
        )
        record = {
            "job": job,
            "output_token_ids": raw["requests"][0]["output_token_ids"],
        }
        divergence = output_gate(prior, record)
        prior.append(record)
        result.append({"arm": arm, "raw_response": raw, "validated": checked})
        write(
            args.output / "stage-profile.json",
            {
                "status": "needs_divergence_analysis" if divergence else "running",
                "diagnostic_only": True,
                "identity": document["identity"],
                "samples": result,
            },
        )
        matrix.require(divergence is None, "stage profile token divergence")
    write(
        args.output / "stage-profile.json",
        {
            "status": "complete",
            "diagnostic_only": True,
            "identity": document["identity"],
            "samples": result,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run-correctness", action="store_true")
    actions.add_argument("--run-long-correctness", action="store_true")
    actions.add_argument("--run-performance", action="store_true")
    actions.add_argument("--run-stage-profile", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/ffn-cache-experiment"
    )
    parser.add_argument("--profile", choices=PROFILES, default="cache")
    parser.add_argument("--cache-budget-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume complete ABBA groups, rerun incomplete groups",
    )
    parser.add_argument("--model-root", type=Path, default=adapter.STANDARD)
    parser.add_argument("--off-model-root", type=Path, default=adapter.OFF)
    parser.add_argument(
        "--allow-inference-pid",
        type=int,
        help="authorized llama-server PID; correctness actions only",
    )
    parser.add_argument(
        "--allow-concurrent-inference",
        action="store_true",
        help="authorized inference services may run alongside correctness only",
    )
    args = parser.parse_args()
    if args.allow_concurrent_inference and not (
        args.run_correctness or args.run_long_correctness
    ):
        parser.error("--allow-concurrent-inference requires a correctness action")
    if args.allow_inference_pid is not None and (
        args.allow_inference_pid <= 0
        or not (args.run_correctness or args.run_long_correctness)
    ):
        parser.error(
            "--allow-inference-pid requires a positive PID and a correctness action"
        )
    if args.resume and not args.run_performance:
        parser.error("--resume only applies to performance")
    if (
        args.prepare
        or args.run_correctness
        or args.run_long_correctness
        or args.run_performance
        or args.run_stage_profile
    ):
        action = (
            "prepare"
            if args.prepare
            else "correctness"
            if args.run_correctness
            else "long-correctness"
            if args.run_long_correctness
            else "stage-profile"
            if args.run_stage_profile
            else "performance"
        )
        try:
            if args.prepare:
                prepare(args)
            elif args.run_correctness:
                correctness(args)
            elif args.run_long_correctness:
                long_correctness(args)
            elif args.run_stage_profile:
                stage_profile(args)
            else:
                performance(args)
        except Exception as error:
            if args.output.is_dir():
                write(
                    args.output / (action + "-run.json"),
                    {"action": action, "state": "failed", "error": str(error)},
                )
            raise
        write(
            args.output / (action + "-run.json"),
            {"action": action, "state": "complete", "exit_code": 0},
        )
    else:
        print(
            json.dumps(
                {
                    "starts_inference": False,
                    "samples": 24,
                    "jobs": jobs(args.profile, args.cache_budget_bytes),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
