"""Correctness-only Swift 27B KV checks; never starts performance measurement."""

import fcntl
import json
import os
from pathlib import Path

from dev.benchmarks import qwen4b_ffn_cache as isolation
from dev.benchmarks import qwen4b_matrix as matrix
from dev.benchmarks import qwen4b_matrix_adapter as adapter
from dev.benchmarks import swift27b_kv as swift

ROOT = swift.ROOT
OUT = ROOT / "build/swift27b-kv-correctness-v1"


def first_difference(left, right):
    matrix.require(len(left) == len(right), "comparison token counts differ")
    for index, (reference, actual) in enumerate(zip(left, right, strict=True)):
        if reference != actual:
            return {"index": index, "reference": reference, "actual": actual}
    return None


def main():
    OUT.mkdir()
    plan = matrix.read_json(swift.OUT / "plan.json")
    identity = {
        "code_sha256": plan["runtime_code_sha256"],
        "weight_sha256": {
            key: value["sha256"] for key, value in plan["weights"].items()
        },
    }
    matrix.validate_identity(identity)
    matrix.require(
        adapter.code_sha256(ROOT) == identity["code_sha256"], "runtime changed"
    )
    for path, digest in plan["artifacts"].items():
        matrix.require(adapter.sha_file(path) == digest, "artifact changed: " + path)
    for weight in plan["weights"].values():
        matrix.require(
            adapter.sha_file(weight["path"]) == weight["sha256"], "weight changed"
        )
    own_hash = adapter.sha_file(Path(__file__))
    report = {
        "status": "running",
        "mode": "correctness",
        "performance_usable": False,
        "identity": identity,
        "artifacts": plan["artifacts"],
        "runner_sha256": own_hash,
        "base_parameters": swift.BASE,
        "planned_requests": 11,
        "numeric": [],
        "requests": [],
        "comparisons": [],
        "known_b4_failure": plan["known_b4_failure"],
        "scope": "C1; B4 remains failed and is not certified",
    }
    swift.save(OUT / "report.json", report)
    lock = (Path(plan["model_root"]) / "model.json").open("rb")
    fcntl.flock(lock, fcntl.LOCK_SH)
    try:
        for kv, share in [(kv, 0.0) for kv in swift.KV] + [("q4h", 0.5)]:
            directory = swift.OUT / ("numeric-" + kv + ("-ane" if share else ""))
            evidence = matrix.read_json(directory / "evidence.json")
            expected = [
                str(swift.OUT / "cold-graph-oracle"),
                str(swift.BUILD / "splash.metallib"),
                plan["model_root"],
                "--kv-format",
                kv,
                "--ane-ffn-share",
                str(share),
                "--single-request-only",
            ]
            matrix.require(evidence["command"] == expected, "numeric command changed")
            matrix.require(
                evidence["status"] == "complete" and evidence["exit_code"] == 0,
                "numeric check failed",
            )
            matrix.require(
                "PASS scope=swift-single-request-only"
                in (directory / "stdout.log").read_text(),
                "numeric PASS missing",
            )
            report["numeric"].append(
                {
                    "kv_format": kv,
                    "ane_share": share,
                    "directory": str(directory),
                    "exit_code": 0,
                    "reused_unchanged_artifacts": True,
                }
            )
        print(
            "NUMERIC VERIFIED 6; thresholds unchanged; short ANE setup is not an execution claim",
            flush=True,
        )
        codec = adapter.tokenizer(Path(plan["model_root"]))
        prompt = matrix.read_json(swift.OUT / "inputs.json")["prompts"][0]
        eos = adapter.eos_ids(Path(plan["model_root"]))
        references, full = {}, {}
        rank = swift.OUT / "input-frequency-ranking.splash"
        matrix.require(rank.is_file(), "compact ranking missing")
        report["ranking_sha256"] = adapter.sha_file(rank)
        cases = [(kv, "off", 512) for kv in swift.KV]
        cases += [(kv, "on", 16384) for kv in swift.KV]
        cases += [("q4h", "repeat", 16384)]
        for index, (kv, arm, length) in enumerate(cases):
            matrix.require(
                adapter.code_sha256(ROOT) == identity["code_sha256"], "runtime changed"
            )
            matrix.require(
                adapter.sha_file(Path(__file__)) == own_hash, "runner changed"
            )
            for path, digest in plan["artifacts"].items():
                matrix.require(
                    adapter.sha_file(path) == digest, "artifact changed: " + path
                )
            enabled = arm != "off"
            os.environ.update(MTL_SHADER_VALIDATION="0", MTL_DEBUG_LAYER="0")
            name = f"{index:02d}-{kv}-{arm}-{length}"
            directory = OUT / name
            directory.mkdir()
            parameters = swift.BASE | {"kv_format": kv, "output_tokens": length}
            job = {
                "parameters": parameters,
                "repeat": int(arm == "repeat"),
                "group_id": matrix.digest(parameters),
            }
            payload = {
                "protocol": "qwen4b-matrix-v1",
                "operation": "run_batch",
                "mode": "correctness",
                "optional_features_correctness": True,
                "optional_features": {
                    "prompt_lookup": enabled,
                    "compact_draft_head": enabled,
                    "fast_topk": False,
                    "compact_draft_ranking": str(rank),
                },
                "model_root": plan["model_root"],
                "metallib": str(swift.BUILD / "splash.metallib"),
                "identity": identity,
                "job": job,
                "prompts": [prompt],
            }
            swift.save(directory / "payload.json", payload)
            swift.save(
                OUT / "progress.json",
                {
                    "status": "running",
                    "mode": "correctness",
                    "active": name,
                    "completed": index,
                    "planned": len(cases),
                    "controller_pid": os.getpid(),
                    "performance_usable": False,
                },
            )
            print("START", name, flush=True)
            evidence = isolation.checked_process(
                [str(swift.OUT / "swift27b-kv-native")],
                json.dumps(payload),
                directory / "stdout.json",
                directory / "stderr.log",
                correctness=True,
                allow_concurrent_inference=True,
            )
            swift.save(directory / "evidence.json", evidence)
            matrix.require(
                evidence["status"] == "complete" and evidence["exit_code"] == 0,
                f"native correctness failed: {name}; {evidence['error']}",
            )
            matrix.require(
                evidence["performance_usable"] is False, "performance mode entered"
            )
            raw = matrix.read_json(directory / "stdout.json")
            raw["commands"] = [evidence["command"]]
            tokens = raw["requests"][0]["output_token_ids"]
            raw["requests"][0]["output_text"] = codec.decode(
                tokens, skip_special_tokens=False
            )
            matrix.require(
                "timings" not in raw
                and "startup_timings" not in raw
                and all("timings" not in request for request in raw["requests"]),
                "correctness response emitted performance timings",
            )
            validated = matrix.validate_batch(
                raw, job, [prompt], identity, eos, performance=False
            )
            swift.save(directory / "validated.json", validated)
            matrix.require(
                validated["status"] == "complete" and validated["facts_pass"],
                "factual or runtime setting validation failed",
            )
            matrix.require(
                raw["compiled_source_sha256"] == plan["compiled_source_sha256"],
                "compiled source changed",
            )
            matrix.require(raw["tail_plan_calls"] > 0, "tail splits not exercised")
            memory = raw["memory_report"]
            matrix.require(
                memory["ffn_fp16_cache_bytes"] == 0, "unexpected FP16 FFN cache"
            )
            matrix.require(
                sum(memory[key] for key in isolation.CATEGORIES)
                == memory["backend_current_bytes"],
                "memory ledger differs",
            )
            features = raw["optional_features"]
            matrix.require(
                features["configured"]
                == {
                    "prompt_lookup": enabled,
                    "compact_draft_head": enabled,
                    "fast_topk": False,
                },
                "actual feature settings differ",
            )
            if enabled:
                matrix.require(
                    features["compact_cycles"] > 0 and features["lookup_hits"] > 0,
                    "enabled optional features were not exercised",
                )
            else:
                matrix.require(
                    features["extra_metal_bytes"] == 0
                    and features["compact_cycles"] == 0
                    and features["lookup_hits"] == 0,
                    "disabled optional features executed or allocated",
                )
            swift.save(directory / "result.json", raw)
            comparison = None
            if arm == "off":
                references[kv] = tokens
            else:
                reference = full[kv] if arm == "repeat" else references[kv]
                comparison = {
                    "kv_format": kv,
                    "arm": arm,
                    "compared_tokens": len(reference),
                    "first_difference": first_difference(
                        reference, tokens[: len(reference)]
                    ),
                }
                report["comparisons"].append(comparison)
                if arm == "on":
                    full[kv] = tokens
            report["requests"].append(
                {
                    "name": name,
                    "kv_format": kv,
                    "arm": arm,
                    "input_tokens": 32768,
                    "output_tokens": len(tokens),
                    "native_exit_code": 0,
                    "facts_pass": True,
                    "output_sha256": matrix.digest(tokens),
                    "ane_actual": raw["ane_actual"],
                    "optional_features": features,
                    "memory_report": memory,
                    "directory": str(directory),
                }
            )
            swift.save(OUT / "report.json", report)
            matrix.require(
                comparison is None or comparison["first_difference"] is None,
                "token correctness divergence: " + str(comparison),
            )
            print("PASS", name, flush=True)
        for weight in plan["weights"].values():
            matrix.require(
                adapter.sha_file(weight["path"]) == weight["sha256"], "weight changed"
            )
        report["status"] = "passed"
        swift.save(OUT / "report.json", report)
        swift.save(
            OUT / "progress.json",
            {
                "status": "passed",
                "mode": "correctness",
                "completed": len(cases),
                "planned": len(cases),
                "performance_usable": False,
            },
        )
        print(
            "PASS ALL C1 CHECKS; B4 remains unresolved; performance stays stopped",
            flush=True,
        )
    except BaseException as error:
        report.update(
            status="failed", error=str(error), error_type=type(error).__name__
        )
        swift.save(OUT / "report.json", report)
        swift.save(
            OUT / "progress.json",
            {
                "status": "failed",
                "mode": "correctness",
                "completed": len(report["requests"]),
                "planned": 11,
                "error": str(error),
                "performance_usable": False,
            },
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
