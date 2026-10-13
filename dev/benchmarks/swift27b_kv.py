"""Swift 27B: fixed mixed GGUF weights, five KV formats, 32K/16K cold requests."""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from dev.benchmarks import qwen4b_ffn_cache as isolation
from dev.benchmarks import qwen4b_matrix as matrix
from dev.benchmarks import qwen4b_matrix_adapter as adapter
from dev.tools.compact_draft_ranking import write_ranking
from install import assembly, gguf, models

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/swift27b-kv16k-c1-v2"
BUILD = ROOT / "build/ghost-merge-validation/block-8"
MODEL_ID = "ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S-mtp"
KV = ("bf16", "int8", "q4_0", "q4_0_unrotated", "q4h")
BASE = {
    "ane_requested": True,
    "ane_share": 0.5,
    "ane_minimum_rows": 512,
    "auto_calibration": False,
    "drafter_enabled": True,
    "drafter_block_size": 8,
    "tail_prefill_splits": 16,
    "concurrency": 1,
    "input_tokens": 32768,
    "output_tokens": 16384,
    "context_tokens": 131072,
    "memory_budget_bytes": 40 << 30,
    "temperature": 0,
    "ignore_eos": True,
    "cold_requests": True,
}


def save(path, value):
    isolation.write(path, value)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def prepare():
    OUT.mkdir()
    selection = models.Selection.of(
        ROOT / "install/models-local", MODEL_ID, language_only=True
    )
    record = assembly.verify(selection.link)
    model_root = selection.link.resolve()
    target_files = list((model_root / "target").glob("*.gguf"))
    assert len(target_files) == 1
    metadata = gguf.Metadata(target_files[0], tensors=True)
    save(
        OUT / "weight-formats.json",
        {
            "formats": dict(
                Counter(gguf.TENSOR_TYPES[k] for k in metadata.tensors.values())
            ),
            "fixed_file": str(target_files[0].resolve()),
            "main_layers": gguf.loaded_layers(metadata, "qwen35"),
            "mtp_layers": metadata.values["qwen35.nextn_predict_layers"],
            "mtp_used": False,
        },
    )
    source = (ROOT / "runtime/ops/PagedAttention.cpp").read_text()
    start = source.index("  const uint32_t splits = rows == 32")
    end = source.index("  const uint32_t fusedRows", start)
    source = (
        source[:start]
        + "  const uint32_t splits = rows == 32 && kernel == KernelLayout::Kv4Group6\n      ? matrixTailSplits() : prefillSplits(tiles);\n"
        + source[end:]
    )
    source = replace_once(
        source,
        "namespace splash::ops {",
        "namespace splash::ops {\nuint32_t matrixTailSplits();\nvoid matrixRecordTailSplit(uint32_t rows, uint32_t splits);",
    )
    source = replace_once(
        source,
        "  const kv::PrefillAttentionParams params{",
        "  matrixRecordTailSplit(plan.rows, plan.splits);\n  const kv::PrefillAttentionParams params{",
    )
    (OUT / "SwiftPagedAttention.cpp").write_text(source)
    oracle = (ROOT / "dev/tests/engine/model_runtime_oracle_test.mm").read_text()
    seed = oracle[
        oracle.index("    const std::vector<uint32_t> productionSeedTokens{") :
    ]
    seed = seed[: seed.index("};") + 2]
    b4 = oracle[
        oracle.index("    // Identical inputs within one B4 graph") : oracle.index(
            "    const std::vector<uint32_t> productionSeedTokens{"
        )
    ]
    head = oracle[
        oracle.index("    // Lanes that finish their prompts") : oracle.index(
            "    constexpr std::array<uint64_t, 4> productionB4Ids"
        )
    ]
    oracle = replace_once(
        oracle,
        "    bool imagesOnly = false, warmupEosOnly = false;",
        "    bool imagesOnly = false, warmupEosOnly = false;\n    bool coldGraphOnly = false;\n    bool singleRequestOnly = false;",
    )
    oracle = replace_once(
        oracle,
        '      if (option == "--images-only") imagesOnly = true;',
        '      if (option == "--cold-graph-only") coldGraphOnly = true;\n      else if (option == "--single-request-only") singleRequestOnly = true;\n      else if (option == "--images-only") imagesOnly = true;',
    )
    oracle = replace_once(
        oracle,
        '        if (value != "int8" && value != "bf16") fail("invalid KV format");\n        format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;',
        '        if (value != "int8" && value != "bf16" && value != "q4h" && value != "q4_0" && value != "q4_0_unrotated") fail("invalid KV format");\n        format = value == "int8" ? kv::Format::Int8 : value == "q4h" ? kv::Format::Int4Hadamard : value == "q4_0" ? kv::Format::Q4_0Hadamard : value == "q4_0_unrotated" ? kv::Format::Q4_0Unrotated : kv::Format::BFloat16;',
    )
    marker = "    // A reusable composite state contains no final hidden."
    assert oracle.count(marker) == 1
    oracle = oracle.replace(
        marker,
        "    if (coldGraphOnly) {\n"
        + seed
        + "\n"
        + b4
        + head
        + '      std::cout << "PASS scope=swift-cold-graph-only; original state cosine>0.999, same-graph token identity, same-row scores<=1e-3\\n";\n      return 0;\n    }\n\n'
        + marker,
        1,
    )
    c1_source = '    if (singleRequestOnly) {\n      std::array<ModelStepResult, 2> repeats;\n      for (uint32_t trial = 0; trial < 2; ++trial) {\n        const uint64_t id = 64 + trial;\n        beginCold(executor, makeRequest(id, {279}, 16), 0);\n        prefillToken(executor, id, 0, 279, {43});\n        repeats[trial] = decodeOne(executor, id, 1, {43});\n        executor.end(id);\n      }\n      require(repeats[0].outputTokens == repeats[1].outputTokens &&\n                  repeats[0].acceptedDraftTokens == repeats[1].acceptedDraftTokens &&\n                  repeats[0].draftedTokens == repeats[1].draftedTokens &&\n                  repeats[0].outputTokensWithoutKv == repeats[1].outputTokensWithoutKv &&\n                  repeats[0].finished == repeats[1].finished &&\n                  repeats[0].nextDecodeStage == repeats[1].nextDecodeStage,\n              "identical single-request cold runs made different decisions");\n      std::array<ModelStepResult, 2> scored;\n      for (uint32_t trial = 0; trial < 2; ++trial) {\n        const uint64_t id = 70 + trial;\n        EngineRequest request = makeRequest(id, prompt128, 0);\n        request.scoreTokens = {11, 220, 1683};\n        beginCold(executor, request, 0);\n        scored[trial] = prefillChunk(executor, id, 0, prompt128, pageRange(48, 8));\n        require(scored[trial].scoreLogits.size() == 3,\n                "single-request scores are missing");\n        executor.end(id);\n      }\n      double maximumRelative = 0;\n      for (uint32_t option = 0; option < 3; ++option) {\n        const float actual = scored[0].scoreLogits[option];\n        const float reference = scored[1].scoreLogits[option];\n        require(std::isfinite(actual) && std::isfinite(reference),\n                "single-request scores are not finite");\n        const float relative = std::fabs(actual - reference) /\n                               std::max(1.0F, std::fabs(reference));\n        require(relative <= 1e-3F, "single-request scores changed across cold runs");\n        maximumRelative = std::max(maximumRelative, double(relative));\n      }\n      std::cout << "single_request_score_max_relative=" << maximumRelative << " bound=0.001\\n";\n      std::cout << "PASS scope=swift-single-request-only; original partition cosine>0.999 and score error<=1e-3; C1 cold decisions identical; B4 not certified\\n";\n      return 0;\n    }\n'
    oracle = oracle.replace(marker, c1_source + "\n" + marker, 1)
    (OUT / "ColdGraphOracle.mm").write_text(oracle)
    flags = [
        "xcrun",
        "-sdk",
        "macosx",
        "clang++",
        "-std=c++20",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Iruntime",
        "-Idev",
        "-Idev/benchmarks",
        "-Idev/tests/engine",
        "-DSPLASH_DRAFT_BLOCK_SIZE=8",
        "-mmacosx-version-min=26.4",
    ]
    framework = [
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "IOKit",
        "-framework",
        "IOSurface",
    ]
    identity = ["-include", str(BUILD / "engine/BuildIdentity.hpp")]
    commands = [
        flags
        + [
            "-c",
            str(OUT / "SwiftPagedAttention.cpp"),
            "-o",
            str(OUT / "SwiftPagedAttention.o"),
        ],
        flags
        + ["-fobjc-arc"]
        + identity
        + [
            "dev/benchmarks/swift27b_kv_native.mm",
            str(OUT / "SwiftPagedAttention.o"),
            str(BUILD / "engine/libsplash.a"),
        ]
        + framework
        + ["-o", str(OUT / "swift27b-kv-native")],
        flags
        + ["-fobjc-arc"]
        + identity
        + [
            str(OUT / "ColdGraphOracle.mm"),
            str(BUILD / "engine/metal/MetalBackendInstrumented.o"),
            str(BUILD / "engine/ane/ProgramInstrumented.o"),
            str(BUILD / "engine/libsplash.a"),
        ]
        + framework
        + ["-o", str(OUT / "cold-graph-oracle")],
        flags
        + ["-fobjc-arc"]
        + identity
        + [
            "dev/benchmarks/qwen4b_matrix_native.mm",
            str(OUT / "SwiftPagedAttention.o"),
            str(BUILD / "engine/libsplash.a"),
        ]
        + framework
        + ["-o", str(OUT / "qwen4b-compatibility-build")],
    ]
    save(OUT / "build-commands.json", commands)
    with (OUT / "build.log").open("w") as log:
        for command in commands:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            assert result.returncode == 0, ("build failed", command)
    adapter.prepare_inputs(model_root, OUT / "inputs.json")
    artifacts = {
        str(p): adapter.sha_file(p)
        for p in [
            OUT / "swift27b-kv-native",
            OUT / "cold-graph-oracle",
            OUT / "SwiftPagedAttention.cpp",
            OUT / "ColdGraphOracle.mm",
            BUILD / "engine/libsplash.a",
            BUILD / "splash.metallib",
            ROOT / "dev/benchmarks/qwen4b_matrix_native.mm",
            Path(__file__),
        ]
    }
    weights = {
        ("target" if k.startswith("target/") else "draft"): {
            "path": v["path"],
            "sha256": v["digest"],
        }
        for k, v in record["files"].items()
        if k.endswith((".gguf", ".safetensors"))
    }
    plan = {
        "model_root": str(model_root),
        "model_id": MODEL_ID,
        "base_parameters": BASE,
        "kv_formats": KV,
        "rounds": 3,
        "planned": 15,
        "features": {"lookup": True, "compact": True, "fast": False},
        "weights": weights,
        "runtime_code_sha256": adapter.code_sha256(ROOT),
        "artifacts": artifacts,
        "compiled_source_sha256": json.loads(
            (BUILD / "engine/build-identity.json").read_text()
        )["source_sha256"],
        "native_sha256": adapter.sha_file(OUT / "swift27b-kv-native"),
        "original_numeric_thresholds_preserved": True,
        "scope": "single request; concurrency4 speculative decisions remain failed",
        "known_b4_failure": "build/swift27b-kv16k-v1/numeric-bf16-diagnostic/evidence.json",
    }
    save(OUT / "plan.json", plan)
    print(
        json.dumps(
            {
                "status": "prepared",
                "directory": str(OUT),
                "planned": 15,
                "quantization_variable": "KV only",
            }
        )
    )


def await_idle(completed):
    streak = 0
    while streak < 3:
        environment = isolation.environment()
        idle = (
            not environment["busy_pids"]
            and environment["gpu_utilization_percent"] is not None
            and environment["gpu_utilization_percent"] <= 50
        )
        save(
            OUT / "progress.json",
            {
                "status": "checking_idle" if idle else "waiting_for_idle",
                "completed": completed,
                "planned": 15,
                "environment": environment,
                "controller_pid": os.getpid(),
            },
        )
        print(
            "IDLE",
            environment["gpu_utilization_percent"],
            environment["busy_pids"],
            flush=True,
        )
        streak = streak + 1 if idle else 0
        if streak < 3:
            time.sleep(15)


def run():
    plan = matrix.read_json(OUT / "plan.json")
    codec = adapter.tokenizer(Path(plan["model_root"]))
    prompts = matrix.read_json(OUT / "inputs.json")["prompts"]
    identity = {
        "code_sha256": plan["runtime_code_sha256"],
        "weight_sha256": {k: v["sha256"] for k, v in plan["weights"].items()},
    }
    matrix.validate_identity(identity)
    for path, digest in plan["artifacts"].items():
        assert adapter.sha_file(Path(path)) == digest, path
    for weight in plan["weights"].values():
        assert adapter.sha_file(Path(weight["path"])) == weight["sha256"]
    record_lock = (Path(plan["model_root"]) / "model.json").open("rb")
    fcntl.flock(record_lock, fcntl.LOCK_SH)
    records, goldens = [], {}
    environment = isolation.environment
    try:
        numeric_cases = [(kv, 0.0) for kv in KV] + [("q4h", 0.5)]
        for index, (kv, ane_share) in enumerate(numeric_cases):
            save(
                OUT / "progress.json",
                {
                    "status": "correctness",
                    "active": "numeric-" + kv + ("-ane" if ane_share else ""),
                    "completed_numeric": index,
                    "planned_numeric": 6,
                    "completed": 0,
                    "planned": 15,
                    "controller_pid": os.getpid(),
                },
            )
            directory = OUT / ("numeric-" + kv + ("-ane" if ane_share else ""))
            directory.mkdir(exist_ok=True)
            print("NUMERIC START", kv, "ANE share", ane_share, flush=True)
            command = [
                str(OUT / "cold-graph-oracle"),
                str(BUILD / "splash.metallib"),
                plan["model_root"],
                "--kv-format",
                kv,
                "--ane-ffn-share",
                str(ane_share),
                "--single-request-only",
            ]
            prior = directory / "evidence.json"
            if prior.exists():
                evidence = matrix.read_json(prior)
                assert evidence["command"] == command and evidence["exit_code"] == 0
                assert (
                    "PASS scope=swift-single-request-only"
                    in (directory / "stdout.log").read_text()
                )
                print("REUSE VERIFIED NUMERIC", kv, ane_share, flush=True)
            else:
                evidence = isolation.checked_process(
                    command,
                    "",
                    directory / "stdout.log",
                    directory / "stderr.log",
                    correctness=True,
                )
            save(directory / "evidence.json", evidence)
            assert (
                evidence["exit_code"] == 0
                and "PASS scope=swift-single-request-only"
                in (directory / "stdout.log").read_text()
            ), ("numeric gate failed", kv)
            print("NUMERIC PASS", kv, flush=True)
        rank = OUT / "input-frequency-ranking.splash"
        short = copy.deepcopy(prompts[0])
        short["token_ids"] = short["token_ids"][:64]
        short["input_sha256"] = matrix.digest(short["token_ids"])
        short["expected_facts"] = []
        short_parameters = BASE | {
            "kv_format": "q4h",
            "ane_requested": False,
            "ane_share": 0,
            "ane_minimum_rows": 0,
            "input_tokens": 64,
            "output_tokens": 32,
        }
        short_job = {
            "parameters": short_parameters,
            "repeat": 0,
            "group_id": matrix.digest(short_parameters),
        }
        short_payload = {
            "protocol": "qwen4b-matrix-v1",
            "operation": "run_batch",
            "mode": "correctness",
            "model_root": plan["model_root"],
            "metallib": str(BUILD / "splash.metallib"),
            "identity": identity,
            "job": short_job,
            "prompts": [short],
        }
        directory = OUT / "short-q4h-identity-fixed"
        directory.mkdir()
        evidence = isolation.checked_process(
            [str(OUT / "swift27b-kv-native")],
            json.dumps(short_payload),
            directory / "stdout.json",
            directory / "stderr.log",
            correctness=True,
        )
        save(directory / "evidence.json", evidence)
        assert evidence["exit_code"] == 0
        raw = matrix.read_json(directory / "stdout.json")
        raw["commands"] = [evidence["command"]]
        raw["requests"][0]["output_text"] = codec.decode(
            raw["requests"][0]["output_token_ids"], skip_special_tokens=False
        )
        save(
            directory / "validated.json",
            matrix.validate_batch(
                raw,
                short_job,
                [short],
                identity,
                adapter.eos_ids(Path(plan["model_root"])),
                performance=False,
            ),
        )
        counts = Counter(token for prompt in prompts for token in prompt["token_ids"])
        write_ranking(
            rank,
            raw["target_manifest_sha256"],
            sorted(range(len(codec)), key=lambda token: (-counts[token], token)),
        )
        os.environ.update(
            MTL_SHADER_VALIDATION="0",
            MTL_DEBUG_LAYER="0",
        )
        orders = [KV, tuple(reversed(KV)), KV[2:] + KV[:2]]
        for repeat, order in enumerate(orders):
            for kv in order:
                attempt = 0
                while True:
                    assert adapter.code_sha256(ROOT) == plan["runtime_code_sha256"]
                    for path, digest in plan["artifacts"].items():
                        assert adapter.sha_file(Path(path)) == digest, path
                    await_idle(len(records))
                    attempt += 1
                    name = f"perf-{repeat}-{kv}-attempt{attempt}"
                    directory = OUT / name
                    directory.mkdir()
                    parameters = BASE | {"kv_format": kv}
                    job = {
                        "parameters": parameters,
                        "repeat": repeat,
                        "group_id": matrix.digest(parameters),
                    }
                    payload = {
                        "optional_features": {
                            "prompt_lookup": True,
                            "compact_draft_head": True,
                            "fast_topk": False,
                            "compact_draft_ranking": str(rank),
                        },
                        "protocol": "qwen4b-matrix-v1",
                        "operation": "run_batch",
                        "mode": "performance",
                        "model_root": plan["model_root"],
                        "metallib": str(BUILD / "splash.metallib"),
                        "identity": identity,
                        "job": job,
                        "prompts": [prompts[0]],
                    }
                    save(directory / "payload.json", payload)
                    save(
                        OUT / "progress.json",
                        {
                            "status": "running",
                            "active": name,
                            "completed": len(records),
                            "planned": 15,
                            "controller_pid": os.getpid(),
                        },
                    )
                    background = []

                    def checked_environment(*args, **kwargs):
                        current = environment(*args, **kwargs)
                        background.append(current)
                        if args and args[0] and current["busy_pids"]:
                            raise RuntimeError(
                                "background-interference: " + str(current["busy_pids"])
                            )
                        return current

                    isolation.environment = checked_environment
                    print("START", name, flush=True)
                    try:
                        evidence = isolation.checked_process(
                            [str(OUT / "swift27b-kv-native")],
                            json.dumps(payload),
                            directory / "stdout.json",
                            directory / "stderr.log",
                            correctness=False,
                        )
                    finally:
                        isolation.environment = environment
                    evidence["background_samples"] = background
                    save(directory / "evidence.json", evidence)
                    if evidence["exit_code"] != 0:
                        error = evidence.get("error") or ""
                        if error.startswith(
                            (
                                "background-interference:",
                                "external inference active:",
                                "GPU not idle enough",
                                "experiment device lock is held",
                            )
                        ):
                            print("RETRY EXTERNAL LOAD", name, error, flush=True)
                            continue
                        raise RuntimeError(("native failed", name, evidence))
                    raw = matrix.read_json(directory / "stdout.json")
                    raw["commands"] = [evidence["command"]]
                    raw["requests"][0]["output_text"] = codec.decode(
                        raw["requests"][0]["output_token_ids"],
                        skip_special_tokens=False,
                    )
                    checked = matrix.validate_batch(
                        raw,
                        job,
                        [prompts[0]],
                        identity,
                        adapter.eos_ids(Path(plan["model_root"])),
                        performance=True,
                    )
                    assert (
                        checked["facts_pass"]
                        and raw["compiled_source_sha256"]
                        == plan["compiled_source_sha256"]
                    )
                    assert raw["tail_plan_calls"] > 0
                    assert raw["memory_report"]["ffn_fp16_cache_bytes"] == 0
                    assert (
                        sum(raw["memory_report"][k] for k in isolation.CATEGORIES)
                        == raw["memory_report"]["backend_current_bytes"]
                    )
                    assert raw["optional_features"]["configured"] == {
                        "prompt_lookup": True,
                        "compact_draft_head": True,
                        "fast_topk": False,
                    }
                    tokens = raw["requests"][0]["output_token_ids"]
                    if kv in goldens:
                        assert tokens == goldens[kv], (
                            "same-KV full-token divergence",
                            kv,
                        )
                    else:
                        goldens[kv] = tokens
                    save(directory / "validated.json", checked)
                    request = checked["requests"][0]
                    result = {
                        "name": name,
                        "kv_format": kv,
                        "repeat": repeat,
                        "native_exit_code": 0,
                        "facts_pass": True,
                        "prefill_tok_s": request["prefill_tok_s"],
                        "decode_tok_s": request["decode_tok_s"],
                        "timings": request["timings"],
                        "memory_report": raw["memory_report"],
                        "ane_actual": raw["ane_actual"],
                        "optional_features": raw["optional_features"],
                        "draft_acceptance": raw["accepted_draft_tokens"]
                        / raw["drafted_tokens"],
                        "observed_busy_pids": sorted(
                            {pid for s in background for pid in s["busy_pids"]}
                        ),
                        "full_token_repeatability": True,
                    }
                    save(directory / "result.json", result)
                    records.append(result)
                    save(
                        OUT / "summary.json",
                        {
                            "status": "running",
                            "completed": len(records),
                            "planned": 15,
                            "records": records,
                        },
                    )
                    print(
                        "END",
                        name,
                        result["prefill_tok_s"],
                        result["decode_tok_s"],
                        flush=True,
                    )
                    break
        for weight in plan["weights"].values():
            assert adapter.sha_file(Path(weight["path"])) == weight["sha256"]
        save(
            OUT / "summary.json",
            {"status": "complete", "completed": 15, "planned": 15, "records": records},
        )
        save(
            OUT / "progress.json",
            {"status": "complete", "completed": 15, "planned": 15},
        )
        print(
            "COMPLETE 15; mixed GGUF unchanged; original numerical thresholds retained",
            flush=True,
        )
    except BaseException as error:
        save(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "completed": len(records),
                "planned": 15,
            },
        )
        save(
            OUT / "progress.json",
            {
                "status": "failed",
                "completed": len(records),
                "planned": 15,
                "error": str(error),
            },
        )
        raise
    finally:
        isolation.environment = environment
        record_lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    arguments = parser.parse_args()
    os.chdir(ROOT)
    prepare() if arguments.action == "prepare" else run()
