"""CPU preparation and the repository-native qwen4b-matrix-v1 adapter.

Capabilities inspect stamps/files only; they never start the native executable.
--stamp-variants is explicit and refuses stale build configurations, generated
build identities, attention snapshots or source timestamps. Each stamp hashes
real artifacts and model files. Later queries verify artifacts and file metadata.
No workload or calibration runs as part of stamping or input preparation.

The normal adapter reads one JSON object from stdin. run_batch requires an
explicit mode; correctness is limited to input128/output32. --run-correctness
prepares a 64-token, four-request fixture and is an explicit inference action.
Do not invoke that action while runtime testing is on hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from dev.benchmarks import qwen4b_matrix as matrix
from dev.tools import build_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = ROOT / "build/qwen4b-matrix-variants"
STANDARD = (
    ROOT
    / "install/models-local/.resolved/dae35a717bda8642d409dbb0ce8655296cd37aaa2288c2bd2893ca02244c4f43"
)
OFF = (
    ROOT
    / "install/models-local/.resolved/0db89c15cde311ca967567bc505737ddd7383d4c31e1123b537d564f46a756f8"
)
BLOCKS = (2, 4, 8, 16)
EXTRA_SOURCES = (
    "Makefile",
    "dev/native.mk",
    "dev/tools/build_config.py",
    "dev/benchmarks/qwen4b_matrix_native.mm",
    "dev/benchmarks/qwen4b_matrix_attention.py",
    "dev/benchmarks/qwen4b_matrix.py",
    "dev/benchmarks/qwen4b_matrix_adapter.py",
    "dev/benchmarks/qwen4b_matrix_manifest.json",
)


def sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def signature(path):
    stat = Path(path).stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def file_record(path):
    path = Path(path).resolve(strict=True)
    before = signature(path)
    actual = sha_file(path)
    matrix.require(signature(path) == before, f"file changed while hashing: {path}")
    return {"path": str(path), "sha256": actual, "signature": before}


def verify_record(record, rehash=False):
    path = Path(record["path"])
    matrix.require(
        path.is_file() and signature(path) == record["signature"], f"stale file: {path}"
    )
    if rehash:
        matrix.require(
            sha_file(path) == record["sha256"], f"changed file digest: {path}"
        )


def source_inputs(root):
    return tuple(
        sorted(set(build_identity.production_input_paths(root)) | set(EXTRA_SOURCES))
    )


def code_sha256(root):
    return build_identity.source_digest(root, source_inputs(root))


def prefix_digest(values):
    return hashlib.sha256(("\0".join(values) + "\0").encode()).hexdigest()[:16]


def expected_configuration(root, block):
    """Mirror current Makefile flags/config records; overridden flags are unsupported."""
    macro = f"-DSPLASH_DRAFT_BLOCK_SIZE={block}"
    deployment = "-mmacosx-version-min=26.4"
    cxx = f"-std=c++20 -O3 -Wall -Wextra -Werror -Iruntime {macro} {deployment}"
    objc = cxx + " -fobjc-arc"
    metal = f"-std=metal4.0 -O3 -Wall -Wextra -Werror -Iruntime {macro} {deployment}"
    link = (
        "-framework Foundation -framework Metal -framework IOKit -framework IOSurface"
    )
    constants = {
        "runtime": "qwen-hybrid-dflash8",
        "engine_cxxflags": cxx,
        "engine_objcxxflags": objc,
        "engine_linkflags": link,
        "production_metalflags": metal,
    }
    rules = "".join(
        f"{sha_file(root / p)}  {p}\n"
        for p in ("Makefile", "dev/native.mk", "dev/tools/build_config.py")
    )
    rules_digest = hashlib.sha256(rules.encode()).hexdigest()[:16]
    production = prefix_digest(
        [
            rules_digest,
            "xcrun -sdk macosx clang++",
            cxx,
            objc,
            "-MMD -MP",
            link,
            "xcrun -sdk macosx ar",
            "xcrun -sdk macosx metal",
            metal,
            "xcrun -sdk macosx metallib",
        ]
    )
    test = prefix_digest(
        [
            production,
            f"-std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -Idev {macro} {deployment}",
            metal.replace(" -O3", ""),
        ]
    )
    headers = sorted(
        p.relative_to(root).as_posix()
        for folder in ("runtime/metal/abi", "runtime/metal/kernels/common")
        for p in (root / folder).glob("*.h")
    )
    kernels = sorted(
        p.relative_to(root).as_posix()
        for phase in ("prefill", "decode", "shared")
        for p in (root / "runtime/metal/kernels" / phase).glob("*.metal")
    )
    return {
        "compile_macro": macro,
        "constants": constants,
        "production": production,
        "test": test,
        "metallib": production
        + "-"
        + prefix_digest(headers)
        + "-"
        + prefix_digest(kernels),
    }


def expected_attention(root):
    """Verify the generator's exact matrix-only override snapshot."""
    source = (root / "runtime/ops/PagedAttention.cpp").read_text()
    old = "kernel == KernelLayout::Kv4Group4 && layout.format == kv::Format::Int8\n      ? 1U : prefillSplits(tiles);"
    replacements = (
        (
            old,
            "kernel == KernelLayout::Kv4Group4\n      ? matrixTailSplits() : prefillSplits(tiles);",
        ),
        (
            "namespace splash::ops {",
            "namespace splash::ops {\nuint32_t matrixTailSplits();\nvoid matrixRecordTailSplit(uint32_t rows, uint32_t splits);",
        ),
        (
            "  const kv::PrefillAttentionParams params{",
            "  matrixRecordTailSplit(plan.rows, plan.splits);\n  const kv::PrefillAttentionParams params{",
        ),
    )
    for before, after in replacements:
        matrix.require(
            source.count(before) == 1, "matrix attention generator contract changed"
        )
        source = source.replace(before, after, 1)
    return source


def verify_build(root, variant, block):
    config = expected_configuration(root, block)
    native = variant / "engine-tests/qwen4b-matrix-native"
    library = variant / "splash.metallib"
    checks = {
        native: config["test"],
        library: config["metallib"],
        variant / "engine/libsplash.a": config["production"],
        variant / "engine-tests/matrix-PagedAttention.o": config["test"],
    }
    for artifact, expected in checks.items():
        matrix.require(artifact.is_file(), f"missing artifact: {artifact}")
        matrix.require(
            Path(str(artifact) + ".config").read_text().strip() == expected,
            f"stale build_config record for block{block}: {artifact}",
        )
    identity = build_identity.build_id(root, constants=config["constants"])
    matrix.require(
        (variant / "engine/build-identity.json").read_bytes()
        == build_identity.stamp_bytes(identity),
        f"stale production build identity for block{block}",
    )
    matrix.require(
        (variant / "engine/BuildIdentity.hpp").read_bytes()
        == build_identity.header_bytes(identity),
        f"stale generated build header for block{block}",
    )
    generated = variant / "engine-tests/matrix-PagedAttention.cpp"
    matrix.require(
        generated.read_text() == expected_attention(root),
        "stale matrix attention source snapshot",
    )
    dependency_names = set(build_identity.production_input_paths(root)) | {
        "Makefile",
        "dev/native.mk",
        "dev/tools/build_config.py",
        "dev/benchmarks/qwen4b_matrix_native.mm",
        "dev/benchmarks/qwen4b_matrix_attention.py",
    }
    metal_dependencies = {
        name
        for name in dependency_names
        if name.endswith(".metal")
        or name.startswith("runtime/metal/abi/")
        or name.startswith("runtime/metal/kernels/common/")
    } | {"Makefile", "dev/native.mk", "dev/tools/build_config.py"}
    for artifact, dependencies in (
        (native, dependency_names),
        (library, metal_dependencies),
    ):
        newest = max((root / name).stat().st_mtime_ns for name in dependencies)
        matrix.require(
            artifact.stat().st_mtime_ns >= newest,
            f"artifact predates source/header/rules/harness: {artifact}",
        )
    matrix.require(
        native.stat().st_mtime_ns >= generated.stat().st_mtime_ns,
        "native predates its matrix-only attention snapshot",
    )
    return config, native, library


def model_records(model_root, cache=None):
    cache = cache if cache is not None else {}
    document = matrix.read_json(model_root / "model.json")
    matrix.require(
        document.get("family") == "Qwen3.5-4B", "model root is not Qwen3.5-4B"
    )
    result = {}
    for name, entry in document.get("files", {}).items():
        if name.endswith((".gguf", ".safetensors")) and name.startswith(
            ("target/", "draft/")
        ):
            path = Path(entry["path"]).resolve(strict=True)
            if path not in cache:
                cache[path] = file_record(path)
            record = cache[path]
            matrix.require(
                record["sha256"] == entry["digest"],
                f"model manifest digest differs: {name}",
            )
            result[name] = record
    matrix.require(
        any(name.startswith("target/") for name in result), "missing target weights"
    )
    return result


def role_hashes(records):
    roles = {}
    for role in ("target", "draft"):
        files = sorted(
            (name, record["sha256"])
            for name, record in records.items()
            if name.startswith(role + "/")
        )
        if files:
            roles[role] = files[0][1] if len(files) == 1 else matrix.digest(files)
    return roles


def eos_ids(model_root):
    config = matrix.read_json(model_root / "tokenizer/tokenizer_config.json")
    tokens = config.get("added_tokens_decoder", {})
    matrix.require(
        config.get("eos_token") == "<|im_end|>"
        and tokens.get("248044", {}).get("content") == "<|endoftext|>"
        and tokens.get("248046", {}).get("content") == "<|im_end|>",
        "unexpected model EOS vocabulary",
    )
    return [248044, 248046]


def stamp_variants(root, variant_root, standard, off):
    cache = {}
    models = {
        "standard": model_records(standard, cache),
        "off": model_records(off, cache),
    }
    matrix.require(
        role_hashes(models["standard"]).get("target")
        == role_hashes(models["off"]).get("target"),
        "off and standard roots use different target weights",
    )
    matrix.require(
        "draft" in role_hashes(models["standard"])
        and "draft" not in role_hashes(models["off"]),
        "model roots do not represent standard and drafter-off states",
    )
    source = code_sha256(root)
    stamped, rejected = [], {}
    for block in BLOCKS:
        variant = variant_root / f"block-{block}"
        try:
            config, native, library = verify_build(root, variant, block)
            document = {
                "schema_version": 1,
                "block_size": block,
                "compile_macro": config["compile_macro"],
                "code_sha256": source,
                "model_roots": {
                    "standard": str(standard.resolve()),
                    "off": str(off.resolve()),
                },
                "models": models,
                "model_manifests": {
                    "standard": file_record(standard / "model.json"),
                    "off": file_record(off / "model.json"),
                },
                "artifact_sha256": {
                    "native": file_record(native),
                    "metallib": file_record(library),
                },
                "configuration": config,
                "eos_token_ids": eos_ids(standard),
            }
            (variant / "matrix-identity.json").write_text(
                json.dumps(document, indent=2) + "\n"
            )
            stamped.append(block)
        except (OSError, ValueError, build_identity.BuildIdentityError) as error:
            rejected[str(block)] = str(error)
    return {
        "mode": "stamp",
        "starts_inference": False,
        "stamped_block_sizes": stamped,
        "unsupported_variants": rejected,
    }


def inspect_variants(root, variant_root, standard, off):
    supported, failures = {}, {}
    source = code_sha256(root)
    for block in BLOCKS:
        variant = variant_root / f"block-{block}"
        try:
            stamp = matrix.read_json(variant / "matrix-identity.json")
            matrix.require(
                stamp.get("schema_version") == 1 and stamp.get("block_size") == block,
                "invalid block stamp",
            )
            matrix.require(
                stamp.get("code_sha256") == source, "stale matrix source identity"
            )
            matrix.require(
                stamp.get("model_roots")
                == {"standard": str(standard.resolve()), "off": str(off.resolve())},
                "model roots differ from stamp",
            )
            config, _, _ = verify_build(root, variant, block)
            matrix.require(
                stamp.get("configuration") == config,
                "matrix compile macro/config changed",
            )
            for record in stamp["artifact_sha256"].values():
                verify_record(record, rehash=True)
            for record in stamp["model_manifests"].values():
                verify_record(record, rehash=True)
            for records in stamp["models"].values():
                for record in records.values():
                    verify_record(record)
            matrix.require(
                stamp.get("eos_token_ids") == eos_ids(standard),
                "EOS vocabulary changed",
            )
            supported[block] = stamp
        except (
            OSError,
            KeyError,
            ValueError,
            build_identity.BuildIdentityError,
        ) as error:
            failures[str(block)] = str(error)
    return supported, failures


def capabilities(root, variant_root, standard, off):
    supported, failures = inspect_variants(root, variant_root, standard, off)
    identity = None
    if supported:
        first = next(iter(supported.values()))
        identity = {
            "code_sha256": first["code_sha256"],
            "weight_sha256": role_hashes(first["models"]["standard"]),
        }
        matrix.require(
            all(
                role_hashes(stamp["models"]["standard"]) == identity["weight_sha256"]
                for stamp in supported.values()
            ),
            "variants do not share the same actual weights",
        )
    return {
        "protocol": matrix.PROTOCOL,
        "status": "compiled_unverified" if supported else "unsupported",
        "identity": identity,
        "eos_token_ids": next(iter(supported.values()))["eos_token_ids"]
        if supported
        else [],
        "supported_block_sizes": sorted(supported),
        "correctness_verified_block_sizes": [],
        "unsupported_variants": failures,
        "kv_formats": list(matrix.KV_FORMATS) if supported else [],
        "ane_modes": ["off", "on"] if supported else [],
        "drafter_states": (["off"] if 8 in supported else []) + sorted(supported),
        "output_tokens": [512, 2048],
        "tail_prefill_splits": list(range(1, 17)),
        "concurrency": [1, 2, 4],
        "ignore_eos": True,
        "cold_requests": True,
        "actual_batch_widths": True,
        "no_auto_calibration": True,
        "max_context_tokens": 36864,
        "max_memory_budget_bytes": 32 << 30,
        "starts_inference": False,
        "ane_capability_note": "requested pins supported; actual evaluations and fallback must be validated",
    }


def tokenizer(model_root):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_root / "tokenizer"), local_files_only=True, trust_remote_code=False
    )


def prepare_inputs(model_root, output):
    from dev.tests.summary_32k_real import FACTS, content_at_size

    codec = tokenizer(model_root)
    prompts = []
    for index, nonce in enumerate(
        ("甲-矩阵-01", "乙-矩阵-02", "丙-矩阵-03", "丁-矩阵-04")
    ):
        content = content_at_size(codec, nonce)
        tokens = codec.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
            reasoning_effort="none",
        )
        prompts.append(
            {
                "id": f"prompt-{index}",
                "nonce": nonce,
                "content": content,
                "token_ids": tokens,
                "input_sha256": matrix.digest(tokens),
                "expected_facts": [fact for facts in FACTS.values() for fact in facts],
            }
        )
    with Path(output).open("x") as stream:
        json.dump(
            {
                "schema_version": 1,
                "model_root": str(model_root.resolve()),
                "tokenizer_sha256": sha_file(model_root / "tokenizer/tokenizer.json"),
                "prompts": prompts,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    matrix.load_inputs(output, matrix.read_json(matrix.MANIFEST))
    return {
        "mode": "prepare_inputs",
        "starts_inference": False,
        "input_tokens": 32768,
        "prompts": matrix.input_identities(prompts),
    }


def validate_short(raw, payload):
    p, prompts = payload["job"]["parameters"], payload["prompts"]
    matrix.require(
        raw.get("status") == "complete" and raw.get("parameters") == p,
        "short batch did not complete requested parameters",
    )
    matrix.require(
        "timings" not in raw
        and "warmup" not in raw
        and all("timings" not in r for r in raw.get("requests", [])),
        "correctness emitted performance timings",
    )
    matrix.require(
        len(raw.get("requests", [])) == len(prompts) == p["concurrency"],
        "short request count differs",
    )
    widths = raw.get("actual_batch_widths", [])
    matrix.require(
        widths
        and max(widths) == len(prompts)
        and all(type(w) is int and 1 <= w <= len(prompts) for w in widths),
        "did_not_achieve_concurrency in short fixture",
    )
    matrix.require(
        raw.get("actual_drafter_block_size") == p["drafter_block_size"]
        and raw.get("actual_tail_prefill_splits") == p["tail_prefill_splits"]
        and type(raw.get("tail_plan_calls")) is int
        and raw["tail_plan_calls"] > 0,
        "short block/tail split override was not exercised",
    )
    for response, prompt in zip(raw["requests"], prompts, strict=True):
        matrix.require(
            response.get("prompt_id") == prompt["id"]
            and response.get("input_sha256") == prompt["input_sha256"],
            "short input identity differs",
        )
        matrix.require(
            len(response.get("output_token_ids", [])) == p["output_tokens"]
            and response.get("cached_input_tokens") == 0,
            "short forced output/cold cache assertion failed",
        )
        text = response.get("output_text")
        matrix.require(
            isinstance(text, str), "short correctness needs decoded output text"
        )
        missing = matrix.missing_facts(text, prompt["expected_facts"])
        response["facts_pass"] = not missing
        response["missing_facts"] = missing
        matrix.require(
            not missing, "short arithmetic expected fact42 was not preserved"
        )
    raw["facts_pass"] = all(response["facts_pass"] for response in raw["requests"])
    raw["mode"] = "correctness"
    return raw


def run_batch(payload, root, variant_root, standard, off):
    mode = payload.get("mode")
    matrix.require(
        mode in ("correctness", "performance"),
        "explicit correctness/performance mode is required",
    )
    p = payload["job"]["parameters"]
    matrix.require(
        p.get("auto_calibration") is False
        and p.get("ignore_eos") is True
        and p.get("cold_requests") is True
        and p.get("temperature") == 0,
        "batch must pin calibration off, ignored EOS, cold cache and greedy sampling",
    )
    if mode == "correctness":
        matrix.require(
            0 < p["input_tokens"] <= 128 and 0 < p["output_tokens"] <= 32,
            "correctness fixture limited to input128/output32",
        )
    else:
        matrix.require(
            p["input_tokens"] == 32768 and p["output_tokens"] in (512, 2048),
            "invalid performance lengths",
        )
    prompts = payload.get("prompts", [])
    matrix.require(
        len(prompts) == p["concurrency"] and 1 <= len(prompts) <= 4,
        "batch concurrency mismatch",
    )
    matrix.require(
        len({tuple(prompt["token_ids"][:32]) for prompt in prompts}) == len(prompts),
        "batch shares an input prefix page",
    )
    for prompt in prompts:
        matrix.require(
            len(prompt["token_ids"]) == p["input_tokens"]
            and matrix.digest(prompt["token_ids"]) == prompt["input_sha256"],
            "batch input hash/length mismatch",
        )
    caps = capabilities(root, variant_root, standard, off)
    block = p["drafter_block_size"] if p["drafter_enabled"] else 8
    if block not in caps["supported_block_sizes"]:
        return {
            "status": "unsupported",
            "identity": caps["identity"],
            "parameters": p,
            "reasons": [
                caps["unsupported_variants"].get(str(block), "missing reviewed variant")
            ],
        }
    matrix.require(
        payload.get("identity") == caps["identity"],
        "requested code/weight identity changed",
    )
    variant = variant_root / f"block-{block}"
    stamp = matrix.read_json(variant / "matrix-identity.json")
    model_root = standard if p["drafter_enabled"] else off
    native = variant / "engine-tests/qwen4b-matrix-native"
    library = variant / "splash.metallib"
    forwarded = payload | {
        "model_root": str(model_root.resolve()),
        "metallib": str(library.resolve()),
    }
    command = [str(native.resolve())]
    result = subprocess.run(
        command,
        input=json.dumps(forwarded),
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    matrix.require(
        result.returncode == 0,
        f"native exit{result.returncode}: {result.stderr[-4000:]}",
    )
    raw = json.loads(result.stdout)
    matrix.require(
        raw.get("compiled_block_size") == block,
        "native actual compiled block differs from stamp",
    )
    matrix.require(
        raw.get("compiled_source_sha256")
        == build_identity.build_id(
            root, constants=stamp["configuration"]["constants"]
        ).removeprefix("src-"),
        "native embedded production identity differs from reviewed build",
    )
    matrix.require(
        raw.get("actual_kv_format") == p["kv_format"]
        and raw.get("actual_context_tokens") == p["context_tokens"],
        "native actual KV/context differs from requested matrix settings",
    )
    matrix.require(
        isinstance(raw.get("device_name"), str)
        and raw["device_name"]
        and isinstance(raw.get("model_layout_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", raw["model_layout_sha256"]),
        "native device or actual model layout identity is missing",
    )
    matrix.require(
        raw.get("identity") == caps["identity"], "native code/weight identity differs"
    )
    codec = tokenizer(model_root)
    for response in raw.get("requests", []):
        ids = response["output_token_ids"]
        response["output_text"] = codec.decode(ids, skip_special_tokens=False)
        observed = [
            index for index, token in enumerate(ids) if token in caps["eos_token_ids"]
        ]
        matrix.require(
            response.get("eos_positions") == observed,
            "native EOS observations differ from model vocabulary",
        )
    raw["commands"] = [command]
    raw["loaded_weight_sha256"] = role_hashes(
        stamp["models"]["standard" if p["drafter_enabled"] else "off"]
    )
    raw["artifact_sha256"] = {
        name: record["sha256"] for name, record in stamp["artifact_sha256"].items()
    }
    raw["eos_token_ids"] = caps["eos_token_ids"]
    if mode == "correctness":
        try:
            validate_short(raw, payload)
        except matrix.MatrixError as error:
            error.raw_response = raw
            raise
    return raw


def short_payload(model_root, block, identity, concurrency=4, tail_splits=1):
    matrix.require(
        concurrency in (1, 2, 4) and tail_splits in range(1, 17),
        "invalid short fixture axes",
    )
    codec = tokenizer(model_root)
    space = codec.encode(" ", add_special_tokens=False)
    matrix.require(len(space) == 1, "short fixture needs one-token space padding")
    prompts = []
    for index, nonce in enumerate(("甲", "乙", "丙", "丁")[:concurrency]):
        ids = codec.apply_chat_template(
            [{"role": "user", "content": f"{nonce}：计算17+25，只回答结果。"}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
            reasoning_effort="none",
        )
        end = ids.index(248046)
        matrix.require(len(ids) <= 64, "short math template exceeds 64 tokens")
        ids[end:end] = space * (64 - len(ids))
        prompts.append(
            {
                "id": f"short-{index}",
                "token_ids": ids,
                "input_sha256": matrix.digest(ids),
                "expected_facts": ["42"],
            }
        )
    config = matrix.read_json(matrix.MANIFEST)
    parameters = next(matrix.groups(config))["parameters"] | {
        "input_tokens": 64,
        "output_tokens": 32,
        "concurrency": concurrency,
        "tail_prefill_splits": tail_splits,
        "drafter_enabled": block != "off",
        "drafter_block_size": None if block == "off" else int(block),
    }
    return {
        "protocol": matrix.PROTOCOL,
        "operation": "run_batch",
        "mode": "correctness",
        "job": {
            "group_id": matrix.digest(parameters),
            "repeat": 0,
            "parameters": parameters,
        },
        "prompts": prompts,
        "fresh_runtime": True,
        "identity": identity,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-root", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--model-root", type=Path, default=STANDARD)
    parser.add_argument("--off-model-root", type=Path, default=OFF)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stamp-variants", action="store_true")
    action.add_argument("--supported-block-sizes", action="store_true")
    action.add_argument("--prepare-inputs", type=Path)
    action.add_argument("--run-correctness", type=Path)
    parser.add_argument("--block", choices=("off", "2", "4", "8", "16"), default="8")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--tail-splits", type=int, choices=range(1, 17), default=1)
    args = parser.parse_args(argv)
    if args.stamp_variants:
        document = stamp_variants(
            ROOT, args.variant_root, args.model_root, args.off_model_root
        )
    elif args.prepare_inputs:
        document = prepare_inputs(args.model_root, args.prepare_inputs)
    elif args.supported_block_sizes:
        document = capabilities(
            ROOT, args.variant_root, args.model_root, args.off_model_root
        )
    elif args.run_correctness:
        caps = capabilities(
            ROOT, args.variant_root, args.model_root, args.off_model_root
        )
        model_root = args.off_model_root if args.block == "off" else args.model_root
        payload = short_payload(
            model_root, args.block, caps["identity"], args.concurrency, args.tail_splits
        )
        try:
            document = run_batch(
                payload, ROOT, args.variant_root, args.model_root, args.off_model_root
            )
        except matrix.MatrixError as error:
            if not hasattr(error, "raw_response"):
                raise
            document = error.raw_response | {"status": "incorrect", "error": str(error)}
        with args.run_correctness.open("x") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"mode": "correctness", "status": document["status"]}))
        return 0 if document["status"] == "complete" else 2
    else:
        payload = json.load(sys.stdin)
        matrix.require(
            payload.get("protocol") == matrix.PROTOCOL, "unknown adapter protocol"
        )
        if payload.get("operation") == "capabilities":
            document = capabilities(
                ROOT, args.variant_root, args.model_root, args.off_model_root
            )
        else:
            matrix.require(
                payload.get("operation") == "run_batch", "unknown adapter operation"
            )
            document = run_batch(
                payload, ROOT, args.variant_root, args.model_root, args.off_model_root
            )
    print(json.dumps(document, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
