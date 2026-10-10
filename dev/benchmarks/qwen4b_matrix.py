"""Plan the six-axis 4B matrix without starting inference by default.

--validate-correctness validates previously recorded responses offline. Only
--run-performance invokes a backend. The default repository adapter implements
qwen4b-matrix-v1 JSON stdin/stdout operations capabilities and run_batch. Missing capabilities yield unsupported records, never guessed
settings. An ANE request that falls back is recorded as unsupported.

Input JSON contains exactly four prompts with id, token_ids and expected_facts.
All have 32768 tokens and distinct first 32-token pages; a shared chat BOS
is allowed. Every request must still report zero cached input tokens.
The backend must echo identities/parameters and report actual output IDs, EOS,
cache reuse, ANE usage, batch widths, timings and memory. Actual output counts
must reach 512/2048 even when EOS occurs earlier.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

MANIFEST = Path(__file__).with_name("qwen4b_matrix_manifest.json")
PROTOCOL = "qwen4b-matrix-v1"
KV_FORMATS = ("int8", "bf16", "q4_0", "q4_0_unrotated", "q4h")
DRAFTER_STATES = ("off", 2, 4, 8, 16)


class MatrixError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MatrixError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_manifest(config):
    expected = {
        "schema_version": 1,
        "input_tokens": 32768,
        "output_tokens": [512, 2048],
        "kv_formats": list(KV_FORMATS),
        "ane_modes": ["off", "on"],
        "drafter_states": list(DRAFTER_STATES),
        "tail_prefill_splits": list(range(1, 17)),
        "concurrency": [1, 2, 4],
        "repeats": 3,
        "context_tokens": 36864,
        "memory_budget_bytes": 32 << 30,
        "temperature": 0,
        "ignore_eos": True,
        "cold_requests": True,
        "allow_shared_prefix_tokens": 31,
        "auto_calibration": False,
        "backend_protocol": PROTOCOL,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"manifest {key} must be {value!r}")
    require(
        config.get("ane_on") == {"share": 0.5, "minimum_rows": 512},
        "ANE on must be pinned to share0.5/minimum_rows512; no calibration",
    )
    return config


def groups(config):
    validate_manifest(config)
    for kv, ane, draft, output, splits, width in itertools.product(
        config["kv_formats"],
        config["ane_modes"],
        config["drafter_states"],
        config["output_tokens"],
        config["tail_prefill_splits"],
        config["concurrency"],
    ):
        parameters = {
            "kv_format": kv,
            "ane_requested": ane == "on",
            "ane_share": config["ane_on"]["share"] if ane == "on" else 0,
            "ane_minimum_rows": config["ane_on"]["minimum_rows"] if ane == "on" else 0,
            "drafter_enabled": draft != "off",
            "drafter_block_size": None if draft == "off" else draft,
            "input_tokens": config["input_tokens"],
            "output_tokens": output,
            "tail_prefill_splits": splits,
            "concurrency": width,
            "context_tokens": config["context_tokens"],
            "memory_budget_bytes": config["memory_budget_bytes"],
            "temperature": 0,
            "ignore_eos": True,
            "cold_requests": True,
            "auto_calibration": False,
        }
        yield {"group_id": digest(parameters), "parameters": parameters}


def plan(config, inputs=None):
    rows = list(groups(config))
    return {
        "schema_version": 1,
        "mode": "plan",
        "starts_inference": False,
        "group_count": len(rows),
        "batch_count": len(rows) * config["repeats"],
        "request_count": sum(row["parameters"]["concurrency"] for row in rows)
        * config["repeats"],
        "manifest": config,
        "inputs_status": "validated" if inputs else "unprepared",
        "inputs": input_identities(inputs) if inputs else [],
        "backend_status": "not probed; repository adapter capabilities required",
        "groups": rows,
    }


def load_inputs(path, config):
    prompts = read_json(path).get("prompts", [])
    require(len(prompts) == 4, "exactly four input prompts are required")
    ids, hashes, first_pages = set(), set(), set()
    for prompt in prompts:
        name, tokens, facts = (
            prompt.get("id"),
            prompt.get("token_ids"),
            prompt.get("expected_facts"),
        )
        require(
            isinstance(name, str) and name and name not in ids,
            "unique input id required",
        )
        require(
            isinstance(tokens, list) and len(tokens) == config["input_tokens"],
            "every input must contain exactly 32768 tokens",
        )
        require(
            all(type(token) is int and 0 <= token < 2**32 for token in tokens),
            "input token IDs must be unsigned integers",
        )
        require(
            isinstance(facts, list)
            and facts
            and all(isinstance(fact, str) and fact.strip() for fact in facts),
            "each prompt needs expected_facts",
        )
        actual = digest(tokens)
        require(prompt.get("input_sha256", actual) == actual, "input hash mismatch")
        require(
            actual not in hashes and tuple(tokens[:32]) not in first_pages,
            "cold prompt bank must have distinct first 32-token pages",
        )
        prompt["input_sha256"] = actual
        ids.add(name)
        hashes.add(actual)
        first_pages.add(tuple(tokens[:32]))
    return prompts


def input_identities(inputs):
    return [
        {
            "id": p["id"],
            "input_sha256": p["input_sha256"],
            "input_tokens": len(p["token_ids"]),
            "expected_facts": p["expected_facts"],
        }
        for p in inputs
    ]


def unsupported(parameters, capabilities):
    requirements = {
        "kv_formats": parameters["kv_format"],
        "ane_modes": "on" if parameters["ane_requested"] else "off",
        "drafter_states": parameters["drafter_block_size"]
        if parameters["drafter_enabled"]
        else "off",
        "output_tokens": parameters["output_tokens"],
        "tail_prefill_splits": parameters["tail_prefill_splits"],
        "concurrency": parameters["concurrency"],
    }
    reasons = [
        f"unsupported {key}={value!r}"
        for key, value in requirements.items()
        if value not in capabilities.get(key, [])
    ]
    if parameters["drafter_enabled"] and parameters[
        "drafter_block_size"
    ] not in capabilities.get("supported_block_sizes", []):
        reasons.append("native variant does not support requested draft block size")
    for key in (
        "ignore_eos",
        "cold_requests",
        "actual_batch_widths",
        "no_auto_calibration",
    ):
        if capabilities.get(key) is not True:
            reasons.append(f"backend does not guarantee {key}")
    for key, actual_key in (
        ("context_tokens", "max_context_tokens"),
        ("memory_budget_bytes", "max_memory_budget_bytes"),
    ):
        limit = capabilities.get(actual_key)
        if type(limit) is not int or limit < parameters[key]:
            reasons.append(f"backend limit does not cover {key}={parameters[key]}")
    return reasons


def validate_identity(identity):
    require(isinstance(identity, dict), "backend identity is missing")
    require(
        isinstance(identity.get("code_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", identity["code_sha256"]),
        "record a real code_sha256",
    )
    weights = identity.get("weight_sha256", {})
    require(
        isinstance(weights, dict)
        and "target" in weights
        and all(
            isinstance(name, str)
            and isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in weights.items()
        ),
        "record real weight SHA-256 identities",
    )


def missing_facts(text, facts):
    text = "".join(unicodedata.normalize("NFKC", text).split())
    return [
        fact
        for fact in facts
        if not re.search(
            (r"(?<!\d)" if fact[0].isdigit() else "")
            + re.escape("".join(unicodedata.normalize("NFKC", fact).split())),
            text,
        )
    ]


def positive_seconds(value, label):
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        f"invalid {label}",
    )
    return value


def validate_batch(raw, job, inputs, identity, eos_ids, performance=False):
    parameters = job["parameters"]
    validate_identity(identity)
    require(isinstance(raw, dict), "backend response must be a JSON object")
    require(
        not parameters["drafter_enabled"] or "draft" in identity["weight_sha256"],
        "enabled drafter needs its weight SHA-256 identity",
    )
    require(
        raw.get("identity") == identity, "code/weight identity changed during matrix"
    )
    require(
        raw.get("parameters") == parameters,
        "backend changed requested matrix parameters",
    )
    status = raw.get("status")
    base = {
        "group_id": job["group_id"],
        "repeat": job["repeat"],
        "mode": "performance" if performance else "correctness",
        "identity": identity,
        "parameters": parameters,
    }
    if status == "unsupported":
        require(
            isinstance(raw.get("reasons"), list) and raw["reasons"],
            "unsupported backend result needs reasons",
        )
        return base | {"status": "unsupported", "reasons": raw["reasons"]}
    require(status == "complete", "backend batch did not complete")
    require(
        raw.get("actual_kv_format") == parameters["kv_format"]
        and raw.get("actual_context_tokens") == parameters["context_tokens"]
        and isinstance(raw.get("device_name"), str)
        and raw["device_name"]
        and isinstance(raw.get("model_layout_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", raw["model_layout_sha256"]),
        "actual KV/context/device/layout identity is missing or differs",
    )
    observed = raw.get("requests", [])
    require(
        len(observed) == parameters["concurrency"] == len(inputs),
        "request count mismatch",
    )
    require(
        raw.get("actual_drafter_block_size") == parameters["drafter_block_size"]
        and raw.get("actual_tail_prefill_splits") == parameters["tail_prefill_splits"],
        "actual block/split settings differ from requested settings",
    )
    commands = raw.get("commands", [])
    require(
        commands
        and all(
            isinstance(cmd, list) and cmd and all(isinstance(arg, str) for arg in cmd)
            for cmd in commands
        ),
        "record actual execution command argument arrays",
    )
    widths = raw.get("actual_batch_widths", [])
    require(
        widths
        and all(type(w) is int and 1 <= w <= len(inputs) for w in widths)
        and max(widths) == len(inputs),
        "did_not_achieve_concurrency or invalid actual decode batch width",
    )
    ane = raw.get("ane_actual", {})
    require(
        type(ane.get("enabled")) is bool, "record actual ANE enabled/fallback state"
    )
    require(
        ane.get("auto_calibration") is False, "automatic ANE calibration is forbidden"
    )
    require(
        type(ane.get("share")) in (int, float)
        and 0 <= ane["share"] < 1
        and type(ane.get("minimum_rows")) is int
        and ane["minimum_rows"] >= 0,
        "record actual ANE share and minimum_rows",
    )
    require(
        not ane["enabled"] or parameters["ane_requested"],
        "ANE enabled despite off request",
    )
    if ane["enabled"]:
        require(
            ane["share"] == parameters["ane_share"]
            and ane["minimum_rows"] == parameters["ane_minimum_rows"],
            "ANE pins changed",
        )
    elif parameters["ane_requested"]:
        require(
            isinstance(ane.get("fallback_reason"), str) and ane["fallback_reason"],
            "ANE on fallback needs a reason",
        )
    records = []
    for response, prompt in zip(observed, inputs, strict=True):
        require(
            response.get("prompt_id") == prompt["id"]
            and response.get("input_sha256") == prompt["input_sha256"]
            and response.get("input_tokens") == parameters["input_tokens"],
            "input identity mismatch",
        )
        tokens = response.get("output_token_ids", [])
        require(
            len(tokens) == parameters["output_tokens"]
            and all(type(t) is int and 0 <= t < 2**32 for t in tokens),
            "actual forced output count mismatch",
        )
        positions = [i for i, token in enumerate(tokens) if token in eos_ids]
        require(
            response.get("eos_positions") == positions,
            "EOS observations differ from output IDs",
        )
        require(
            response.get("cached_input_tokens") == 0,
            "cold request reused cached prefix",
        )
        text = response.get("output_text")
        require(isinstance(text, str), "output text is needed for factual checks")
        missing = missing_facts(text, prompt["expected_facts"])
        peak = response.get("memory_peak_bytes")
        require(
            type(peak) is int and 0 < peak <= parameters["memory_budget_bytes"],
            "request memory peak missing or over budget",
        )
        record = {
            "prompt_id": prompt["id"],
            "input_sha256": prompt["input_sha256"],
            "input_tokens": parameters["input_tokens"],
            "output_tokens": len(tokens),
            "output_sha256": digest(tokens),
            "first_eos_position": positions[0] if positions else None,
            "eos_count": len(positions),
            "eos_rate": len(positions) / len(tokens),
            "missing_facts": missing,
            "facts_pass": not missing,
            "memory_peak_bytes": peak,
            "cached_input_tokens": 0,
        }
        if performance:
            timings = response.get("timings", {})
            prefill = positive_seconds(
                timings.get("prefill_wall_seconds"), "request prefill seconds"
            )
            decode = positive_seconds(
                timings.get("decode_wall_seconds"), "request decode seconds"
            )
            total = positive_seconds(
                timings.get("total_wall_seconds"), "request total seconds"
            )
            require(
                total + 1e-9 >= prefill + decode,
                "request total is shorter than its phases",
            )
            record["timings"] = timings
            record["prefill_tok_s"] = parameters["input_tokens"] / prefill
            record["decode_tok_s"] = len(tokens) / decode
        records.append(record)
    peak = raw.get("memory_peak_bytes")
    require(
        type(peak) is int and 0 < peak <= parameters["memory_budget_bytes"],
        "batch exceeds memory budget",
    )
    base |= {
        "status": "unsupported"
        if parameters["ane_requested"] and not ane["enabled"]
        else "complete",
        "ane_actual": ane,
        "commands": commands,
        "actual_batch_widths": widths,
        "requests": records,
        "facts_pass": all(r["facts_pass"] for r in records),
        "memory_peak_bytes": peak,
        "actual_settings": {
            key: raw.get(key)
            for key in (
                "compiled_block_size",
                "actual_kv_format",
                "actual_context_tokens",
                "device_name",
                "model_layout_sha256",
            )
        },
        "tail_plan_calls": raw.get("tail_plan_calls"),
        "drafted_tokens": raw.get("drafted_tokens"),
        "accepted_draft_tokens": raw.get("accepted_draft_tokens"),
    }
    drafted, accepted = raw.get("drafted_tokens"), raw.get("accepted_draft_tokens")
    if type(drafted) is int and type(accepted) is int and 0 <= accepted <= drafted:
        base["acceptance_rate"] = accepted / drafted if drafted else None
    if not base["facts_pass"]:
        base["status"] = "incorrect"
    if parameters["ane_requested"] and not ane["enabled"]:
        base["reasons"] = ["ANE on request fell back: " + ane["fallback_reason"]]
    if performance:
        expected_warmup = {
            "prefill_rows": 2048,
            "decode_width": len(inputs),
            "iterations": 1,
        }
        require(
            raw.get("warmup") == expected_warmup,
            "fixed performance warmup differs or is missing",
        )
        base["warmup"] = expected_warmup
        timing = raw.get("timings", {})
        prefill = positive_seconds(
            timing.get("prefill_wall_seconds"), "batch prefill seconds"
        )
        decode = positive_seconds(
            timing.get("decode_wall_seconds"), "batch decode seconds"
        )
        total = positive_seconds(
            timing.get("total_wall_seconds"), "batch total seconds"
        )
        require(
            total + 1e-9 >= prefill + decode, "batch total is shorter than its phases"
        )
        base["timing_definitions"] = {
            "request_prefill_wall_seconds": "submit_to_prompt_complete_including_queue",
            "aggregate_prefill_wall_seconds": "sum_of_Engine_prefill_cycle_seconds",
            "aggregate_decode_wall_seconds": "sum_of_Engine_decode_cycle_seconds",
        }
        base["aggregate"] = {
            "timings": timing,
            "prefill_tok_s": len(inputs) * parameters["input_tokens"] / prefill,
            "decode_tok_s": len(inputs) * parameters["output_tokens"] / decode,
            "request_tok_s": len(inputs)
            * (parameters["input_tokens"] + parameters["output_tokens"])
            / total,
        }
    return base


class Backend:
    def __init__(self, command):
        require(
            isinstance(command, list)
            and command
            and all(isinstance(v, str) and v for v in command),
            "backend command must be a nonempty JSON string array",
        )
        self.command = command

    def call(self, operation, payload):
        result = subprocess.run(
            self.command,
            input=json.dumps({"protocol": PROTOCOL, "operation": operation, **payload}),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        require(
            result.returncode == 0,
            f"backend exit {result.returncode}: {result.stderr[-2000:]}",
        )
        return json.loads(result.stdout)


def run_performance(config, prompts, backend, output):
    capabilities = backend.call("capabilities", {})
    identity, eos_ids = (
        capabilities.get("identity"),
        capabilities.get("eos_token_ids", []),
    )
    validate_identity(identity)
    require(
        eos_ids and all(type(t) is int and t >= 0 for t in eos_ids),
        "backend EOS IDs are missing",
    )
    counts = {"complete": 0, "unsupported": 0, "incorrect": 0, "invalid": 0}
    with Path(output).open("x") as stream:
        for group in groups(config):
            reasons = unsupported(group["parameters"], capabilities)
            for repeat in range(config["repeats"]):
                selected = [
                    prompts[(repeat + i) % 4]
                    for i in range(group["parameters"]["concurrency"])
                ]
                job = group | {"repeat": repeat}
                envelope = {
                    **job,
                    "identity": identity,
                    "inputs": input_identities(selected),
                    "mode": "performance",
                    "eos_token_ids": eos_ids,
                }
                raw = None
                try:
                    if reasons:
                        record = envelope | {
                            "status": "unsupported",
                            "reasons": reasons,
                        }
                    else:
                        raw = backend.call(
                            "run_batch",
                            {
                                "job": job,
                                "prompts": selected,
                                "fresh_runtime": True,
                                "mode": "performance",
                                "identity": identity,
                            },
                        )
                        record = validate_batch(
                            raw, job, selected, identity, eos_ids, performance=True
                        )
                        record["raw_response"] = raw
                except (MatrixError, ValueError, subprocess.SubprocessError) as error:
                    record = envelope | {
                        "status": "invalid",
                        "error": str(error),
                        "raw_response": raw,
                    }
                record["eos_token_ids"] = eos_ids
                counts[record["status"]] += 1
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
    return counts


def validate_correctness(records, config, prompts):
    require(records, "no recorded batches to validate")
    results = []
    valid_groups = {g["group_id"]: g for g in groups(config)}
    seen = set()
    first_identity = records[0].get("identity")
    for stored in records:
        require(
            stored.get("identity") == first_identity,
            "mixed code/weight identities in saved matrix",
        )
        raw = stored.get("raw_response") or stored
        group_id, repeat = stored.get("group_id"), stored.get("repeat")
        require(
            group_id in valid_groups
            and type(repeat) is int
            and 0 <= repeat < config["repeats"],
            "result does not belong to this matrix",
        )
        require((group_id, repeat) not in seen, "duplicate matrix batch result")
        seen.add((group_id, repeat))
        job = valid_groups[group_id] | {"repeat": repeat}
        selected = [
            prompts[(repeat + i) % 4] for i in range(job["parameters"]["concurrency"])
        ]
        eos_ids = stored.get("eos_token_ids", raw.get("eos_token_ids", []))
        require(eos_ids, "offline validation requires recorded EOS token IDs")
        results.append(
            validate_batch(
                raw,
                job,
                selected,
                stored.get("identity", raw.get("identity")),
                eos_ids,
                performance=False,
            )
        )
    return {
        "mode": "correctness",
        "starts_inference": False,
        "validated_batches": len(results),
        "expected_batches": 14400,
        "coverage_complete": len(results) == 14400,
        "matrix_complete": len(results) == 14400
        and all(r["status"] == "complete" and r.get("facts_pass") for r in results),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--validate-correctness", action="store_true")
    mode.add_argument("--run-performance", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument(
        "--variant-root", type=Path, default=Path("build/qwen4b-matrix-variants")
    )
    parser.add_argument(
        "--backend-command",
        help="override the repository adapter command as a JSON string array",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = validate_manifest(read_json(args.manifest))
    prompts = load_inputs(args.inputs, config) if args.inputs else None
    if args.run_performance:
        require(prompts, "performance requires validated real input prompts")
        command = (
            json.loads(args.backend_command)
            if args.backend_command
            else [
                sys.executable,
                "-m",
                "dev.benchmarks.qwen4b_matrix_adapter",
                "--variant-root",
                str(args.variant_root),
            ]
        )
        counts = run_performance(config, prompts, Backend(command), args.output)
        print(json.dumps({"mode": "performance", "batch_status_counts": counts}))
        return 0 if counts["complete"] == 14400 else 2
    if args.validate_correctness:
        require(
            prompts and args.results,
            "correctness validation requires inputs and saved results",
        )
        records = [
            json.loads(line) for line in args.results.read_text().splitlines() if line
        ]
        document = validate_correctness(records, config, prompts)
    else:
        document = plan(config, prompts)
    with args.output.open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in document.items()
                if key not in ("groups", "results", "manifest", "inputs")
            }
        )
    )
    return (
        0
        if not args.validate_correctness
        or all(
            r["status"] == "complete" and r.get("facts_pass")
            for r in document["results"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
