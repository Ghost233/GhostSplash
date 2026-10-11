"""Optional proposal/sampling features: correctness only, never throughput."""

import argparse
import copy
import hashlib
import json
import os
import uuid
from pathlib import Path

from dev.benchmarks import qwen4b_ffn_cache as isolation
from dev.benchmarks import qwen4b_matrix_adapter as adapter
from dev.tools.compact_draft_ranking import write_ranking

ROOT = Path(__file__).resolve().parents[2]
FLAGS = {
    "lookup": "SPLASH_PROMPT_LOOKUP",
    "compact": "SPLASH_COMPACT_DRAFT_HEAD",
    "fast": "SPLASH_FAST_TOPK",
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def first_difference(left, right):
    for index, pair in enumerate(zip(left, right, strict=True)):
        if pair[0] != pair[1]:
            return {"index": index, "off": pair[0], "on": pair[1]}
    return None


def run_case(args, base, name, enabled, ranking=None):
    attempt = args.output / (name + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir()
    payload = copy.deepcopy(base)
    payload["model_root"] = str(args.model_root.resolve())
    payload["metallib"] = str((args.build / "splash.metallib").resolve())
    payload["mode"] = "correctness"
    payload["optional_features_correctness"] = True
    payload["identity"] = {"code_sha256": adapter.code_sha256(ROOT)}
    payload.pop("ffn_long_correctness", None)
    write(attempt / "payload.json", payload)
    keys = list(FLAGS.values()) + ["SPLASH_COMPACT_DRAFT_RANKING"]
    prior = {key: os.environ.get(key) for key in keys}
    try:
        for feature, key in FLAGS.items():
            os.environ[key] = str(int(feature in enabled))
        if ranking:
            os.environ["SPLASH_COMPACT_DRAFT_RANKING"] = str(ranking.resolve())
        else:
            os.environ.pop("SPLASH_COMPACT_DRAFT_RANKING", None)
        evidence = isolation.checked_process(
            [str((args.build / "engine-tests/qwen4b-matrix-native").resolve())],
            json.dumps(payload),
            attempt / "stdout.json",
            attempt / "stderr.log",
            correctness=True,
            allow_concurrent_inference=args.allow_concurrent_inference,
        )
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    evidence["source_sha256"] = adapter.code_sha256(ROOT)
    write(attempt / "evidence.json", evidence)
    if evidence["status"] != "complete" or evidence["exit_code"] != 0:
        raise RuntimeError(f"{name}: process correctness failed, see {attempt}")
    raw = json.loads((attempt / "stdout.json").read_text())
    stamp = json.loads((args.build / "engine/build-identity.json").read_text())
    if raw["compiled_source_sha256"] != stamp["source_sha256"]:
        raise RuntimeError("native source identity mismatch")
    codec = adapter.tokenizer(args.model_root)
    for response in raw["requests"]:
        response["output_text"] = codec.decode(
            response["output_token_ids"], skip_special_tokens=False
        )
    adapter.validate_short(raw, payload)
    parameters = payload["job"]["parameters"]
    if (
        parameters["ane_requested"]
        and parameters["input_tokens"] >= parameters["ane_minimum_rows"]
    ):
        actual = raw["ane_actual"]
        if (
            not actual["enabled"]
            or actual["share"] != parameters["ane_share"]
            or actual["minimum_rows"] != parameters["ane_minimum_rows"]
        ):
            raise RuntimeError(
                "requested ANE correctness path did not run: "
                + actual["fallback_reason"]
            )
    if raw["parameters"].get("ffn_weight_mode") == "quantized":
        isolation.validate_memory(raw["memory_report"], "quantized")
    write(attempt / "result.json", raw)
    return raw, attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--allow-concurrent-inference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.reference_payload.read_text())
    base["job"]["parameters"]["ffn_weight_mode"] = "quantized"
    base["job"]["parameters"]["ffn_prefill_variant"] = "baseline"
    base["job"]["parameters"]["ffn_cache_budget_bytes"] = 0
    base["job"]["parameters"]["ffn_stage_profile"] = False
    base["job"]["parameters"]["temperature"] = 0
    codec = adapter.tokenizer(args.model_root)
    if args.long:
        cases = [("32k-512", 512), ("32k-2048", 2048)]
    else:
        content = (
            "Copy the text between BEGIN and END exactly. Do not add any introduction.\n"
            "BEGIN\n"
            + "".join(
                f'Record {i:03d}: apples = 42; schema = {{"id": {i}, "enabled": true}}.\n'
                for i in range(80)
            )
            + "END"
        )
        tokens = codec.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(tokens, list) or not all(
            type(token) is int for token in tokens
        ):
            raise RuntimeError("copy fixture must contain actual integer token IDs")
        if "Record 000" not in codec.decode(tokens):
            raise RuntimeError("copy fixture text is absent from encoded input")
        space = codec.encode(" ", add_special_tokens=False)
        if len(space) != 1:
            raise RuntimeError("copy fixture needs one-token space padding")
        tail = len(tokens) % 2048
        padding = (64 - tail) % 2048
        end = tokens.index(248046)
        tokens[end:end] = space * padding
        base["job"]["parameters"]["input_tokens"] = len(tokens)
        base["prompts"] = [
            {
                "id": "copy-json-42",
                "content": content,
                "token_ids": tokens,
                "input_sha256": hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode()
                ).hexdigest(),
                "expected_facts": ["42"],
            }
        ]
        cases = [("copy-256", 256)]
    report = {"mode": "correctness", "performance_usable": False, "cases": []}
    for case, output in cases:
        base["job"]["parameters"]["output_tokens"] = output
        print("START", case, "off", flush=True)
        off, off_path = run_case(args, base, case + "-off", ())
        if off["optional_features"]["extra_metal_bytes"] != 0:
            raise RuntimeError("all-off path allocated optional Metal buffers")
        prompt_tokens = base["prompts"][0]["token_ids"]
        preferred = list(
            dict.fromkeys(prompt_tokens + off["requests"][0]["output_token_ids"])
        )
        vocabulary = len(codec)
        preferred_set = set(preferred)
        order = preferred + [
            token for token in range(vocabulary) if token not in preferred_set
        ]
        ranking = args.output / (case + "-correctness-only-ranking.splash")
        write_ranking(ranking, off["target_manifest_sha256"], order)
        write(
            ranking.with_suffix(".json"),
            {
                "purpose": "synthetic correctness fixture, not a popularity ranking",
                "target_manifest_sha256": off["target_manifest_sha256"],
                "preferred_token_count": len(preferred),
            },
        )
        for name, enabled in [
            ("lookup", ("lookup",)),
            ("compact", ("compact",)),
            ("fast", ("fast",)),
            ("combined", ("lookup", "compact", "fast")),
        ]:
            print("START", case, name, flush=True)
            on, path = run_case(args, base, case + "-" + name, enabled, ranking)
            divergence = first_difference(
                off["requests"][0]["output_token_ids"],
                on["requests"][0]["output_token_ids"],
            )
            if (
                name in ("compact", "combined")
                and not on["optional_features"]["compact_cycles"]
            ):
                raise RuntimeError("compact head was enabled but never exercised")
            if (
                not args.long
                and name in ("lookup", "combined")
                and not on["optional_features"]["lookup_hits"]
            ):
                raise RuntimeError("copy lookup was enabled but never exercised")
            record = {
                "case": case,
                "features": name,
                "off": str(off_path),
                "on": str(path),
                "first_difference": divergence,
                "actual": on["optional_features"],
                "passed": divergence is None,
            }
            report["cases"].append(record)
            write(args.output / "report.json", report)
            print("END", case, name, "PASS" if record["passed"] else "FAIL", flush=True)
            if divergence:
                raise RuntimeError(
                    f"{case}/{name}: first token divergence {divergence}"
                )
        if not args.long:
            # Different proposal Q can change seeded sample trajectories.
            # Repetition is checked here; distribution and residual correctness
            # are checked against the FP64 oracle in target-sampling.
            base["job"]["parameters"].update(
                temperature=0.8, top_k=20, top_p=0.95, seed=42, output_tokens=64
            )
            sampled, sampled_path = run_case(
                args, base, "sampled-combined", ("lookup", "compact", "fast"), ranking
            )
            repeated, repeated_path = run_case(
                args,
                base,
                "sampled-combined-repeat",
                ("lookup", "compact", "fast"),
                ranking,
            )
            if first_difference(
                sampled["requests"][0]["output_token_ids"],
                repeated["requests"][0]["output_token_ids"],
            ):
                raise RuntimeError("sampled output is not reproducible")
            if not sampled["optional_features"]["fast_topk_selections"]:
                raise RuntimeError("sampled fast top-k was not exercised")
            report["sampled_repeat"] = {
                "passed": True,
                "first": str(sampled_path),
                "repeat": str(repeated_path),
                "actual": sampled["optional_features"],
            }
    report["status"] = "passed"
    report["source_sha256"] = adapter.code_sha256(ROOT)
    write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
