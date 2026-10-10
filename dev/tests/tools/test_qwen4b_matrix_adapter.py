import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.benchmarks import qwen4b_matrix as matrix
from dev.benchmarks import qwen4b_matrix_adapter as adapter
from dev.tools import build_identity


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in adapter.EXTRA_SOURCES + (
            "dev/tools/build_identity.py",
            "runtime/ops/PagedAttention.cpp",
        ):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(adapter.ROOT / name, destination)
        self.variant_root = self.root / "variants"
        self.standard, self.off = self.root / "standard", self.root / "off"
        target, draft = self.root / "target.gguf", self.root / "draft.safetensors"
        target.write_bytes(b"mocked-test-target-weights")
        draft.write_bytes(b"mocked-test-draft-weights")
        for model_root, enabled in ((self.standard, True), (self.off, False)):
            model_root.mkdir()
            records = {
                "target/target.gguf": {
                    "path": str(target),
                    "digest": adapter.sha_file(target),
                }
            }
            if enabled:
                records["draft/draft.safetensors"] = {
                    "path": str(draft),
                    "digest": adapter.sha_file(draft),
                }
            (model_root / "model.json").write_text(
                json.dumps({"family": "Qwen3.5-4B", "files": records})
            )
            (model_root / "tokenizer").mkdir()
            (model_root / "tokenizer/tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "eos_token": "<|im_end|>",
                        "added_tokens_decoder": {
                            "248044": {"content": "<|endoftext|>"},
                            "248046": {"content": "<|im_end|>"},
                        },
                    }
                )
            )
        for block in adapter.BLOCKS:
            self.make_variant(block)

    def make_variant(self, block):
        variant = self.variant_root / f"block-{block}"
        config = adapter.expected_configuration(self.root, block)
        (variant / "engine").mkdir(parents=True, exist_ok=True)
        (variant / "engine-tests").mkdir(exist_ok=True)
        (variant / "engine-tests/matrix-PagedAttention.cpp").write_text(
            adapter.expected_attention(self.root)
        )
        identity = build_identity.build_id(self.root, constants=config["constants"])
        (variant / "engine/build-identity.json").write_bytes(
            build_identity.stamp_bytes(identity)
        )
        (variant / "engine/BuildIdentity.hpp").write_bytes(
            build_identity.header_bytes(identity)
        )
        artifacts = {
            "engine/libsplash.a": config["production"],
            "engine-tests/matrix-PagedAttention.o": config["test"],
            "engine-tests/qwen4b-matrix-native": config["test"],
            "splash.metallib": config["metallib"],
        }
        for name, digest in artifacts.items():
            path = variant / name
            path.write_bytes(f"mocked compiler output block{block} {name}".encode())
            Path(str(path) + ".config").write_text(digest + "\n")
        return variant

    def stamp(self):
        result = adapter.stamp_variants(
            self.root, self.variant_root, self.standard, self.off
        )
        self.assertEqual(result["stamped_block_sizes"], [2, 4, 8, 16])
        return adapter.capabilities(
            self.root, self.variant_root, self.standard, self.off
        )

    def payload(self, mode="correctness", block=8):
        caps = self.stamp()
        parameters = next(matrix.groups(matrix.read_json(matrix.MANIFEST)))[
            "parameters"
        ] | {
            "input_tokens": 64,
            "output_tokens": 32,
            "concurrency": 4,
            "drafter_enabled": block is not None,
            "drafter_block_size": block,
        }
        prompts = [
            {
                "id": f"short-{i}",
                "token_ids": [5, i] + [7] * 62,
                "expected_facts": ["42"],
            }
            for i in range(4)
        ]
        for prompt in prompts:
            prompt["input_sha256"] = matrix.digest(prompt["token_ids"])
        return {
            "protocol": matrix.PROTOCOL,
            "operation": "run_batch",
            "mode": mode,
            "job": {
                "parameters": parameters,
                "repeat": 0,
                "group_id": matrix.digest(parameters),
            },
            "prompts": prompts,
            "identity": caps["identity"],
            "fresh_runtime": True,
        }

    def native_response(self, payload):
        p = payload["job"]["parameters"]
        requests = []
        for prompt in payload["prompts"]:
            tokens = [7] * p["output_tokens"]
            tokens[3] = 248046
            requests.append(
                {
                    "prompt_id": prompt["id"],
                    "input_sha256": prompt["input_sha256"],
                    "input_tokens": p["input_tokens"],
                    "output_token_ids": tokens,
                    "eos_positions": [3],
                    "cached_input_tokens": 0,
                    "memory_peak_bytes": 1024,
                }
            )
        return {
            "status": "complete",
            "identity": payload["identity"],
            "parameters": p,
            "compiled_block_size": p["drafter_block_size"] or 8,
            "compiled_source_sha256": build_identity.build_id(
                self.root,
                constants=adapter.expected_configuration(
                    self.root, p["drafter_block_size"] or 8
                )["constants"],
            ).removeprefix("src-"),
            "actual_kv_format": p["kv_format"],
            "actual_context_tokens": p["context_tokens"],
            "device_name": "Mocked test device",
            "model_layout_sha256": "e" * 64,
            "actual_drafter_block_size": p["drafter_block_size"],
            "actual_tail_prefill_splits": p["tail_prefill_splits"],
            "tail_plan_calls": 4,
            "actual_batch_widths": [4, 4, 2, 1],
            "requests": requests,
            "ane_actual": {
                "enabled": False,
                "share": 0,
                "minimum_rows": 512,
                "auto_calibration": False,
                "evaluations": 0,
            },
            "memory_peak_bytes": 2048,
            "drafted_tokens": 64,
            "accepted_draft_tokens": 32,
        }

    def test_capabilities_and_stamping_never_start_native(self):
        with mock.patch.object(adapter.subprocess, "run") as native:
            caps = self.stamp()
            native.assert_not_called()
        self.assertEqual(caps["supported_block_sizes"], [2, 4, 8, 16])
        self.assertEqual(
            caps["identity"]["weight_sha256"]["target"],
            adapter.sha_file(self.root / "target.gguf"),
        )
        self.assertEqual(
            caps["identity"]["weight_sha256"]["draft"],
            adapter.sha_file(self.root / "draft.safetensors"),
        )
        self.assertFalse(caps["starts_inference"])
        self.assertEqual(caps["status"], "compiled_unverified")
        self.assertEqual(caps["correctness_verified_block_sizes"], [])

    def test_missing_stamps_are_unsupported_without_gpu(self):
        with mock.patch.object(adapter.subprocess, "run") as native:
            caps = adapter.capabilities(
                self.root, self.variant_root, self.standard, self.off
            )
            native.assert_not_called()
        self.assertEqual(caps["status"], "unsupported")
        self.assertEqual(caps["supported_block_sizes"], [])
        self.assertIsNone(caps["identity"])
        self.assertEqual(len(caps["unsupported_variants"]), 4)

    def test_artifact_and_macro_drift_invalidate_capability(self):
        self.stamp()
        artifact = self.variant_root / "block-4/engine-tests/qwen4b-matrix-native"
        artifact.write_bytes(b"changed artifact")
        config = self.variant_root / "block-2/engine-tests/qwen4b-matrix-native.config"
        config.write_text(adapter.expected_configuration(self.root, 8)["test"] + "\n")
        caps = adapter.capabilities(
            self.root, self.variant_root, self.standard, self.off
        )
        self.assertEqual(caps["supported_block_sizes"], [8, 16])
        self.assertIn("stale", caps["unsupported_variants"]["2"])
        self.assertIn("stale", caps["unsupported_variants"]["4"])

    def test_stamping_cannot_pair_old_artifacts_with_new_override(self):
        source = self.root / "dev/benchmarks/qwen4b_matrix_attention.py"
        source.write_text(source.read_text() + "\n# source changed after compilation\n")
        result = adapter.stamp_variants(
            self.root, self.variant_root, self.standard, self.off
        )
        self.assertEqual(result["stamped_block_sizes"], [])
        self.assertTrue(
            all(
                "predates" in message
                for message in result["unsupported_variants"].values()
            )
        )

    def test_stale_generated_identity_is_rejected(self):
        path = self.variant_root / "block-8/engine/build-identity.json"
        path.write_bytes(build_identity.stamp_bytes("src-" + "f" * 64))
        result = adapter.stamp_variants(
            self.root, self.variant_root, self.standard, self.off
        )
        self.assertNotIn(8, result["stamped_block_sizes"])
        self.assertIn("build identity", result["unsupported_variants"]["8"])

    def test_stale_attention_snapshot_is_rejected(self):
        path = self.variant_root / "block-16/engine-tests/matrix-PagedAttention.cpp"
        path.write_text(
            path.read_text().replace("matrixTailSplits()", "notTheMatrixTail()", 1)
        )
        result = adapter.stamp_variants(
            self.root, self.variant_root, self.standard, self.off
        )
        self.assertNotIn(16, result["stamped_block_sizes"])
        self.assertIn("attention source snapshot", result["unsupported_variants"]["16"])

    def test_weight_changes_require_restamping_and_real_hash_match(self):
        self.stamp()
        path = self.root / "target.gguf"
        path.write_bytes(b"different target weights")
        caps = adapter.capabilities(
            self.root, self.variant_root, self.standard, self.off
        )
        self.assertEqual(caps["supported_block_sizes"], [])
        with self.assertRaisesRegex(matrix.MatrixError, "manifest digest differs"):
            adapter.stamp_variants(
                self.root, self.variant_root, self.standard, self.off
            )

    def test_long_correctness_and_missing_mode_never_call_native(self):
        for mode, count in ((None, 64), ("correctness", 129)):
            payload = {
                "mode": mode,
                "job": {
                    "parameters": {
                        "input_tokens": count,
                        "output_tokens": 32,
                        "auto_calibration": False,
                        "ignore_eos": True,
                        "cold_requests": True,
                        "temperature": 0,
                    }
                },
            }
            with mock.patch.object(adapter.subprocess, "run") as native:
                with self.assertRaises(matrix.MatrixError):
                    adapter.run_batch(
                        payload, self.root, self.variant_root, self.standard, self.off
                    )
                native.assert_not_called()

    def test_mock_native_mode_and_paths_are_forwarded_and_decode_keeps_specials(self):
        payload = self.payload()
        response = self.native_response(payload)
        codec = mock.Mock()
        codec.decode.return_value = "42<|im_end|>"
        result = SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")
        with (
            mock.patch.object(adapter.subprocess, "run", return_value=result) as native,
            mock.patch.object(adapter, "tokenizer", return_value=codec),
        ):
            raw = adapter.run_batch(
                payload, self.root, self.variant_root, self.standard, self.off
            )
        forwarded = json.loads(native.call_args.kwargs["input"])
        self.assertEqual(forwarded["mode"], "correctness")
        self.assertEqual(forwarded["model_root"], str(self.standard.resolve()))
        self.assertEqual(
            forwarded["metallib"],
            str((self.variant_root / "block-8/splash.metallib").resolve()),
        )
        self.assertEqual(codec.decode.call_args.kwargs, {"skip_special_tokens": False})
        self.assertEqual(raw["requests"][0]["output_text"], "42<|im_end|>")
        self.assertNotIn("timings", raw)
        self.assertEqual(raw["accepted_draft_tokens"], 32)

    def test_short_result_rejects_fake_compiled_block_or_perf_fields(self):
        payload = self.payload()
        for change in (
            lambda raw: raw.update(compiled_block_size=4),
            lambda raw: raw.update(compiled_source_sha256="0" * 64),
            lambda raw: raw.update(actual_kv_format="bf16"),
            lambda raw: raw.update(actual_context_tokens=32768),
            lambda raw: raw.update(timings={"decode_wall_seconds": 1}),
            lambda raw: raw.update(tail_plan_calls=0),
        ):
            raw = self.native_response(payload)
            change(raw)
            result = SimpleNamespace(returncode=0, stdout=json.dumps(raw), stderr="")
            with (
                mock.patch.object(adapter.subprocess, "run", return_value=result),
                mock.patch.object(adapter, "tokenizer", return_value=mock.Mock()),
            ):
                with self.assertRaises(matrix.MatrixError):
                    adapter.run_batch(
                        payload, self.root, self.variant_root, self.standard, self.off
                    )

    def test_four_submitted_short_requests_allow_decode_width_two(self):
        payload = self.payload()
        raw = self.native_response(payload)
        raw["actual_batch_widths"] = [1, 2, 2, 1]
        for response in raw["requests"]:
            response["output_text"] = "42"
        adapter.validate_short(raw, payload)
        self.assertTrue(raw["facts_pass"])

    def test_short_math_fact_is_required_and_recorded(self):
        payload = self.payload()
        response = self.native_response(payload)
        result = SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")
        codec = mock.Mock()
        codec.decode.return_value = "41"
        with (
            mock.patch.object(adapter.subprocess, "run", return_value=result),
            mock.patch.object(adapter, "tokenizer", return_value=codec),
        ):
            with self.assertRaisesRegex(matrix.MatrixError, "fact42"):
                adapter.run_batch(
                    payload, self.root, self.variant_root, self.standard, self.off
                )
        codec.decode.return_value = "42<|im_end|>"
        with (
            mock.patch.object(adapter.subprocess, "run", return_value=result),
            mock.patch.object(adapter, "tokenizer", return_value=codec),
        ):
            observed = adapter.run_batch(
                payload, self.root, self.variant_root, self.standard, self.off
            )
        self.assertTrue(observed["facts_pass"])
        self.assertTrue(all(r["facts_pass"] for r in observed["requests"]))

    def test_native_failure_is_not_marked_complete(self):
        payload = self.payload()
        result = SimpleNamespace(
            returncode=7, stdout="", stderr="original native failure"
        )
        with mock.patch.object(adapter.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(
                matrix.MatrixError, "exit7.*original native failure"
            ):
                adapter.run_batch(
                    payload, self.root, self.variant_root, self.standard, self.off
                )

    def test_short_fixture_counts_and_distinct_prefixes_without_inference(self):
        codec = mock.Mock()
        codec.encode.return_value = [220]
        codec.apply_chat_template.side_effect = [
            [248045, i, 248046] + [7] * 36 for i in range(4)
        ]
        with mock.patch.object(adapter, "tokenizer", return_value=codec):
            payload = adapter.short_payload(
                self.standard, "8", {"test_identity": "mocked"}
            )
        self.assertEqual(payload["mode"], "correctness")
        self.assertTrue(
            all(
                call.kwargs["return_dict"] is False
                for call in codec.apply_chat_template.call_args_list
            )
        )
        self.assertEqual(payload["job"]["parameters"]["input_tokens"], 64)
        self.assertEqual(payload["job"]["parameters"]["output_tokens"], 32)
        self.assertEqual(
            len({tuple(p["token_ids"][:32]) for p in payload["prompts"]}), 4
        )
        self.assertTrue(all(len(p["token_ids"]) == 64 for p in payload["prompts"]))

    def test_short_fixture_selected_concurrency_and_tail_split(self):
        for width in (1, 2, 4):
            codec = mock.Mock()
            codec.encode.return_value = [220]
            codec.apply_chat_template.side_effect = [
                [248045, i, 248046] + [7] * 36 for i in range(width)
            ]
            with mock.patch.object(adapter, "tokenizer", return_value=codec):
                payload = adapter.short_payload(self.standard, "2", {}, width, 16)
            parameters = payload["job"]["parameters"]
            self.assertEqual(parameters["concurrency"], width)
            self.assertEqual(parameters["tail_prefill_splits"], 16)
            self.assertEqual(len(payload["prompts"]), width)
            self.assertEqual(payload["job"]["group_id"], matrix.digest(parameters))

    def test_short_cli_preserves_raw_failed_business_result(self):
        error = matrix.MatrixError("short arithmetic fact42 missing")
        error.raw_response = {
            "status": "complete",
            "requests": [{"output_text": "wrong", "output_token_ids": [9]}],
        }
        output = self.root / "failed-short.json"
        with (
            mock.patch.object(adapter, "capabilities", return_value={"identity": {}}),
            mock.patch.object(adapter, "short_payload", return_value={}),
            mock.patch.object(adapter, "run_batch", side_effect=error),
        ):
            result = adapter.main(["--run-correctness", str(output)])
        saved = json.loads(output.read_text())
        self.assertEqual(result, 2)
        self.assertEqual(saved["status"], "incorrect")
        self.assertEqual(saved["requests"][0]["output_token_ids"], [9])
        self.assertIn("fact42", saved["error"])
