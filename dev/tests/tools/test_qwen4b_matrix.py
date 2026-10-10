import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.benchmarks import qwen4b_matrix as matrix


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.config = matrix.read_json(matrix.MANIFEST)
        self.inputs = [
            {
                "id": f"prompt-{i}",
                "token_ids": [5, i] + [9] * 32766,
                "expected_facts": ["480万元", "林悦"],
            }
            for i in range(4)
        ]
        for prompt in self.inputs:
            prompt["input_sha256"] = matrix.digest(prompt["token_ids"])
        self.identity = {
            "code_sha256": "a" * 64,
            "weight_sha256": {"target": "b" * 64, "draft": "c" * 64},
        }
        self.group = next(matrix.groups(self.config))
        self.job = self.group | {"repeat": 0}

    def response(self, job=None, inputs=None):
        job = job or self.job
        p = job["parameters"]
        requests = []
        for prompt in (inputs or self.inputs)[: p["concurrency"]]:
            tokens = [7] * p["output_tokens"]
            tokens[3] = 99
            requests.append(
                {
                    "prompt_id": prompt["id"],
                    "input_sha256": prompt["input_sha256"],
                    "input_tokens": 32768,
                    "output_token_ids": tokens,
                    "output_text": "预算４８０ 万元，负责人林悦。",
                    "eos_positions": [3],
                    "cached_input_tokens": 0,
                    "memory_peak_bytes": 1024,
                    "timings": {
                        "prefill_wall_seconds": 1,
                        "decode_wall_seconds": 2,
                        "total_wall_seconds": 3,
                    },
                }
            )
        return {
            "status": "complete",
            "identity": self.identity,
            "parameters": p,
            "actual_drafter_block_size": p["drafter_block_size"],
            "compiled_block_size": p["drafter_block_size"] or 8,
            "actual_kv_format": p["kv_format"],
            "actual_context_tokens": p["context_tokens"],
            "device_name": "Mocked test device",
            "model_layout_sha256": "e" * 64,
            "actual_tail_prefill_splits": p["tail_prefill_splits"],
            "commands": [["mock-native", "--ignore-eos"]],
            "actual_batch_widths": [p["concurrency"]] * 2,
            "ane_actual": {
                "enabled": p["ane_requested"],
                "share": p["ane_share"],
                "minimum_rows": p["ane_minimum_rows"],
                "auto_calibration": False,
            },
            "requests": requests,
            "warmup": {
                "prefill_rows": 2048,
                "decode_width": p["concurrency"],
                "iterations": 1,
            },
            "memory_peak_bytes": 2048,
            "timings": {
                "prefill_wall_seconds": 1,
                "decode_wall_seconds": 2,
                "total_wall_seconds": 3,
            },
        }

    def capabilities(self):
        return {
            "identity": self.identity,
            "eos_token_ids": [99],
            "kv_formats": list(matrix.KV_FORMATS),
            "ane_modes": ["off", "on"],
            "drafter_states": list(matrix.DRAFTER_STATES),
            "supported_block_sizes": [2, 4, 8, 16],
            "output_tokens": [512, 2048],
            "tail_prefill_splits": list(range(1, 17)),
            "concurrency": [1, 2, 4],
            "ignore_eos": True,
            "cold_requests": True,
            "actual_batch_widths": True,
            "no_auto_calibration": True,
            "max_context_tokens": 36864,
            "max_memory_budget_bytes": 32 << 30,
        }

    def test_full_matrix_counts_and_unique_off_state(self):
        result = matrix.plan(self.config)
        self.assertEqual(
            (result["group_count"], result["batch_count"], result["request_count"]),
            (4800, 14400, 33600),
        )
        self.assertEqual(len({row["group_id"] for row in result["groups"]}), 4800)
        off = [
            row for row in result["groups"] if not row["parameters"]["drafter_enabled"]
        ]
        self.assertEqual(len(off), 960)
        self.assertTrue(
            all(row["parameters"]["drafter_block_size"] is None for row in off)
        )
        self.assertFalse(result["starts_inference"])
        self.assertEqual(result["inputs_status"], "unprepared")

    def test_manifest_pins_ane_and_forbids_calibration(self):
        on = next(
            g for g in matrix.groups(self.config) if g["parameters"]["ane_requested"]
        )
        self.assertEqual(
            (on["parameters"]["ane_share"], on["parameters"]["ane_minimum_rows"]),
            (0.5, 512),
        )
        for key, value in (
            ("auto_calibration", True),
            ("ignore_eos", False),
            ("context_tokens", 32768),
            ("memory_budget_bytes", 12 << 30),
            ("ane_on", {"share": 0.25, "minimum_rows": 512}),
        ):
            changed = self.config | {key: value}
            with self.subTest(key=key), self.assertRaises(matrix.MatrixError):
                matrix.validate_manifest(changed)

    def test_prompt_bos_can_match_but_first_page_cannot(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.json"
            path.write_text(json.dumps({"prompts": self.inputs}))
            self.assertEqual(len(matrix.load_inputs(path, self.config)), 4)
            changed = copy.deepcopy(self.inputs)
            changed[1]["token_ids"][:32] = changed[0]["token_ids"][:32]
            changed[1]["token_ids"][100] = 100
            changed[1].pop("input_sha256")
            path.write_text(json.dumps({"prompts": changed}))
            with self.assertRaisesRegex(matrix.MatrixError, "first 32-token pages"):
                matrix.load_inputs(path, self.config)

    def test_prompt_length_and_hash_are_checked(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.json"
            for change in (
                lambda p: p["token_ids"].pop(),
                lambda p: p.update(input_sha256="wrong"),
            ):
                inputs = copy.deepcopy(self.inputs)
                change(inputs[0])
                path.write_text(json.dumps({"prompts": inputs}))
                with self.assertRaises(matrix.MatrixError):
                    matrix.load_inputs(path, self.config)

    def test_forced_output_lengths_eos_and_aggregate_throughput(self):
        for size in (512, 2048):
            group = next(
                g
                for g in matrix.groups(self.config)
                if g["parameters"]["output_tokens"] == size
                and g["parameters"]["concurrency"] == 4
            )
            job = group | {"repeat": 0}
            result = matrix.validate_batch(
                self.response(job),
                job,
                self.inputs,
                self.identity,
                [99],
                performance=True,
            )
            self.assertEqual(result["aggregate"]["decode_tok_s"], 4 * size / 2)
            self.assertTrue(result["facts_pass"])
            for request in result["requests"]:
                self.assertEqual(
                    (
                        request["output_tokens"],
                        request["first_eos_position"],
                        request["eos_count"],
                    ),
                    (size, 3, 1),
                )
                self.assertEqual(request["eos_rate"], 1 / size)
            short = self.response(job)
            short["requests"][0]["output_token_ids"] = [7] * (size - 1)
            with self.assertRaisesRegex(matrix.MatrixError, "forced output count"):
                matrix.validate_batch(short, job, self.inputs, self.identity, [99])

    def test_performance_requires_fixed_warmup_but_correctness_has_no_marker(self):
        raw = self.response()
        raw.pop("warmup")
        with self.assertRaisesRegex(matrix.MatrixError, "fixed performance warmup"):
            matrix.validate_batch(
                raw, self.job, self.inputs[:1], self.identity, [99], performance=True
            )
        raw = self.response()
        raw["warmup"]["decode_width"] = 4
        with self.assertRaisesRegex(matrix.MatrixError, "fixed performance warmup"):
            matrix.validate_batch(
                raw, self.job, self.inputs[:1], self.identity, [99], performance=True
            )
        result = matrix.validate_batch(
            raw, self.job, self.inputs[:1], self.identity, [99]
        )
        self.assertNotIn("warmup", result)

    def test_early_eos_is_observed_without_truncation(self):
        raw = self.response()
        raw["requests"][0]["output_token_ids"] = [99] * 512
        raw["requests"][0]["eos_positions"] = list(range(512))
        result = matrix.validate_batch(
            raw, self.job, self.inputs[:1], self.identity, [99]
        )
        self.assertEqual(result["requests"][0]["first_eos_position"], 0)
        self.assertEqual(result["requests"][0]["eos_rate"], 1)
        raw["requests"][0]["output_token_ids"] = [7] * 512
        raw["requests"][0]["eos_positions"] = []
        result = matrix.validate_batch(
            raw, self.job, self.inputs[:1], self.identity, [99]
        )
        self.assertIsNone(result["requests"][0]["first_eos_position"])

    def test_wrong_facts_do_not_count_as_complete(self):
        raw = self.response()
        raw["requests"][0]["output_text"] = "1480万元，林悦"
        result = matrix.validate_batch(
            raw, self.job, self.inputs[:1], self.identity, [99]
        )
        self.assertEqual(result["status"], "incorrect")
        self.assertEqual(result["requests"][0]["missing_facts"], ["480万元"])

    def test_ane_fallback_is_unsupported_and_pins_are_verified(self):
        group = next(
            g for g in matrix.groups(self.config) if g["parameters"]["ane_requested"]
        )
        job = group | {"repeat": 0}
        raw = self.response(job)
        raw["ane_actual"] = {
            "enabled": False,
            "share": 0,
            "minimum_rows": 512,
            "auto_calibration": False,
            "fallback_reason": "unsupported weights",
        }
        result = matrix.validate_batch(raw, job, self.inputs[:1], self.identity, [99])
        self.assertEqual(result["status"], "unsupported")
        raw = self.response(job)
        raw["ane_actual"]["share"] = 0.25
        with self.assertRaisesRegex(matrix.MatrixError, "pins changed"):
            matrix.validate_batch(raw, job, self.inputs[:1], self.identity, [99])

    def test_cold_actual_width_identity_and_memory_must_match(self):
        group = next(
            g for g in matrix.groups(self.config) if g["parameters"]["concurrency"] == 2
        )
        job = group | {"repeat": 0}
        changes = (
            lambda r: r.update(actual_batch_widths=[1, 1]),
            lambda r: r["requests"][0].update(cached_input_tokens=32),
            lambda r: r.update(memory_peak_bytes=(32 << 30) + 1),
            lambda r: r.update(identity=self.identity | {"code_sha256": "d" * 64}),
            lambda r: r.update(actual_tail_prefill_splits=2),
            lambda r: r["ane_actual"].update(auto_calibration=True),
        )
        for change in changes:
            raw = self.response(job)
            change(raw)
            with self.subTest(change=change), self.assertRaises(matrix.MatrixError):
                matrix.validate_batch(raw, job, self.inputs[:2], self.identity, [99])

    def test_finished_lanes_can_reduce_width_after_full_concurrency(self):
        group = next(
            g for g in matrix.groups(self.config) if g["parameters"]["concurrency"] == 4
        )
        job = group | {"repeat": 0}
        raw = self.response(job)
        raw["actual_batch_widths"] = [4, 4, 3, 2, 1]
        result = matrix.validate_batch(raw, job, self.inputs, self.identity, [99])
        self.assertEqual(result["actual_batch_widths"], [4, 4, 3, 2, 1])
        raw["actual_batch_widths"] = [3, 2, 1]
        with self.assertRaisesRegex(matrix.MatrixError, "did_not_achieve_concurrency"):
            matrix.validate_batch(raw, job, self.inputs, self.identity, [99])

    def test_capabilities_do_not_fake_compiled_block_support(self):
        group = next(
            g
            for g in matrix.groups(self.config)
            if g["parameters"]["drafter_block_size"] == 16
        )
        capabilities = self.capabilities()
        capabilities["supported_block_sizes"] = [8]
        self.assertIn(
            "native variant does not support requested draft block size",
            matrix.unsupported(group["parameters"], capabilities),
        )
        self.assertEqual(
            matrix.unsupported(self.job["parameters"], self.capabilities()), []
        )

    def test_correctness_is_offline_and_has_no_performance_metrics(self):
        stored = self.job | {
            "identity": self.identity,
            "eos_token_ids": [99],
            "raw_response": self.response(),
        }
        document = matrix.validate_correctness([stored], self.config, self.inputs)
        result = document["results"][0]
        self.assertEqual(document["mode"], "correctness")
        self.assertFalse(document["matrix_complete"])
        self.assertNotIn("aggregate", result)
        self.assertNotIn("timings", result["requests"][0])
        self.assertNotIn("decode_tok_s", result["requests"][0])
        self.assertNotIn("performance_pass", json.dumps(document))
        with self.assertRaisesRegex(matrix.MatrixError, "duplicate"):
            matrix.validate_correctness([stored, stored], self.config, self.inputs)

    def test_correctness_cli_never_calls_backend(self):
        stored = self.job | {
            "identity": self.identity,
            "eos_token_ids": [99],
            "raw_response": self.response(),
        }
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(matrix.Backend, "call") as call,
        ):
            directory = Path(temporary)
            inputs, results, output = (
                directory / "inputs.json",
                directory / "results.jsonl",
                directory / "validated.json",
            )
            inputs.write_text(json.dumps({"prompts": self.inputs}))
            results.write_text(json.dumps(stored) + "\n")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    matrix.main(
                        [
                            "--validate-correctness",
                            "--inputs",
                            str(inputs),
                            "--results",
                            str(results),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            call.assert_not_called()
            self.assertEqual(json.loads(output.read_text())["mode"], "correctness")

    def test_default_cli_only_plans_without_calling_backend(self):
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(matrix.Backend, "call") as call,
        ):
            output = Path(temporary) / "plan.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(matrix.main(["--output", str(output)]), 0)
            call.assert_not_called()
            self.assertEqual(json.loads(output.read_text())["request_count"], 33600)

    def test_mocked_performance_interface_records_all_repeats_and_raw_data(self):
        backend = mock.Mock()
        backend.call.side_effect = lambda operation, payload: (
            self.capabilities()
            if operation == "capabilities"
            else self.response(payload["job"], payload["prompts"])
        )
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(matrix, "groups", return_value=[self.group]),
        ):
            path = Path(temporary) / "results.jsonl"
            counts = matrix.run_performance(self.config, self.inputs, backend, path)
            self.assertEqual(counts["complete"], 3)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["repeat"] for r in records], [0, 1, 2])
            self.assertTrue(
                all(r["eos_token_ids"] == [99] and "raw_response" in r for r in records)
            )
            self.assertEqual(backend.call.call_args_list[0].args[0], "capabilities")
            self.assertTrue(
                all(
                    call.args[1]["fresh_runtime"]
                    and call.args[1]["mode"] == "performance"
                    for call in backend.call.call_args_list[1:]
                )
            )

    def test_unsupported_combinations_do_not_call_inference(self):
        backend = mock.Mock()
        backend.call.return_value = self.capabilities() | {"kv_formats": []}
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(matrix, "groups", return_value=[self.group]),
        ):
            counts = matrix.run_performance(
                self.config, self.inputs, backend, Path(temporary) / "results.jsonl"
            )
            self.assertEqual(counts["unsupported"], 3)
            backend.call.assert_called_once_with("capabilities", {})
