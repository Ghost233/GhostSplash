import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dev.benchmarks import d3_models


class D3BenchmarkIsolationTests(unittest.TestCase):
    def test_stale_frontend_report_cannot_start_performance_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "splash"
            binary.write_bytes(b"engine")
            report = root / "correctness.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "phase": "correctness",
                        "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
                        "frontend_sha256": "stale",
                        "models": {"lite": {}},
                        "validated_lengths": [4096],
                        "reference_mode": "llama",
                    }
                )
            )
            args = SimpleNamespace(
                output=root / "performance",
                binary=binary,
                phase="performance",
                correctness_report=report,
                models=["lite"],
                lengths=[4096],
                reference_mode="llama",
            )
            with (
                patch.object(d3_models, "frontend_hash", return_value="current"),
                patch.object(d3_models, "Splash") as startup,
            ):
                with self.assertRaisesRegex(
                    ValueError, "correctness report did not pass"
                ):
                    d3_models.run(args)
                startup.assert_not_called()

    def test_llama_reference_preserves_recorded_lmstudio_settings(self):
        args = SimpleNamespace(reference_binary=Path("/reference/llama-server"))
        for name, suffix in (("lite", "BF16"), ("mini", "Q4_K_M")):
            with self.subTest(model=name):
                command = d3_models.llama_command(args, name)
                for flag, value in {
                    "--ctx-size": "262144",
                    "--parallel": "4",
                    "--batch-size": "32768",
                    "--ubatch-size": "32768",
                    "--cache-type-k": "q4_0",
                    "--cache-type-v": "q4_0",
                    "--flash-attn": "on",
                    "--n-gpu-layers": "999999",
                    "--threads": "13",
                    "--host": "127.0.0.1",
                    "--port": "18144",
                }.items():
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertTrue(
                    command[command.index("--model") + 1].endswith(
                        f"vllm-d3-{name}-{suffix}.gguf"
                    )
                )
                self.assertTrue(
                    command[command.index("--mmproj") + 1].endswith(
                        f"mmproj-vllm-d3-{name}-BF16.gguf"
                    )
                )
                self.assertNotIn("--api-key", command)

    def test_desktop_load_still_waits_for_external_inference(self):
        busy = {"gpu_utilization_percent": 90, "busy_pids": [123]}
        idle = {"gpu_utilization_percent": 90, "busy_pids": []}
        samples = []
        with (
            patch.object(
                d3_models, "environment", side_effect=[busy, idle, idle, idle]
            ),
            patch.object(d3_models.time, "sleep"),
        ):
            d3_models.wait_for_device([456], samples, True)
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["busy_pids"], [123])
        self.assertTrue(all(not row["busy_pids"] for row in samples[1:]))

    def test_default_keeps_existing_gpu_idle_policy(self):
        samples = []
        with patch.object(d3_models, "wait_idle") as wait:
            d3_models.wait_for_device([456], samples, False)
        wait.assert_called_once_with([456], samples)
