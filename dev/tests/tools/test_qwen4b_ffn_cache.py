"""Offline checks for the isolated FFN weight-cache experiment."""

import argparse
import json
import os
import plistlib
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dev.benchmarks import qwen4b_ffn_cache as experiment
from dev.benchmarks.qwen4b_matrix import MatrixError


class FfnCacheTest(unittest.TestCase):
    def test_reports_first_token_divergence(self):
        self.assertIsNone(experiment.first_divergence([1, 2], [1, 2]))
        self.assertEqual(
            experiment.first_divergence([1, 2, 3], [1, 9, 3]),
            {"index": 1, "full_fp16": 2, "quantized": 9},
        )
        self.assertEqual(experiment.first_divergence([1], [1, 2])["index"], 1)

    def test_memory_ledger_is_not_inferred_from_a_constant(self):
        report = {
            "target_source_weights_bytes": 100,
            "ffn_fp16_cache_bytes": 40,
            "draft_weights_bytes": 10,
            "vision_weights_bytes": 0,
            "state_bytes": 20,
            "prefill_workspace_bytes": 30,
            "decode_workspace_bytes": 5,
            "kv_bytes": 25,
            "state_staging_bytes": 0,
            "ane_bytes": 10,
            "unclassified_bytes": 2,
            "backend_current_bytes": 242,
            "backend_peak_bytes": 260,
            "ffn_cached_matrices": 96,
        }
        experiment.validate_memory(report, "full_fp16")
        with self.assertRaises(MatrixError):
            experiment.validate_memory(report, "quantized")
        report["ffn_fp16_cache_bytes"] = 0
        report["ffn_cached_matrices"] = 0
        report["backend_current_bytes"] = 202
        experiment.validate_memory(report, "quantized")
        report["backend_current_bytes"] = 201
        with self.assertRaises(MatrixError):
            experiment.validate_memory(report, "quantized")

    def test_busy_service_prevents_inference(self):
        with patch.object(experiment, "environment", return_value={"busy_pids": [99]}):
            with self.assertRaisesRegex(MatrixError, "external inference"):
                experiment.require_idle()

    def test_low_cpu_benchmark_is_busy_but_owned_child_is_excluded(self):
        gpu = plistlib.dumps([{"PerformanceStatistics": {"Device Utilization %": 10}}])
        processes = "99101 0.1 /tmp/qwen4b-matrix-native\n"
        with patch.object(
            experiment.subprocess, "check_output", side_effect=[gpu, processes]
        ):
            snapshot = experiment.environment()
        self.assertEqual(snapshot["busy_pids"], [99101])
        with patch.object(
            experiment.subprocess, "check_output", side_effect=[gpu, processes]
        ):
            self.assertEqual(experiment.environment((99101,))["busy_pids"], [])

    def test_idle_llama_service_remains_allowed(self):
        import io

        gpu = plistlib.dumps([{"PerformanceStatistics": {"Device Utilization %": 10}}])
        with (
            patch.object(
                experiment.subprocess,
                "check_output",
                side_effect=[gpu, "99102 0.1 /tmp/llama-server\n"],
            ),
            patch.object(experiment.subprocess, "run") as ps,
            patch.object(
                experiment.urllib.request,
                "urlopen",
                return_value=io.BytesIO(b'[{"is_processing": false}]'),
            ),
        ):
            ps.return_value.stdout = "llama-server --port 9000"
            snapshot = experiment.environment()
        self.assertEqual(snapshot["busy_pids"], [])
        self.assertIs(snapshot["servers"][0]["processing"], False)

    def test_correctness_allows_display_load_but_performance_still_rejects_it(self):
        with (
            patch.object(
                experiment,
                "environment",
                return_value={"busy_pids": [], "gpu_utilization_percent": 97},
            ),
            patch.object(experiment.time, "sleep"),
        ):
            self.assertEqual(
                len(experiment.require_idle(ignore_gpu_utilization=True)), 2
            )
            with self.assertRaisesRegex(MatrixError, "GPU not idle"):
                experiment.require_idle()

    def test_correctness_display_permission_still_blocks_external_benchmark(self):
        with patch.object(
            experiment,
            "environment",
            return_value={"busy_pids": [43], "gpu_utilization_percent": 97},
        ):
            with self.assertRaisesRegex(MatrixError, "external inference active"):
                experiment.require_idle(ignore_gpu_utilization=True)

    def test_only_authorized_llama_pid_can_run_alongside_correctness(self):
        gpu = plistlib.dumps([{"PerformanceStatistics": {"Device Utilization %": 97}}])
        with (
            patch.object(
                experiment.subprocess,
                "check_output",
                side_effect=[
                    gpu,
                    "12794 32 /tmp/llama-server\n43 0.1 /tmp/backend-benchmark\n",
                ],
            ),
            patch.object(experiment.subprocess, "run") as ps,
        ):
            ps.return_value.stdout = "llama-server"
            snapshot = experiment.environment(allow_inference_pid=12794)
        self.assertEqual(snapshot["busy_pids"], [43])
        self.assertTrue(snapshot["servers"][0]["busy"])
        self.assertTrue(snapshot["servers"][0]["authorized_concurrent_inference"])
        with patch.object(
            experiment.subprocess,
            "check_output",
            side_effect=[gpu, "12794 0.1 /tmp/backend-benchmark\n"],
        ):
            self.assertEqual(
                experiment.environment(allow_inference_pid=12794)["busy_pids"], [12794]
            )

    def test_authorized_inference_services_survive_pid_changes(self):
        gpu = plistlib.dumps([{"PerformanceStatistics": {"Device Utilization %": 97}}])
        with (
            patch.object(
                experiment.subprocess,
                "check_output",
                side_effect=[
                    gpu,
                    "101 32 /tmp/llama-server\n102 12 /tmp/llama-server\n43 0.1 /tmp/backend-benchmark\n",
                ],
            ),
            patch.object(experiment.subprocess, "run") as ps,
        ):
            ps.return_value.stdout = "llama-server"
            snapshot = experiment.environment(allow_concurrent_inference=True)
        self.assertEqual(snapshot["busy_pids"], [43])
        self.assertTrue(
            all(r["authorized_concurrent_inference"] for r in snapshot["servers"][:2])
        )
        self.assertFalse(snapshot["servers"][2]["authorized_concurrent_inference"])

    def test_all_services_permission_is_rejected_for_performance(self):
        with patch.object(
            sys, "argv", ["ffn", "--run-performance", "--allow-concurrent-inference"]
        ):
            with self.assertRaises(SystemExit) as error:
                experiment.main()
        self.assertEqual(error.exception.code, 2)

    def test_parallel_inference_permission_is_rejected_for_performance(self):
        with (
            patch.object(
                sys,
                "argv",
                ["ffn", "--run-performance", "--allow-inference-pid", "12794"],
            ),
            patch.object(experiment, "performance") as performance,
        ):
            with self.assertRaises(SystemExit) as error:
                experiment.main()
        self.assertEqual(error.exception.code, 2)
        performance.assert_not_called()

    def test_high_gpu_prevents_inference(self):
        with patch.object(
            experiment,
            "environment",
            return_value={"busy_pids": [], "gpu_utilization_percent": 95},
        ):
            with self.assertRaisesRegex(MatrixError, "GPU not idle"):
                experiment.require_idle()

    def test_candidate_profiles_change_only_the_intended_ffn_policy(self):
        for name in experiment.PROFILES:
            configs = experiment.configurations(name)
            a, b = configs["A"], configs["B"]
            changed = {key for key in a if a[key] != b[key]}
            expected = (
                {"ffn_weight_mode"}
                if name == "cache"
                else {"ffn_weight_mode", "ffn_cache_budget_bytes"}
                if name == "limited_cache"
                else {"ffn_prefill_variant"}
            )
            self.assertEqual(changed, expected, name)
            self.assertEqual(len(experiment.jobs(name)), 24)
        self.assertEqual(
            experiment.configurations("fused_gate_up")["A"]["ffn_prefill_variant"],
            "m64n32",
        )

    def test_plan_is_abba_with_only_cache_mode_changing(self):
        jobs = experiment.jobs()
        self.assertEqual(len(jobs), 24)
        for at in range(0, len(jobs), 4):
            pair = jobs[at : at + 4]
            self.assertEqual(
                [j["parameters"]["ffn_weight_mode"] for j in pair],
                ["full_fp16", "quantized", "quantized", "full_fp16"],
            )
            fixed = [
                {k: v for k, v in j["parameters"].items() if k != "ffn_weight_mode"}
                for j in pair
            ]
            self.assertTrue(all(p == fixed[0] for p in fixed))
            self.assertEqual(fixed[0]["input_tokens"], 32768)
            self.assertIn(fixed[0]["output_tokens"], (512, 2048))
            self.assertEqual(fixed[0]["concurrency"], 1)


class LongCorrectnessTest(unittest.TestCase):
    def run_long(self, divergent=False):
        identity = {
            "code_sha256": "a" * 64,
            "weight_sha256": {"target": "b" * 64, "draft": "c" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write(
                root / "correctness.json",
                {"status": "pass", "identity": identity, "profile": "q4k_group_scale"},
            )
            experiment.write(root / "inputs.json", {"prompts": [{"id": "long-p0"}]})
            calls = []

            def fake_native(
                args, document, job, prompts, mode, label, *, long_correctness=False
            ):
                self.assertEqual(mode, "correctness")
                self.assertTrue(long_correctness)
                self.assertEqual(job["parameters"]["input_tokens"], 32768)
                self.assertEqual(prompts, [{"id": "long-p0"}])
                tokens = [42] * job["parameters"]["output_tokens"]
                if divergent and len(calls) == 1:
                    tokens[400] = 43
                calls.append(label)
                return {
                    "requests": [{"output_token_ids": tokens}],
                    "attempt_directory": label,
                }, {"status": "complete", "facts_pass": True}

            args = argparse.Namespace(
                output=root, profile="q4k_group_scale", cache_budget_bytes=1 << 30
            )
            with (
                patch.object(
                    experiment, "prepared", return_value={"identity": identity}
                ),
                patch.object(experiment, "native", side_effect=fake_native),
            ):
                if divergent:
                    with self.assertRaisesRegex(MatrixError, "divergence"):
                        experiment.long_correctness(args)
                else:
                    experiment.long_correctness(args)
            gate = json.loads((root / "long-correctness.json").read_text())
            self.assertEqual(len(calls), 2 if divergent else 4)
            self.assertEqual(
                gate["status"], "needs_divergence_analysis" if divergent else "pass"
            )
            if divergent:
                self.assertEqual(
                    gate["comparisons"][0]["first_divergence"]["index"], 400
                )
            else:
                self.assertEqual(
                    [r["output_tokens"] for r in gate["comparisons"]], [512, 2048]
                )

    def test_long_correctness_uses_both_forced_lengths_without_performance(self):
        self.run_long()

    def test_long_divergence_after_token_32_is_recorded_and_stops_next_length(self):
        self.run_long(divergent=True)


class PerformanceGateTest(unittest.TestCase):
    def test_long_cross_mode_divergence_stops_after_first_pair(self):
        self.run_divergence(1, 2, "cross_mode")

    def test_same_mode_repeat_divergence_stops_at_second_repeat(self):
        self.run_divergence(2, 3, "same_mode_repeat")

    def run_divergence(self, divergent_call, expected_calls, kind):
        identity = {
            "code_sha256": "a" * 64,
            "weight_sha256": {"target": "b" * 64, "draft": "c" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write(
                root / "correctness.json",
                {"status": "pass", "identity": identity, "profile": "cache"},
            )
            experiment.write(
                root / "inputs.json", {"prompts": [{"id": f"p-{i}"} for i in range(3)]}
            )
            calls = []

            def fake_native(args, document, job, prompts, mode, label):
                tokens = [42] * job["parameters"]["output_tokens"]
                if len(calls) == divergent_call:
                    tokens[33] = 43
                calls.append(label)
                raw = {"requests": [{"output_token_ids": tokens}], "memory_report": {}}
                checked = {
                    "requests": [
                        {
                            "prefill_tok_s": 2000,
                            "decode_tok_s": 100,
                            "timings": {
                                "prefill_wall_seconds": 16,
                                "decode_wall_seconds": 5,
                                "ttft_seconds": 16,
                                "total_wall_seconds": 21,
                            },
                        }
                    ],
                    "memory_peak_bytes": 100,
                }
                return raw, checked

            args = argparse.Namespace(
                output=root, profile="cache", cache_budget_bytes=1 << 30, resume=False
            )
            with (
                patch.object(
                    experiment, "prepared", return_value={"identity": identity}
                ),
                patch.object(experiment, "native", side_effect=fake_native),
            ):
                with self.assertRaisesRegex(MatrixError, "divergence"):
                    experiment.performance(args)
            self.assertEqual(len(calls), expected_calls)
            report = json.loads((root / "summary.json").read_text())
            self.assertEqual(report["status"], "needs_divergence_analysis")
            self.assertEqual(report["divergence"]["kind"], kind)
            self.assertEqual(report["divergence"]["index"], 33)


class ResumeGroupTest(unittest.TestCase):
    def test_incomplete_abba_is_restarted_from_A_and_archived(self):
        identity = {
            "code_sha256": "a" * 64,
            "weight_sha256": {"target": "b" * 64, "draft": "c" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write(
                root / "correctness.json",
                {"status": "pass", "identity": identity, "profile": "cache"},
            )
            experiment.write(
                root / "inputs.json", {"prompts": [{"id": str(i)} for i in range(3)]}
            )
            args = argparse.Namespace(
                output=root, profile="cache", cache_budget_bytes=1 << 30, resume=False
            )
            positions = []

            def fake_native(args, document, job, prompts, mode, label):
                positions.append(job["position"])
                if len(positions) == 2:
                    raise MatrixError("interference")
                tokens = [42] * job["parameters"]["output_tokens"]
                return {
                    "requests": [{"output_token_ids": tokens}],
                    "memory_report": {},
                }, {"requests": [], "memory_peak_bytes": 1}

            with (
                patch.object(
                    experiment, "prepared", return_value={"identity": identity}
                ),
                patch.object(experiment, "native", side_effect=fake_native),
            ):
                with self.assertRaisesRegex(MatrixError, "interference"):
                    experiment.performance(args)
            checkpoint = root / "groups/out-512-prompt-0.json"
            self.assertEqual(json.loads(checkpoint.read_text())["status"], "incomplete")
            args.resume = True
            positions.clear()

            def retry(args, document, job, prompts, mode, label):
                positions.append(job["position"])
                if len(positions) == 5:
                    raise MatrixError("next group paused")
                tokens = [42] * job["parameters"]["output_tokens"]
                return {
                    "requests": [{"output_token_ids": tokens}],
                    "memory_report": {},
                }, {"requests": [], "memory_peak_bytes": 1}

            with (
                patch.object(
                    experiment, "prepared", return_value={"identity": identity}
                ),
                patch.object(experiment, "native", side_effect=retry),
            ):
                with self.assertRaisesRegex(MatrixError, "next group"):
                    experiment.performance(args)
            self.assertEqual(positions[:4], [0, 1, 2, 3])
            self.assertEqual(json.loads(checkpoint.read_text())["status"], "complete")
            self.assertEqual(
                len(list((root / "groups/history").glob("out-512-prompt-0-*.json"))), 1
            )


class AtomicAndOperatorRetryTest(unittest.TestCase):
    def test_interrupted_replace_keeps_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "group.json"
            experiment.write(checkpoint, {"status": "complete", "records": [1]})
            with patch.object(
                experiment.os, "replace", side_effect=InterruptedError("interrupted")
            ):
                with self.assertRaises(InterruptedError):
                    experiment.write(
                        checkpoint, {"status": "running", "records": [2, 3]}
                    )
            self.assertEqual(
                json.loads(checkpoint.read_text()),
                {"status": "complete", "records": [1]},
            )
            self.assertEqual(list(Path(directory).glob("*.tmp-*")), [])

    def test_failed_operator_gate_can_retry_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"code_sha256": "a" * 64}
            args = argparse.Namespace(
                output=root,
                model_root=root,
                profile="cache",
                cache_budget_bytes=1 << 30,
            )
            document = {
                "identity": identity,
                "artifacts": {
                    "numerical_check": {"path": "/fake/tool"},
                    "metallib": {"path": "/fake/library"},
                },
            }
            experiment.write(
                root / "correctness.json",
                {"status": "pass", "identity": identity, "profile": "cache"},
            )
            failure = {
                "performance_usable": False,
                "exit_code": 1,
                "status": "invalid",
                "error": "operator failure",
            }
            with (
                patch.object(experiment, "prepared", return_value=document),
                patch.object(
                    experiment, "checked_process", return_value=failure.copy()
                ) as run,
            ):
                for _ in range(2):
                    with self.assertRaisesRegex(MatrixError, "operator check failed"):
                        experiment.correctness(args)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                json.loads((root / "correctness.json").read_text())["status"], "failed"
            )
            attempts = list((root / "attempts").glob("operators-*"))
            self.assertEqual(len(attempts), 2)
            self.assertTrue(all((p / "evidence.json").is_file() for p in attempts))
            self.assertEqual(
                len((root / "operators-attempts.jsonl").read_text().splitlines()), 2
            )


class ProcessEvidenceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        lock = patch.object(
            experiment, "DEVICE_LOCK", Path(directory.name) / "device.lock"
        )
        lock.start()
        self.addCleanup(lock.stop)

    def test_successful_correctness_is_not_a_performance_sample(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(experiment, "require_idle", return_value=[]),
            patch.object(
                experiment, "environment", return_value={"busy_pids": [], "servers": []}
            ),
        ):
            root = Path(directory)
            result = experiment.checked_process(
                [sys.executable, "-c", "print(1)"],
                None,
                root / "stdout",
                root / "stderr",
                poll_seconds=0.01,
                correctness=True,
                allow_concurrent_inference=True,
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["performance_usable"])
        self.assertTrue(result["authorized_all_inference_services"])

    def test_device_lock_blocks_launch_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(experiment, "DEVICE_LOCK", root / "device.lock"):
                with experiment.device_lock():
                    with patch.object(experiment.subprocess, "Popen") as launch:
                        evidence = experiment.checked_process(
                            ["never-start"], None, root / "stdout", root / "stderr"
                        )
                    launch.assert_not_called()
                    self.assertEqual(evidence["status"], "failed")
                    self.assertIn("device lock", evidence["error"])
                    self.assertIsNone(evidence["exit_code"])
                with self.assertRaisesRegex(ValueError, "simulated failure"):
                    with experiment.device_lock():
                        raise ValueError("simulated failure")
                with experiment.device_lock():
                    pass

    def test_sigterm_cancels_only_owned_child_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory) / "stdout", Path(directory) / "stderr"

            def terminate(exclude=()):
                os.kill(os.getpid(), signal.SIGTERM)

            before = signal.getsignal(signal.SIGTERM)
            with (
                patch.object(experiment, "require_idle", return_value=[]),
                patch.object(experiment, "environment", side_effect=terminate),
            ):
                evidence = experiment.checked_process(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    None,
                    out,
                    err,
                    poll_seconds=0.01,
                )
            self.assertEqual(evidence["status"], "cancelled")
            self.assertFalse(evidence["performance_usable"])
            self.assertIsNotNone(evidence["exit_code"])
            self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_nonzero_exit_is_an_invalid_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory) / "stdout", Path(directory) / "stderr"
            with (
                patch.object(experiment, "require_idle", return_value=[]),
                patch.object(experiment, "environment", return_value={"busy_pids": []}),
            ):
                evidence = experiment.checked_process(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    None,
                    out,
                    err,
                    poll_seconds=0.01,
                )
            self.assertEqual(evidence["exit_code"], 3)
            self.assertEqual(evidence["status"], "invalid")
            self.assertFalse(evidence["performance_usable"])

    def test_timeout_returns_failed_evidence_and_keeps_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory) / "stdout", Path(directory) / "stderr"
            with (
                patch.object(experiment, "require_idle", return_value=[]),
                patch.object(experiment, "environment", return_value={"busy_pids": []}),
            ):
                evidence = experiment.checked_process(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('started', flush=True); time.sleep(10)",
                    ],
                    None,
                    out,
                    err,
                    timeout_seconds=0.1,
                    poll_seconds=0.01,
                )
            self.assertEqual(evidence["status"], "timed_out")
            self.assertFalse(evidence["performance_usable"])
            self.assertIsNotNone(evidence["exit_code"])
            self.assertIn("started", out.read_text())


if __name__ == "__main__":
    unittest.main()
