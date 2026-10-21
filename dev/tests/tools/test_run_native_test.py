import contextlib
import io
import json
import os
import selectors
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dev.tools import run_native_test as runner


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "results"
        machine_patch = patch.object(
            runner, "machine_identity", return_value={"complete": True, "gpu": "test"}
        )
        self.machine = machine_patch.start()
        self.addCleanup(machine_patch.stop)

    def run_check(self, code, **options):
        output = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True)
        with contextlib.redirect_stdout(output):
            result = runner.run(
                [sys.executable, "-c", code],
                directory=self.directory,
                root=self.root,
                **options,
            )
        text = output.buffer.getvalue().decode()
        self.assertEqual(
            json.loads(Path(result["result_file"]).read_text())["exit_code"],
            result["exit_code"],
        )
        return result, text

    def test_live_output_heartbeat_and_real_failure_are_preserved(self):
        result, output = self.run_check(
            "import time; print('before', flush=True); time.sleep(.15); raise SystemExit(7)",
            heartbeat=0.03,
        )
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["passed"])
        self.assertIn("before", Path(result["log"]).read_text())
        self.assertIn("HEARTBEAT", output)
        self.assertLess(output.index("before"), output.index("END exit=7"))
        self.assertFalse(list(self.directory.glob("*.passed.json")))

    def test_reuse_requires_complete_identity_and_untampered_log(self):
        first, _ = self.run_check("print('checked')")
        reused, output = self.run_check("print('checked')", reuse=True)
        self.assertIn("REUSE", output)
        self.assertEqual(first["result_file"], reused["result_file"])
        Path(first["log"]).write_text("tampered")
        fresh, output = self.run_check("print('checked')", reuse=True)
        self.assertNotIn("REUSE", output)
        self.assertNotEqual(first["result_file"], fresh["result_file"])

    def test_data_validation_and_gpu_changes_refuse_reuse(self):
        data = self.root / "fixture.bin"
        data.write_bytes(b"one")
        first, _ = self.run_check("print('ok')", extra=[data])
        data.write_bytes(b"two")
        changed, _ = self.run_check("print('ok')", extra=[data], reuse=True)
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
        with patch.dict(os.environ, {"MTL_SHADER_VALIDATION": "1"}):
            validated, _ = self.run_check("print('ok')", extra=[data], reuse=True)
        self.assertNotEqual(changed["fingerprint"], validated["fingerprint"])
        self.machine.return_value = {"complete": True, "gpu": "different"}
        gpu, _ = self.run_check("print('ok')", extra=[data], reuse=True)
        self.assertNotEqual(changed["fingerprint"], gpu["fingerprint"])

    def test_build_parameter_changes_refuse_reuse(self):
        flags = self.root / "Makefile"
        flags.write_text("FLAGS=first\n")
        first, _ = self.run_check("print('ok')")
        flags.write_text("FLAGS=second\n")
        changed, output = self.run_check("print('ok')", reuse=True)
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
        self.assertNotIn("REUSE", output)

    def test_changed_input_during_execution_is_not_a_pass_or_reusable(self):
        data = self.root / "fixture.bin"
        data.write_text("before")
        result, _ = self.run_check(
            f"from pathlib import Path; Path({str(data)!r}).write_text('after')",
            extra=[data],
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["inputs_unchanged"])
        self.assertFalse(result["passed"])
        self.assertFalse(list(self.directory.glob("*.passed.json")))

    def test_unknown_hardware_and_corrupt_cache_do_not_skip_execution(self):
        first, _ = self.run_check("print('ok')")
        (self.directory / (first["fingerprint"] + ".passed.json")).write_text("broken")
        fresh, output = self.run_check("print('ok')", reuse=True)
        self.assertNotIn("REUSE", output)
        self.assertTrue(fresh["passed"])
        self.machine.return_value = {"complete": False, "gpu": {"unavailable": "test"}}
        _, output = self.run_check("print('ok')", reuse=True)
        self.assertNotIn("REUSE", output)

    def test_cancellation_cleans_only_the_child_group_and_records_its_exit(self):
        class InterruptedSelector(selectors.SelectSelector):
            def select(self, timeout=None):
                raise KeyboardInterrupt

        with patch.object(runner.selectors, "DefaultSelector", InterruptedSelector):
            result, _ = self.run_check("import time; time.sleep(10)")
        self.assertLess(result["exit_code"], 0)
        self.assertIn("KeyboardInterrupt", result["tool_error"])
        self.assertFalse(result["passed"])
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)


class MachineIdentityTests(unittest.TestCase):
    def test_gpu_json_order_and_probe_stderr_do_not_make_identity_unstable(self):
        count = 0

        def probe(command, **options):
            nonlocal count
            self.assertEqual(options["stderr"], runner.subprocess.PIPE)
            if command[0] == "system_profiler":
                count += 1
                gpu = {
                    "sppci_model": "M5",
                    "sppci_cores": 20,
                    "spdisplays_ndrvs": [{"serial": str(count)}],
                }
                if count % 2:
                    gpu = dict(reversed(list(gpu.items())))
                return json.dumps({"SPDisplaysDataType": [gpu]})
            return "fixed-version"

        with (
            patch.object(runner.platform, "platform", return_value="fixed-system"),
            patch.object(runner.platform, "machine", return_value="arm64"),
            patch.object(runner.subprocess, "check_output", side_effect=probe),
        ):
            first, second = runner.machine_identity(), runner.machine_identity()
        self.assertTrue(first["complete"])
        self.assertEqual(first, second)


class FixtureChildLogsTests(unittest.TestCase):
    def launch(self, exit_code):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "native"
            binary.write_text(
                f"#!{sys.executable}\nimport sys\nprint('original native details')\n"
                f"print('native done')\nprint('PIPELINE_CHILD_TRACE', file=sys.stderr)\nraise SystemExit({exit_code})\n"
            )
            binary.chmod(0o755)
            bootstrap = (
                "from dev.tests.engine import run_draft_preparation as d; "
                "d.draft_fixture = lambda *args, **kwargs: None; d.main()"
            )
            return subprocess.run(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    str(binary),
                    "unused.metallib",
                    "unused.json",
                ],
                cwd=runner.ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_success_forwards_child_pipeline_logs_and_keeps_stdout_contract(self):
        result = self.launch(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "native done")
        self.assertIn("PIPELINE_CHILD_TRACE", result.stderr)
        self.assertIn("FIXTURE_CHILD_END exit=0", result.stderr)

    def test_failure_keeps_native_exit_and_original_assertion_details(self):
        result = self.launch(7)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PIPELINE_CHILD_TRACE", result.stderr)
        self.assertIn("FIXTURE_CHILD_END exit=7", result.stderr)
        self.assertIn("original native details", result.stderr)


if __name__ == "__main__":
    unittest.main()
