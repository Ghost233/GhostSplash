import contextlib
import io
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
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

    def test_wall_timeout_stops_a_silent_child_and_cannot_be_reused(self):
        result, output = self.run_check(
            "import time; print('waiting', flush=True); time.sleep(30)",
            timeout=0.3,
            heartbeat=0.05,
        )
        self.assertLess(result["exit_code"], 0)
        self.assertFalse(result["passed"])
        self.assertIn("TimeoutExpired", result["tool_error"])
        self.assertEqual(result["timeout_seconds"], 0.3)
        self.assertIn("waiting", Path(result["log"]).read_text())
        self.assertIn("HEARTBEAT", output)
        self.assertFalse(list(self.directory.glob("*.passed.json")))
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)

    def test_timeout_stops_an_output_descendant_after_its_parent_exits(self):
        other = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        descendant = None
        try:
            result, _ = self.run_check(
                "import os, signal, time\n"
                "if os.fork() == 0:\n"
                " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                " print('descendant', os.getpid(), flush=True)\n"
                " for _ in range(600):\n"
                "  print('still active', flush=True); time.sleep(.05)\n"
                "else:\n"
                " raise SystemExit(0)\n",
                timeout=0.4,
            )
            self.assertEqual(result["exit_code"], 0)
            self.assertFalse(result["passed"])
            self.assertIn("TimeoutExpired", result["tool_error"])
            log = Path(result["log"]).read_text()
            descendant = int(log.split("descendant ", 1)[1].splitlines()[0])
            until = time.monotonic() + 2
            while True:
                try:
                    os.kill(descendant, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= until:
                    self.fail("the timed-out descendant survived group cleanup")
                time.sleep(0.01)
            self.assertIsNone(other.poll())
            self.assertFalse(list(self.directory.glob("*.passed.json")))
        finally:
            if descendant is not None:
                try:
                    os.kill(descendant, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            other.terminate()
            other.wait(timeout=5)

    def test_timeout_is_part_of_the_reusable_result_identity(self):
        first, _ = self.run_check("print('ok')")
        bounded, output = self.run_check("print('ok')", timeout=2, reuse=True)
        self.assertTrue(bounded["passed"])
        self.assertNotEqual(first["fingerprint"], bounded["fingerprint"])
        self.assertNotIn("REUSE", output)
        self.assertEqual(bounded["identity"]["timeout_seconds"], 2)
        reused, output = self.run_check("print('ok')", timeout=2, reuse=True)
        self.assertEqual(bounded["result_file"], reused["result_file"])
        self.assertIn("REUSE", output)

    def test_timeout_stops_a_descendant_that_closed_its_output(self):
        pid_file = self.root / "descendant.pid"
        other = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        descendant = None
        try:
            result, _ = self.run_check(
                "import os, signal, sys, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "if os.fork() == 0:\n"
                " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                " os.close(1); os.close(2)\n"
                f" Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
                " time.sleep(30); os._exit(0)\n"
                "else:\n"
                f" while not Path({str(pid_file)!r}).exists(): time.sleep(.01)\n"
                " print('descendant ready', flush=True)\n"
                " time.sleep(30)\n",
                timeout=0.4,
            )
            descendant = int(pid_file.read_text())
            self.assertEqual(result["exit_code"], 0)
            self.assertFalse(result["passed"])
            self.assertIn("TimeoutExpired", result["tool_error"])
            self.assertIn("descendant ready", Path(result["log"]).read_text())
            until = time.monotonic() + 2
            while True:
                try:
                    os.kill(descendant, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= until:
                    self.fail("the detached-output descendant survived group cleanup")
                time.sleep(0.01)
            self.assertIsNone(other.poll())
            self.assertFalse(list(self.directory.glob("*.passed.json")))
        finally:
            if descendant is not None:
                try:
                    os.kill(descendant, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            other.terminate()
            other.wait(timeout=5)

    def test_nonpositive_or_nonfinite_timeout_is_rejected_before_execution(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with (
                self.subTest(timeout=timeout),
                self.assertRaisesRegex(ValueError, "finite positive"),
            ):
                self.run_check("raise SystemExit(0)", timeout=timeout)
        self.assertFalse(self.directory.exists())

    def test_cli_timeout_returns_failure_and_retains_the_real_child_exit(self):
        bootstrap = (
            "from dev.tools import run_native_test as runner; "
            "runner.machine_identity = lambda: {'complete': True, 'gpu': 'test'}; "
            "raise SystemExit(runner.main())"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                bootstrap,
                "--directory",
                str(self.directory),
                "--timeout",
                "0.3",
                "--",
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(30)",
            ],
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 128 + signal.SIGTERM, result.stderr)
        records = list(self.directory.glob("*.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(record["exit_code"], -signal.SIGTERM)
        self.assertFalse(record["passed"])
        self.assertIn("TimeoutExpired", record["tool_error"])

    def test_cli_rejects_invalid_timeout_values(self):
        for value in ("0", "-1", "inf", "nan"):
            with self.subTest(value=value):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(runner.__file__)),
                        "--timeout",
                        value,
                        "--",
                        sys.executable,
                        "-c",
                        "raise SystemExit(0)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--timeout requires a finite positive", result.stderr)


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
