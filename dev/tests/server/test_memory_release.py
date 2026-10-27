"""POST /v1/memory/release over HTTP: a request takes no fields, and the answer
says whether the engine gave its weights back."""

import argparse
import contextlib
import http.client
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import native_peer
from dev.tests.server_fixtures import RELEASED_WEIGHTS, FakeRuntime, HarnessTestCase
from server import runtime as engine_runtime
from server import server as api
from server import startup_report

REAL_MODEL_OPTIONS = None

# The weights section the engine answers a release request with when it keeps
# the weights, in each case native_engine_loop_test.cpp checks it reports, and
# the code and the words of the 409 that says why.
KEPT = {
    # A restore leaves the weights released until its last image is back, and
    # the Neural Engine split's program comes back a tick later.
    "restoring": (
        {**RELEASED_WEIGHTS, "restoring": True},
        "engine_busy",
        "restoring its weights",
    ),
    "restoring the split": (
        {**RELEASED_WEIGHTS, "released": False, "restoring": True},
        "engine_busy",
        "restoring its weights",
    ),
    "serving": (
        {**RELEASED_WEIGHTS, "released": False},
        "engine_busy",
        "serving requests",
    ),
}


class MemoryReleaseTests(HarnessTestCase):
    def post(self, harness, headers, data=None):
        """A POST of exactly these headers and bytes, and its status and body."""
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/v1/memory/release")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders(data)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_preconfigured_readonly_runtime_limit_keeps_http_lifecycle(self):
        class FixedQueueRuntime:
            def __init__(self):
                self.runtime = FakeRuntime()

            @property
            def pending_limit(self):
                return self.runtime.pending_limit

            def __getattr__(self, name):
                return getattr(self.runtime, name)

        runtime = FixedQueueRuntime()
        harness = self.harness(runtime, queue_size=runtime.pending_limit)
        self.assertEqual(
            self.post(harness, {"Content-Length": "0"}),
            (200, {"released": True, "weights": RELEASED_WEIGHTS}),
        )

    def test_an_idle_engine_releases_for_a_request_without_fields(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for headers, data in (
            ({}, None),
            ({"Content-Length": "0"}, None),
            ({"Content-Type": "application/json", "Content-Length": "2"}, b"{}"),
        ):
            with self.subTest(headers=headers):
                self.assertEqual(
                    self.post(harness, headers, data),
                    (200, {"released": True, "weights": RELEASED_WEIGHTS}),
                )
        self.assertEqual(runtime.release_calls, 3)

    def test_an_encoder_explicitly_reports_unsupported_release(self):
        runtime = FakeRuntime()
        runtime.release_event = native_peer.status_event(
            weights={"release_supported": False, "released": False, "restoring": False}
        )
        harness = self.harness(runtime)
        status, payload = self.post(harness, {})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "unsupported_operation")
        self.assertIn("does not support", payload["error"]["message"])
        self.assertEqual(runtime.release_calls, 1)

    def test_the_idle_release_off_does_not_stop_a_release(self):
        # --idle-release off turns only the timer off: the engine releases on
        # request, and the server answers its document as for any release.
        weights = {**RELEASED_WEIGHTS, "idle_release_seconds": None}
        runtime = FakeRuntime()
        runtime.release_event = native_peer.status_event(weights=weights)
        harness = self.harness(runtime)
        self.assertEqual(
            self.post(harness, {}), (200, {"released": True, "weights": weights})
        )
        self.assertEqual(runtime.release_calls, 1)

    def test_a_body_with_fields_or_not_json_is_refused_before_the_engine(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for headers, data, status in (
            (
                {"Content-Type": "application/json", "Content-Length": "8"},
                b'{"x": 1}',
                400,
            ),
            ({"Content-Type": "application/json", "Content-Length": "2"}, b"[]", 400),
            ({"Content-Type": "text/plain", "Content-Length": "2"}, b"{}", 415),
        ):
            with self.subTest(data=data):
                self.assertEqual(self.post(harness, headers, data)[0], status)
        self.assertEqual(runtime.release_calls, 0)

    def test_an_engine_that_keeps_its_weights_gets_an_unlogged_409(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for case, (weights, code, words) in KEPT.items():
            runtime.release_event = native_peer.status_event(weights=weights)
            with (
                self.subTest(case),
                mock.patch.object(api, "print_status") as console,
            ):
                status, payload = self.post(harness, {})
                self.assertEqual((status, payload["error"]["code"]), (409, code))
                self.assertIn(words, payload["error"]["message"])
                # The engine's state, not a failure: the console does not log it.
                console.assert_not_called()
        self.assertEqual(runtime.release_calls, len(KEPT))

    def test_an_engine_that_does_not_answer_or_fails_is_unavailable(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for error, code in (
            (TimeoutError("native status response timed out"), "engine_busy"),
            (
                engine_runtime.EngineUnhealthy("native loop failed"),
                "runtime_unavailable",
            ),
        ):
            with self.subTest(code=code):
                with mock.patch.object(runtime, "release_weights", side_effect=error):
                    status, payload = self.post(harness, {})
                self.assertEqual((status, payload["error"]["code"]), (503, code))

    def test_the_release_needs_the_api_key(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, api_key="test-server-key")
        self.assertEqual(self.post(harness, {})[0], 401)
        self.assertEqual(
            self.post(harness, {"Authorization": "Bearer test-server-key"})[0], 200
        )
        self.assertEqual(runtime.release_calls, 1)


class RealModelMemoryReleaseTests(HarnessTestCase):
    """Existing HTTP harness over real MultiplexedRuntime owners.

    The operator chooses an already prepared, separately authorized model;
    the test never prepares/downloads it or changes launcher defaults.
    """

    def _close_real_runtime(self, runtime, processes):
        runtime.close()
        for process in processes:
            self.assertIsNotNone(process.poll())
            print(
                "real_public_cleanup="
                + json.dumps({"pid": process.pid, "exit": process.returncode}),
                flush=True,
            )

    def _open_real_owner(self, model_root, budget, model_name, mode, options):
        argv = [
            str(model_root),
            "--model",
            model_name,
            "--max-context",
            "8192",
            "--max-memory",
            budget,
            "--max-cache-disk",
            "0",
            "--queue-size",
            "1",
            "--kv-format",
            "q4h",
            "--disable-ane",
            "--verify-rows",
            "all",
            "--idle-release",
            "off",
        ]
        if mode == "on":
            argv += ["--moe-memory-pressure", "on"]
        if options.real_binary is not None:
            argv += ["--binary", str(options.real_binary)]
        if ple := options.real_ple_memory:
            argv += ["--ple-memory", ple]
        args = api.parse_args(argv)
        self.assertEqual(args.moe_memory_pressure, mode)
        configured = list(startup_report.configuration_lines(args))
        self.assertIn(f"  --moe-memory-pressure = {mode}", configured)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            api.parse_args(
                argv + ["--moe-memory-pressure", "on", "--expert-memory", "1G"]
            )
        root, held = api.assembly.hold(Path(args.model_root))
        self.addCleanup(held.close)
        tokenizer_options = {}
        config = root / "tokenizer/config.json"
        if config.is_file() and json.loads(config.read_text()).get("model_type") in (
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        ):
            tokenizer_options["fix_mistral_regex"] = False
        tokenizer = api.AutoTokenizer.from_pretrained(
            root / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
            **tokenizer_options,
        )
        api.validate_tokenizer(tokenizer)
        command = api._native_command(args, root)
        processes = []

        def spawn():
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=0,
            )
            processes.append(process)
            print(
                "real_public_process="
                + json.dumps(
                    {
                        "pid": process.pid,
                        "command": command,
                        "pressure_case": "default-off" if mode == "off" else "on",
                    }
                ),
                flush=True,
            )
            return process

        runtime = engine_runtime.MultiplexedRuntime(
            command, process_factory=spawn, startup_timeout=30, pending_limit=1
        )
        self.addCleanup(self._close_real_runtime, runtime, processes)
        self.assertTrue(runtime.wait_ready())
        harness = self.harness(
            runtime,
            tokenizer=tokenizer,
            model=args.model,
            queue_size=1,
            max_context=8192,
            timeout=15,
            vision=False,
        )

        return harness, runtime, args, configured, processes

    def _real_request(self, harness, method, path, body=None):
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=20
        )
        try:
            connection.request(
                method,
                path,
                None if body is None else json.dumps(body),
                {} if body is None else {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200, payload)
            return payload
        finally:
            connection.close()

    def _real_status(self, harness, phase):
        snapshot = self._real_request(harness, "GET", "/status")
        self.assertTrue(snapshot["ready"])
        self.assertEqual(snapshot["transport"]["restarts"], 0)
        self.assertFalse(snapshot["transport"]["status_stale"])
        print(
            "real_public_status=" + json.dumps({"phase": phase, "snapshot": snapshot}),
            flush=True,
        )
        return snapshot

    def test_public_startup_generation_release_and_next_request_restore(self):
        options = REAL_MODEL_OPTIONS
        if options is None or options.real_model_root is None:
            self.skipTest("explicit --real-model-root correctness run only")
        mode = options.real_moe_pressure
        self.assertIn(mode, ("off", "on"))
        harness, runtime, args, configured, processes = self._open_real_owner(
            options.real_model_root, options.real_max_memory, "local/T10", mode, options
        )
        before = self._real_status(harness, "ready")
        pressure = before["moe_memory_pressure"]
        self.assertEqual(pressure["enabled"], mode == "on")
        self.assertEqual(pressure["mode"], "local" if mode == "on" else "off")
        self.assertTrue(pressure["model_ready"])
        self.assertTrue(
            {
                "role",
                "group",
                "instance",
                "leader",
                "epoch",
                "coordination_ready",
                "heartbeat_age_seconds",
            }.isdisjoint(pressure)
        )
        effective = list(
            startup_report.effective_lines(before, runtime.readiness, args)
        )
        print(
            "real_public_manifest="
            + json.dumps({"configured": configured, "effective": effective}),
            flush=True,
        )
        body = {
            "model": args.model,
            "prompt": list(range(1, 9)),
            "max_tokens": 16,
            "temperature": 0,
            "seed": 314159,
        }
        first = self._real_request(
            harness, "POST", "/v1/completions", {**body, "cache_salt": "t10-before"}
        )
        self.assertEqual(first["usage"]["prompt_tokens"], 8)
        self.assertEqual(first["usage"]["completion_tokens"], 16)
        generated = self._real_status(harness, "generated")
        released = self._real_request(harness, "POST", "/v1/memory/release")
        self.assertTrue(released["released"])
        during = self._real_status(harness, "released")
        self.assertTrue(during["weights"]["released"])
        self.assertLess(
            during["memory_actual"]["allocated_bytes"],
            generated["memory_actual"]["allocated_bytes"],
        )
        second = self._real_request(
            harness, "POST", "/v1/completions", {**body, "cache_salt": "t10-after"}
        )
        self.assertEqual(first["choices"], second["choices"])
        self.assertEqual(
            first["usage"]["completion_tokens"], second["usage"]["completion_tokens"]
        )
        after = self._real_status(harness, "restored")
        self.assertFalse(after["weights"]["released"])
        self.assertFalse(after["weights"]["restoring"])
        self.assertEqual(
            after["weights"]["restores"], generated["weights"]["restores"] + 1
        )
        self.assertEqual(after["memory_plan"], before["memory_plan"])
        self.assertTrue(after["moe_memory_pressure"]["model_ready"])
        self.assertEqual(after["moe_memory_pressure"]["mode"], pressure["mode"])
        self.assertFalse(during["moe_memory_pressure"]["model_ready"])
        self.assertEqual(len(processes), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--real-model-root", type=Path)
    parser.add_argument("--real-binary", type=Path)
    parser.add_argument("--real-max-memory", default="8G")
    parser.add_argument(
        "--real-moe-pressure",
        choices=("off", "on"),
        default="on",
        help="off checks the omitted native flag (default off); on explicitly enables it",
    )
    parser.add_argument("--real-ple-memory")
    REAL_MODEL_OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *remaining])
