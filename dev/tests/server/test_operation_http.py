import http.client
import json
import threading
import time
from unittest import mock

from dev.tests.server.test_d3 import config as d3_config
from dev.tests.server.test_judgments import TURNS_TEMPLATE, byte_tokenizer
from dev.tests.server.test_kev import KevTokenizer
from dev.tests.server.test_kev import config as kev_config
from dev.tests.server_fixtures import (
    CharTokenizer,
    FakeRuntime,
    HarnessTestCase,
    Plan,
    chat_body,
)
from server import protocol as wire
from server import runtime
from server.modernbert import ModernBertScorer


class OperationHttpTests(HarnessTestCase):
    def test_runtime_prefill_capabilities_follow_instance_and_readiness(self):
        native = FakeRuntime()
        harness = self.harness(
            native, tokenizer=byte_tokenizer(TURNS_TEMPLATE), max_context=8192
        )
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        runtime = json.loads(payload)
        capabilities = runtime["prefill_capabilities"]
        self.assertEqual(runtime["service_instance"], harness.server.instance_id)
        self.assertEqual(runtime["runtime_generation"], 1)
        self.assertEqual(capabilities["schema"], "splash-prefill/1")
        self.assertTrue(capabilities["operation_tracking"])
        self.assertTrue(capabilities["chat_zero_budget"])
        self.assertTrue(capabilities["systemone_state"])
        native.ready = False
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        runtime = json.loads(payload)
        self.assertIsNone(runtime["runtime_generation"])
        self.assertFalse(runtime["prefill_capabilities"]["operation_tracking"])
        self.assertFalse(runtime["prefill_capabilities"]["chat_zero_budget"])
        self.assertFalse(runtime["prefill_capabilities"]["systemone_state"])

    def test_runtime_capabilities_refuse_noncausal_decision_models(self):
        for decision in (d3_config(),):
            native = FakeRuntime()
            harness = self.harness(
                native,
                tokenizer=CharTokenizer(),
                max_context=8192,
                decision_config=decision,
                vision=False,
            )
            status, _, payload = harness.request("GET", "/v1/splash/runtime")
            self.assertEqual(status, 200, payload)
            caps = json.loads(payload)["prefill_capabilities"]
            self.assertTrue(caps["operation_tracking"])
            self.assertFalse(caps["chat_zero_budget"])
            self.assertFalse(caps["systemone_state"])
        harness.app.decision_scorer = object.__new__(ModernBertScorer)
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        self.assertFalse(json.loads(payload)["prefill_capabilities"]["systemone_state"])

    def test_runtime_capabilities_confirm_kev_state_prefix(self):
        native = FakeRuntime()
        harness = self.harness(
            native,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=kev_config(),
            vision=False,
        )
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        caps = json.loads(payload)["prefill_capabilities"]
        self.assertFalse(caps["chat_zero_budget"])
        self.assertTrue(caps["systemone_state"])

    def test_tracked_kev_prefill_and_questions_share_actual_prefix(self):
        native = FakeRuntime(Plan(reason="length"), Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            native,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=kev_config(),
            vision=False,
        )
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "kev-prefill",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        body = {
            "model": harness.app.model,
            "state": "state material",
            "prefill_only": True,
        }
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body, headers=headers
        )
        self.assertEqual(status, 200, payload)
        body.pop("prefill_only")
        body["questions"] = {"q": {"type": "noul"}}
        headers["X-Splash-Operation-ID"] = "kev-consume"
        headers["X-Splash-Operation-Kind"] = "consume"
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body, headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(native.requests), 2)

    def test_tracked_d3_normal_judgment_is_not_blocked(self):
        native = FakeRuntime(Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            native,
            tokenizer=CharTokenizer(),
            max_context=8192,
            decision_config=d3_config(),
            vision=False,
        )
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "d3-consume",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "consume",
        }
        body = {
            "model": harness.app.model,
            "state": "input",
            "questions": {"q": {"type": "noul"}},
        }
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body, headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(native.requests), 1)

    def test_runtime_capabilities_refuse_question_dependent_prefix(self):
        template = TURNS_TEMPLATE.replace(
            "{{ message.content }}",
            "{% if 'options' in message.content %}QUESTION-FIRST{% endif %}{{ message.content }}",
        )
        harness = self.harness(
            FakeRuntime(), tokenizer=byte_tokenizer(template), max_context=8192
        )
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        caps = json.loads(payload)["prefill_capabilities"]
        self.assertTrue(caps["chat_zero_budget"])
        self.assertFalse(caps["systemone_state"])

    def test_capability_probe_error_is_reported_and_stable_for_instance(self):
        harness = self.harness(
            FakeRuntime(), tokenizer=byte_tokenizer(TURNS_TEMPLATE), max_context=8192
        )
        with (
            mock.patch.object(
                harness.app,
                "systemone_state_prefix",
                side_effect=RuntimeError("private input"),
            ),
            mock.patch("server.frontend.print_status") as log,
        ):
            status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        caps = json.loads(payload)["prefill_capabilities"]
        self.assertFalse(caps["systemone_state"])
        self.assertEqual(caps["systemone_reason"], "probe_error")
        self.assertNotIn("private input", str(log.call_args))
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        self.assertFalse(json.loads(payload)["prefill_capabilities"]["systemone_state"])

    def test_tracked_systemone_rejects_actual_question_that_changes_state_prefix(self):
        template = TURNS_TEMPLATE.replace(
            "{{ message.content }}",
            "{% if 'trigger-reorder' in message.content %}QUESTION-FIRST{% endif %}{{ message.content }}",
        )
        native = FakeRuntime(Plan(reason="length"), Plan(logits=(1.0, 0.0)))
        harness = self.harness(
            native, tokenizer=byte_tokenizer(template), max_context=8192
        )
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["prefill_capabilities"]["systemone_state"])
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "prefix",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        body = {
            "model": "test-model",
            "state": {"material": "X" * 128},
            "prefill_only": True,
            "cache_salt": "test",
        }
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body, headers=headers
        )
        self.assertEqual(status, 200, payload)
        body.pop("prefill_only")
        body["questions"] = {
            "q": {
                "type": "choice",
                "instructions": "trigger-reorder",
                "criteria": {"a": "a", "b": "b"},
            }
        }
        headers["X-Splash-Operation-ID"] = "consume"
        headers["X-Splash-Operation-Kind"] = "consume"
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body, headers=headers
        )
        self.assertEqual(status, 422, payload)
        self.assertEqual(len(native.requests), 1)
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(native.requests), 2)

    def test_cannot_use_final_reservation_for_prefill(self):
        native = FakeRuntime()
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "wrong-kind",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "consume",
        }
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 400)
        self.assertEqual(native.calls, [])
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/wrong-kind", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["lease_released"])
        self.assertEqual(json.loads(payload)["children"], [])

    def test_full_native_queue_confirms_no_job_was_admitted(self):
        native = FakeRuntime()
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "full",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        with mock.patch.object(
            native, "submit", side_effect=runtime.PendingLimitExceeded("full")
        ):
            status, _, _ = harness.request(
                "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
            )
        self.assertEqual(status, 503)
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/full", headers=headers
        )
        self.assertEqual(status, 200, payload)
        child = json.loads(payload)["children"][0]
        self.assertEqual(child["state"], "not_admitted")
        self.assertTrue(child["lease_released"])
        self.assertEqual(native.calls, [])

    def test_cancel_before_native_submission_never_starts_a_job(self):
        native = FakeRuntime()
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "queued",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        waiting, release = threading.Event(), threading.Event()
        prepare = harness.app.prepare

        def paused(*args, **kwargs):
            waiting.set()
            self.assertTrue(release.wait(2))
            return prepare(*args, **kwargs)

        result = []
        with mock.patch.object(harness.app, "prepare", side_effect=paused):
            worker = threading.Thread(
                target=lambda: result.append(
                    harness.request(
                        "POST",
                        "/v1/chat/completions",
                        chat_body(max_tokens=0),
                        headers=headers,
                    )
                )
            )
            worker.start()
            self.assertTrue(waiting.wait(2))
            status, _, payload = harness.request(
                "POST", "/v1/splash/operations/queued/cancel", {}, headers=headers
            )
            self.assertEqual(status, 200, payload)
            self.assertFalse(json.loads(payload)["terminal"])
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(native.calls, [])
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/queued", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["lease_released"])
        self.assertEqual(json.loads(payload)["children"], [])

    def test_service_and_native_generation_are_bound_before_execution(self):
        native = FakeRuntime()
        harness = self.harness(native)
        status, _, payload = harness.request("GET", "/v1/splash/runtime")
        self.assertEqual(status, 200, payload)
        identity = json.loads(payload)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "stale",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": identity["service_instance"],
            "X-Splash-Runtime-Generation": "2",
            "X-Splash-Operation-Kind": "prefill",
        }
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 409)
        self.assertEqual(native.calls, [])
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/stale", headers=headers
        )
        self.assertEqual(status, 200, payload)
        child = json.loads(payload)["children"][0]
        self.assertEqual(child["state"], "not_admitted")
        self.assertTrue(child["lease_released"])
        headers["X-Splash-Operation-ID"] = "old-instance"
        headers["X-Splash-Service-Instance"] = "old"
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 409)
        self.assertEqual(native.calls, [])

    def test_prefill_operation_reports_native_identity_and_real_counters(self):
        native = FakeRuntime(Plan(reason="length", matched_tokens=1))
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "prefill-1",
            "X-Splash-Operation-Token": "a" * 64,
        }
        headers.update(
            {
                "X-Splash-Service-Instance": harness.server.instance_id,
                "X-Splash-Runtime-Generation": "1",
                "X-Splash-Operation-Kind": "prefill",
            }
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 200, payload)
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/prefill-1", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertNotIn("a" * 64, payload.decode())
        operation = json.loads(payload)
        self.assertTrue(operation["terminal"])
        self.assertTrue(operation["lease_released"])
        self.assertEqual(operation["state"], "completed")
        child = operation["children"][0]
        self.assertEqual(child["native_request_id"], native.calls[0].request_id)
        self.assertEqual(child["runtime_generation"], 1)
        self.assertEqual(child["matched_prompt_tokens"], 1)
        self.assertEqual(child["completion_tokens"], 0)
        self.assertNotIn("newly_evaluated_tokens", child)
        bad = {"X-Splash-Operation-Token": "b" * 64}
        status, _, _ = harness.request(
            "GET", "/v1/splash/operations/prefill-1", headers=bad
        )
        self.assertEqual(status, 404)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 409)
        self.assertEqual(len(native.calls), 1)
        status, _, _ = harness.request(
            "DELETE", "/v1/splash/operations/prefill-1", headers=bad
        )
        self.assertEqual(status, 404)
        status, _, payload = harness.request(
            "DELETE", "/v1/splash/operations/prefill-1", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["acknowledged"])
        status, _, _ = harness.request(
            "GET", "/v1/splash/operations/prefill-1", headers=headers
        )
        self.assertEqual(status, 404)

    def test_cancelled_operation_waits_for_native_terminal(self):
        plan = Plan(reason="length", block=True)
        native = FakeRuntime(plan)
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "prefill-2",
            "X-Splash-Operation-Token": "a" * 64,
        }
        headers.update(
            {
                "X-Splash-Service-Instance": harness.server.instance_id,
                "X-Splash-Runtime-Generation": "1",
                "X-Splash-Operation-Kind": "prefill",
            }
        )
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                harness.request(
                    "POST",
                    "/v1/chat/completions",
                    chat_body(max_tokens=0),
                    headers=headers,
                )
            )
        )
        worker.start()
        self.assertTrue(plan.started.wait(2))
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/prefill-2", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertFalse(json.loads(payload)["lease_released"])
        status, _, payload = harness.request(
            "POST", "/v1/splash/operations/prefill-2/cancel", {}, headers=headers
        )
        self.assertEqual(status, 200, payload)
        worker.join(3)
        self.assertFalse(worker.is_alive())
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/prefill-2", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["lease_released"])
        self.assertEqual(json.loads(payload)["children"][0]["state"], "cancelled")

    def test_engine_loss_does_not_claim_release(self):
        native = FakeRuntime(Plan(exception=runtime.EngineUnhealthy("engine lost")))
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "prefill-3",
            "X-Splash-Operation-Token": "a" * 64,
        }
        headers.update(
            {
                "X-Splash-Service-Instance": harness.server.instance_id,
                "X-Splash-Runtime-Generation": "1",
                "X-Splash-Operation-Kind": "prefill",
            }
        )
        harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/prefill-3", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["state"], "unknown")
        self.assertFalse(json.loads(payload)["lease_released"])

    def test_native_request_error_confirms_terminal_after_release(self):
        failure = runtime.RequestFailed(1, b"capacity_exhausted", b"no capacity")
        native = FakeRuntime(Plan(exception=failure))
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "failure",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 400)
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/failure", headers=headers
        )
        self.assertEqual(status, 200, payload)
        state = json.loads(payload)
        self.assertTrue(state["terminal"])
        self.assertTrue(state["lease_released"])
        self.assertEqual(state["children"][0]["state"], "failed")

    def test_new_service_does_not_claim_prior_operation(self):
        first = self.harness(FakeRuntime(Plan(reason="length")))
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "previous",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": first.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        status, _, payload = first.request(
            "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
        )
        self.assertEqual(status, 200, payload)
        second = self.harness(FakeRuntime())
        self.assertNotEqual(first.server.instance_id, second.server.instance_id)
        status, _, _ = second.request(
            "GET", "/v1/splash/operations/previous", headers=headers
        )
        self.assertEqual(status, 404)

    def test_cancellation_remains_unreleased_until_native_done(self):
        native = FakeRuntime(manual=True)
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "hold",
            "X-Splash-Operation-Token": "a" * 64,
        }
        headers.update(
            {
                "X-Splash-Service-Instance": harness.server.instance_id,
                "X-Splash-Runtime-Generation": "1",
                "X-Splash-Operation-Kind": "prefill",
            }
        )
        worker = threading.Thread(
            target=lambda: harness.request(
                "POST", "/v1/chat/completions", chat_body(max_tokens=0), headers=headers
            )
        )
        worker.start()
        deadline = time.monotonic() + 2
        while not native.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(native.calls)
        call = native.calls[0]
        call.emit(wire.StartEvent(call.request_id, 0, 0))
        status, _, payload = harness.request(
            "POST", "/v1/splash/operations/hold/cancel", {}, headers=headers
        )
        self.assertEqual(status, 200, payload)
        state = json.loads(payload)
        self.assertFalse(state["terminal"])
        self.assertFalse(state["lease_released"])
        self.assertTrue(call.cancel_requested)
        status, _, _ = harness.request(
            "DELETE", "/v1/splash/operations/hold", headers=headers
        )
        self.assertEqual(status, 409)
        call.complete(
            result=wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 0, ()
            )
        )
        worker.join(3)
        self.assertFalse(worker.is_alive())
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/hold", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["lease_released"])

    def test_systemone_tracks_each_native_question(self):
        native = FakeRuntime(Plan(logits=(1.0, 0.0)), Plan(logits=(0.0, 1.0)))
        harness = self.harness(
            native, tokenizer=byte_tokenizer(TURNS_TEMPLATE), max_context=8192
        )
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "questions",
            "X-Splash-Operation-Token": "a" * 64,
        }
        headers.update(
            {
                "X-Splash-Service-Instance": harness.server.instance_id,
                "X-Splash-Runtime-Generation": "1",
                "X-Splash-Operation-Kind": "consume",
            }
        )
        body = {
            "model": "test-model",
            "state": "evidence",
            "questions": {"one": {"type": "noul"}, "two": {"type": "noul"}},
        }
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body, headers=headers
        )
        self.assertEqual(status, 200, payload)
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/questions", headers=headers
        )
        self.assertEqual(status, 200, payload)
        children = json.loads(payload)["children"]
        self.assertEqual([child["question_id"] for child in children], ["one", "two"])
        self.assertEqual([child["native_request_id"] for child in children], [1, 2])

    def test_disconnected_request_retains_operation_until_native_terminal(self):
        native = FakeRuntime(manual=True)
        harness = self.harness(native)
        headers = {
            "Content-Type": "application/json",
            "X-Splash-Operation-ID": "disconnect",
            "X-Splash-Operation-Token": "a" * 64,
            "X-Splash-Service-Instance": harness.server.instance_id,
            "X-Splash-Runtime-Generation": "1",
            "X-Splash-Operation-Kind": "prefill",
        }
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        connection.request(
            "POST", "/v1/chat/completions", json.dumps(chat_body(max_tokens=0)), headers
        )
        deadline = time.monotonic() + 2
        while not native.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(native.calls)
        call = native.calls[0]
        call.emit(wire.StartEvent(call.request_id, 0, 0))
        connection.close()
        deadline = time.monotonic() + 3
        while not call.cancel_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(call.cancel_requested)
        status, _, payload = harness.request(
            "GET", "/v1/splash/operations/disconnect", headers=headers
        )
        self.assertEqual(status, 200, payload)
        self.assertFalse(json.loads(payload)["lease_released"])
        call.complete(
            result=wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 0, ()
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status, _, payload = harness.request(
                "GET", "/v1/splash/operations/disconnect", headers=headers
            )
            if json.loads(payload)["terminal"]:
                break
            time.sleep(0.01)
        self.assertEqual(status, 200, payload)
        self.assertTrue(json.loads(payload)["lease_released"])
