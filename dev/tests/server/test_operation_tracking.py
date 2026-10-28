import unittest
from types import SimpleNamespace

from server import protocol as wire
from server.operation_tracking import OperationStore


class OperationTrackingTests(unittest.TestCase):
    def test_operation_waits_for_native_terminal_after_handler_disconnects(self):
        store = OperationStore("instance-a")
        operation = store.create("op-1", "a" * 64)
        job = SimpleNamespace(
            request_id=4, cache=SimpleNamespace(status="hit", matched_tokens=7)
        )
        operation.add(job, "question-a")
        call = SimpleNamespace(request_id=19, generation=3)
        operation.bind(job, call)
        operation.handler_finished("cancelled")
        before = store.get("op-1", "a" * 64)
        self.assertFalse(before["terminal"])
        self.assertFalse(before["lease_released"])
        done = SimpleNamespace(
            reason=wire.FinishReason.CANCELLED,
            prompt_tokens=30,
            completion_tokens=0,
            deduplicated_tokens=2,
        )
        operation.completed(job, call, done)
        after = store.get("op-1", "a" * 64)
        self.assertTrue(after["terminal"])
        self.assertTrue(after["lease_released"])
        self.assertEqual(after["children"][0]["native_request_id"], 19)
        self.assertEqual(after["children"][0]["runtime_generation"], 3)
        self.assertEqual(after["children"][0]["question_id"], "question-a")
        self.assertEqual(after["children"][0]["matched_prompt_tokens"], 7)
        self.assertEqual(after["children"][0]["deduplicated_tokens"], 2)
        self.assertIsNone(store.get("op-1", "b" * 64))

    def test_lost_native_terminal_is_unknown_not_released(self):
        store = OperationStore("instance-a")
        operation = store.create("op-2", "a" * 64)
        job = SimpleNamespace(request_id=4, cache=SimpleNamespace(matched_tokens=0))
        operation.add(job, None)
        operation.unknown(job)
        operation.handler_finished("failed")
        state = store.get("op-2", "a" * 64)
        self.assertEqual(state["state"], "unknown")
        self.assertFalse(state["terminal"])
        self.assertFalse(state["lease_released"])

    def test_duplicate_operation_cannot_execute_twice(self):
        store = OperationStore("instance-a")
        store.create("op-3", "a" * 64)
        with self.assertRaises(ValueError):
            store.create("op-3", "a" * 64)

    def test_only_settled_records_expire(self):
        now = [0]
        store = OperationStore(
            "instance-a", capacity=1, retention=10, clock=lambda: now[0]
        )
        first = store.create("op-1", "a" * 64)
        now[0] = 11
        with self.assertRaises(ValueError):
            store.create("op-2", "a" * 64)
        first.handler_finished("completed")
        now[0] = 22
        store.create("op-2", "a" * 64)
        self.assertIsNone(store.get("op-1", "a" * 64))

    def test_acknowledgement_cannot_remove_live_or_unknown_operation(self):
        store = OperationStore("instance-a", capacity=1)
        operation = store.create("op-1", "a" * 64)
        job = SimpleNamespace(
            request_id=4, cache=SimpleNamespace(status="unknown", matched_tokens=0)
        )
        operation.add(job, None)
        self.assertEqual(store.acknowledge("op-1", "a" * 64), "pending")
        operation.unknown(job)
        operation.handler_finished("failed")
        self.assertEqual(store.acknowledge("op-1", "a" * 64), "pending")
        self.assertEqual(store.acknowledge("op-1", "b" * 64), "missing")
        with self.assertRaises(ValueError):
            store.create("op-2", "a" * 64)

    def test_acknowledgement_reclaims_only_terminal_operation(self):
        store = OperationStore("instance-a", capacity=1)
        operation = store.create("op-1", "a" * 64)
        operation.handler_finished("completed")
        self.assertEqual(store.acknowledge("op-1", "a" * 64), "removed")
        self.assertIsNone(store.get("op-1", "a" * 64))
        store.create("op-2", "a" * 64)

    def test_late_finalizer_failure_does_not_revoke_native_release(self):
        store = OperationStore("instance-a")
        operation = store.create("op-1", "a" * 64)
        job = SimpleNamespace(
            request_id=4, cache=SimpleNamespace(status="miss", matched_tokens=0)
        )
        operation.add(job, None)
        call = SimpleNamespace(request_id=19, generation=3)
        operation.handler_finished("cancelled")
        done = SimpleNamespace(
            reason=wire.FinishReason.CANCELLED,
            prompt_tokens=30,
            completion_tokens=0,
            deduplicated_tokens=0,
        )
        operation.completed(job, call, done)
        operation.unknown(job)
        self.assertTrue(operation.snapshot()["lease_released"])
        self.assertEqual(store.acknowledge("op-1", "a" * 64), "removed")
        operation.unknown(job)
        self.assertIsNone(store.get("op-1", "a" * 64))

    def test_prefill_cannot_consume_reserved_final_capacity(self):
        store = OperationStore("instance-a", capacity=4, prefill_capacity=3)
        for index in range(3):
            store.create(f"prefill-{index}", "a" * 64, kind="prefill")
        with self.assertRaises(ValueError):
            store.create("prefill-extra", "a" * 64, kind="prefill")
        final = store.create("final", "b" * 64, kind="consume")
        final.handler_finished("completed")
        self.assertEqual(store.acknowledge("final", "b" * 64), "removed")
        store.create("another-final", "b" * 64, kind="consume")
