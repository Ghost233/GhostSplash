"""In-memory correlation of HTTP operations with native requests.

Records are local to one HTTP service incarnation. A missing record after a
restart is not evidence that the former native request stopped.
"""

import hashlib
import hmac
import re
import threading
import time

_ID = re.compile(r"[A-Za-z0-9._~-]{1,128}\Z")
_TOKEN = re.compile(r"[0-9a-fA-F]{64}\Z")


class Operation:
    def __init__(
        self,
        operation_id,
        token,
        instance,
        clock,
        expected_generation=None,
        kind="consume",
    ):
        self.id = operation_id
        self.instance = instance
        self.expected_generation = expected_generation
        self.kind = kind
        self._credential = hashlib.sha256(bytes.fromhex(token)).digest()
        self._lock = threading.Lock()
        self._children = {}
        self._jobs = {}
        self._handler_outcome = None
        self._cancel_requested = False
        self._clock = clock
        self._retired_at = None

    def _retire_if_finished(self):
        if (
            self._handler_outcome is not None
            and not self._jobs
            and all(
                child.get("lease_released", False) for child in self._children.values()
            )
        ):
            if self._retired_at is None:
                self._retired_at = self._clock()

    def matches(self, token):
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            return False
        return hmac.compare_digest(
            self._credential, hashlib.sha256(bytes.fromhex(token)).digest()
        )

    def add(self, job, question_id):
        with self._lock:
            if self._cancel_requested or self._handler_outcome is not None:
                return False
            self._jobs[job.request_id] = job
            self._children[job.request_id] = {
                "question_id": question_id,
                "state": "submitting",
            }
            return True

    def bind(self, job, call):
        with self._lock:
            child = self._children[job.request_id]
            child["native_request_id"] = call.request_id
            child["runtime_generation"] = call.generation
            if child["state"] == "submitting":
                child["state"] = "running"

    def completed(self, job, call, done):
        with self._lock:
            child = self._children[job.request_id]
            child.update(
                native_request_id=call.request_id,
                runtime_generation=call.generation,
                state=done.reason.name.lower(),
                prompt_tokens=done.prompt_tokens,
                completion_tokens=done.completion_tokens,
                deduplicated_tokens=done.deduplicated_tokens,
                lease_released=True,
            )
            if job.cache.status != "unknown":
                child["matched_prompt_tokens"] = job.cache.matched_tokens
            self._jobs.pop(job.request_id, None)
            self._retire_if_finished()

    def failed(self, job, call):
        with self._lock:
            child = self._children[job.request_id]
            child.update(
                native_request_id=call.request_id,
                runtime_generation=call.generation,
                state="failed",
                lease_released=True,
            )
            self._jobs.pop(job.request_id, None)
            self._retire_if_finished()

    def unknown(self, job):
        with self._lock:
            child = self._children[job.request_id]
            # A late HTTP/finalizer error cannot revoke an observed native
            # terminal, nor can it turn an unconfirmed request into one.
            if child.get("lease_released", False):
                return
            child.update(state="unknown", lease_released=False)
            self._retired_at = None
            self._jobs.pop(job.request_id, None)

    def not_admitted(self, job):
        with self._lock:
            self._children[job.request_id].update(
                state="not_admitted", lease_released=True
            )
            self._jobs.pop(job.request_id, None)
            self._retire_if_finished()

    def request_cancel(self):
        with self._lock:
            self._cancel_requested = True
            return list(self._jobs.values())

    def handler_finished(self, outcome):
        with self._lock:
            self._handler_outcome = "cancelled" if self._cancel_requested else outcome
            self._retire_if_finished()

    def expired(self, now, retention):
        with self._lock:
            return self._retired_at is not None and now - self._retired_at >= retention

    def snapshot(self):
        with self._lock:
            children = [dict(child) for child in self._children.values()]
            unknown = any(child["state"] == "unknown" for child in children)
            released = all(child.get("lease_released", False) for child in children)
            terminal = self._handler_outcome is not None and released and not unknown
            state = (
                "unknown"
                if unknown
                else self._handler_outcome
                if terminal
                else "cancel_requested"
                if self._cancel_requested
                else "running"
            )
            return {
                "operation_id": self.id,
                "service_instance": self.instance,
                "state": state,
                "terminal": terminal,
                "lease_released": terminal,
                "children": children,
            }


class OperationStore:
    def __init__(
        self,
        instance,
        capacity=4096,
        prefill_capacity=3584,
        retention=3600,
        clock=time.monotonic,
    ):
        self.instance = instance
        self.capacity = capacity
        self.prefill_capacity = min(prefill_capacity, capacity)
        self._prefill_count = 0
        self.retention = retention
        self.clock = clock
        self._lock = threading.Lock()
        self._operations = {}

    def create(self, operation_id, token, expected_generation=None, kind="consume"):
        if (
            not isinstance(operation_id, str)
            or not _ID.fullmatch(operation_id)
            or not isinstance(token, str)
            or not _TOKEN.fullmatch(token)
            or kind not in ("prefill", "consume")
        ):
            raise ValueError("invalid operation identity or credential")
        with self._lock:
            if operation_id in self._operations:
                raise ValueError("operation already exists")
            if len(self._operations) >= self.capacity or (
                kind == "prefill" and self._prefill_count >= self.prefill_capacity
            ):
                now = self.clock()
                self._operations = {
                    key: operation
                    for key, operation in self._operations.items()
                    if not operation.expired(now, self.retention)
                }
                self._prefill_count = sum(
                    operation.kind == "prefill"
                    for operation in self._operations.values()
                )
            if len(self._operations) >= self.capacity or (
                kind == "prefill" and self._prefill_count >= self.prefill_capacity
            ):
                raise ValueError("operation registry is full")
            operation = Operation(
                operation_id,
                token,
                self.instance,
                self.clock,
                expected_generation,
                kind,
            )
            self._operations[operation_id] = operation
            self._prefill_count += kind == "prefill"
            return operation

    def find(self, operation_id, token):
        with self._lock:
            operation = self._operations.get(operation_id)
        return operation if operation is not None and operation.matches(token) else None

    def get(self, operation_id, token):
        operation = self.find(operation_id, token)
        return None if operation is None else operation.snapshot()

    def acknowledge(self, operation_id, token):
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None or not operation.matches(token):
                return "missing"
            if not operation.snapshot()["terminal"]:
                return "pending"
            del self._operations[operation_id]
            self._prefill_count -= operation.kind == "prefill"
            return "removed"
