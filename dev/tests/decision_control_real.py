"""Real encoder protocol, cancellation and shared-workspace isolation check."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from install.gguf import Metadata
from server import protocol as wire
from server.runtime import GenerationRequest, MultiplexedRuntime, RequestFailed


def run(args):
    source = next((args.model_root / "target").glob("*.gguf"))
    metadata = Metadata(source)
    d3 = metadata.values.get("qwen35.decision.type") in {"vllm-d3", "h2o-lightning"}
    context = args.context if args.context is not None else (262144 if d3 else 8192)
    bos = 248045 if d3 else metadata.values["tokenizer.ggml.bos_token_id"]
    sep = 248046 if d3 else metadata.values["tokenizer.ggml.seperator_token_id"]
    marker = 8 if d3 else metadata.values["tokenizer.ggml.mask_token_id"]
    option_ids = (
        tuple(
            json.loads((args.model_root / "config.json").read_text())[
                "decision_config"
            ]["token_ids"][:2]
        )
        if d3
        else (3, 5)
    )
    command = [
        str(args.binary),
        "serve-d3-native" if d3 else "serve-encoder-native",
        str(args.model_root),
        str(context),
        str((40 if d3 else 4) << 30),
    ]
    if d3:
        command.extend(("--kv-format", "q4h"))
    runtime = MultiplexedRuntime(command, startup_timeout=180, pending_limit=4)
    result = {"command": command, "checks": [], "performance_test": False}
    process = None
    try:
        assert runtime.wait_ready(180)
        process = runtime._process
        assert process is not None
        result["pid"] = process.pid
        readiness = runtime.capabilities
        assert readiness is not None
        startup_status = json.loads(runtime.status(timeout=2).json)
        assert startup_status["maximum_context_tokens"] == context
        assert startup_status["maximum_context_tokens"] == readiness.max_context_tokens
        assert readiness.max_concurrent_requests == 4
        assert startup_status["encoder_requests"] == []
        result["checks"].append(
            "status context matches configured and native Ready capacity"
        )
        completed = []

        def record(answer):
            assert answer.completion_tokens == 0 and answer.decode_micros == 0
            if answer.prompt_tokens:
                completed.append(answer)
            return answer

        def snapshot():
            status = json.loads(runtime.status(timeout=2).json)
            assert status["generation"] is False
            rows = status["encoder_requests"]
            assert status["active_requests"] == len(rows)
            assert len({row["id"] for row in rows}) == len(rows)
            for row in rows:
                assert row["phase"] in {"queued", "running", "prefill"}
                assert row["prompt_tokens"] > 0
                assert row["prompt_processed"] is None
                assert row["generated_tokens"] == 0
                assert isinstance(row["cancellation_requested"], bool)
                assert "slot" not in row
            return status

        def telemetry():
            status = snapshot()
            counters = status["telemetry"]
            assert counters["completed_requests"] == len(completed)
            assert counters["total_input_tokens"] == sum(
                answer.prompt_tokens for answer in completed
            )
            assert counters["total_prefill_wall_seconds"] >= 0
            assert counters["total_prefill_gpu_seconds"] >= 0
            return counters

        initial = telemetry()
        assert initial["submission_batches"] == 0
        assert initial["total_prefill_wall_seconds"] == 0
        assert initial["total_prefill_gpu_seconds"] == 0
        flags = [
            wire.RequestFlag.DECISION_SCORING,
            wire.RequestFlag.DECISION_SCORING | wire.RequestFlag.DECISION_SCORE_TYPE,
            wire.RequestFlag.DECISION_SCORING | wire.RequestFlag.DECISION_NOUL_TYPE,
        ]
        if d3:
            flags = [wire.RequestFlag(0)] * 3

        def request(kind=0, number=0, long=False):
            prompt = [bos, 8 + number, sep, marker, 10, marker, 11, sep]
            prompt += [12] * (min(8192, context) - len(prompt) - 1 if long else 1)
            prompt += [sep]
            frame = wire.RequestFrame(
                request_id=0,
                priority=wire.RequestPriority.NORMAL,
                absolute_deadline_unix_micros=0,
                remaining_deadline_micros=0,
                logical_max_output_tokens=0,
                prompt_tokens=tuple(prompt),
                sampling=wire.SamplingParameters(),
                seed=1,
                constraint=wire.ConstraintMode.NONE,
                image_spans=(),
                image_pixels=b"",
                score_tokens=option_ids,
                generation_prompt_tokens=0,
                flags=flags[kind],
                cache_salt=0,
            )
            return GenerationRequest(
                frame, time.monotonic() + 120, None, wire.request_fingerprint(frame)
            )

        serial = []
        previous = initial
        for i in range(4):
            answer = record(runtime.submit(request(i % 3, i)).result(120))
            serial.append(answer)
            current = telemetry()
            assert math.isclose(
                current["total_prefill_wall_seconds"]
                - previous["total_prefill_wall_seconds"],
                answer.prefill_micros / 1e6,
                rel_tol=2e-5,
                abs_tol=1e-6,
            )
            previous = current
        parallel = [runtime.submit(request(i % 3, i)) for i in range(4)]
        parallel_answers = []
        for baseline, pending in zip(serial, parallel):
            answer = record(pending.result(120))
            parallel_answers.append(answer)
            assert answer.reason is wire.FinishReason.STOP
            assert len(answer.option_logits) == 2
            assert all(
                abs(x - y) <= 1e-3 * max(1, abs(x))
                for x, y in zip(answer.option_logits, baseline.option_logits)
            )
        result["checks"].append(
            "four distinct requests and mixed types match serial at the existing 1e-3 bound"
        )
        current = telemetry()
        batch_times = {answer.prefill_micros for answer in parallel_answers}
        batches = current["submission_batches"] - previous["submission_batches"]
        if len(batch_times) == batches:
            # Every Done in a shared batch repeats that batch's wall time.
            assert math.isclose(
                current["total_prefill_wall_seconds"]
                - previous["total_prefill_wall_seconds"],
                sum(batch_times) / 1e6,
                rel_tol=2e-5,
                abs_tol=1e-6,
            )

        invalid = request()
        invalid = replace(
            invalid, frame=replace(invalid.frame, score_tokens=(0, 1) if d3 else (3, 4))
        )
        try:
            runtime.submit(invalid).result(120)
        except RequestFailed:
            pass
        else:
            raise AssertionError("non-marker score position was accepted")
        assert telemetry() == current
        assert (
            record(runtime.submit(request()).result(120)).reason
            is wire.FinishReason.STOP
        )
        assert process.pid == runtime._process.pid
        result["checks"].append(
            "invalid position fails one request and the same process remains usable"
        )

        def wait_for_rows(predicate):
            deadline = time.monotonic() + 10
            while True:
                status = snapshot()
                if predicate(status["encoder_requests"]):
                    return status
                assert time.monotonic() < deadline, status
                time.sleep(0.01)

        pending = runtime.submit(request(long=True))
        wait_for_rows(
            lambda rows: any(
                row["id"] == pending.request_id and row["phase"] == "prefill"
                for row in rows
            )
        )
        queued = runtime.submit(request())
        observed = snapshot()
        rows = {row["id"]: row for row in observed["encoder_requests"]}
        assert rows[pending.request_id]["phase"] == "prefill"
        assert rows[pending.request_id]["prompt_tokens"] == min(8192, context)
        assert rows[queued.request_id]["phase"] == "queued"
        assert rows[queued.request_id]["prompt_tokens"] == 10
        result["active_status_sample"] = observed
        assert queued.cancel()
        assert pending.cancel()
        observed = snapshot()
        assert {row["id"] for row in observed["encoder_requests"]} == {
            pending.request_id,
            queued.request_id,
        }
        assert all(
            row["cancellation_requested"] for row in observed["encoder_requests"]
        )
        result["cancel_status_sample"] = observed
        cancelled = record(pending.result(120))
        assert (
            cancelled.reason is wire.FinishReason.CANCELLED
            and not cancelled.option_logits
        )
        queued_cancelled = queued.result(120)
        assert queued_cancelled.reason is wire.FinishReason.CANCELLED
        assert queued_cancelled.prompt_tokens == 10
        assert queued_cancelled.prefill_micros == 0
        assert queued_cancelled.completion_tokens == queued_cancelled.decode_micros == 0
        wait_for_rows(lambda rows: not rows)
        telemetry()
        assert (
            record(runtime.submit(request()).result(120)).reason
            is wire.FinishReason.STOP
        )
        result["checks"].append(
            "queued and prefill tasks expose real IDs and input lengths, unknown progress, cancellation and retirement"
        )
        release = json.loads(runtime.release_weights(timeout=2).json)
        assert release["weights"]["release_supported"] is False
        assert release["weights"]["released"] is False
        assert record(runtime.submit(request()).result(120)).option_logits
        telemetry()
        result["checks"].append(
            "encoder telemetry excludes warmup and invalid work, counts executed cancelled inputs, and records batch wall time once"
        )
        result["checks"].append("unsupported encoder release preserves usable weights")
        result["status"] = json.loads(runtime.status(timeout=2).json)
        result["status_result"] = "passed"
    finally:
        runtime.close()
        if process is not None:
            result["exit"] = process.returncode
        args.output.write_text(json.dumps(result, indent=2))
    assert result["exit"] == 0
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": len(result["checks"]),
                "exit": result["exit"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
