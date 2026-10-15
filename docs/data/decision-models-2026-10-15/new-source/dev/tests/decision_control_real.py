"""Real encoder protocol, cancellation and shared-workspace isolation check."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from install.gguf import Metadata
from server import protocol as wire
from server.runtime import GenerationRequest, MultiplexedRuntime, RequestFailed


def run(args):
    source = next((args.model_root / "target").glob("*.gguf"))
    metadata = Metadata(source)
    bos = metadata.values["tokenizer.ggml.bos_token_id"]
    sep = metadata.values["tokenizer.ggml.seperator_token_id"]
    marker = metadata.values["tokenizer.ggml.mask_token_id"]
    command = [
        str(args.binary),
        "serve-encoder-native",
        str(args.model_root),
        "8192",
        str(4 << 30),
    ]
    runtime = MultiplexedRuntime(command, startup_timeout=180, pending_limit=4)
    result = {"command": command, "checks": [], "performance_test": False}
    process = None
    try:
        assert runtime.wait_ready(180)
        process = runtime._process
        assert process is not None
        result["pid"] = process.pid
        flags = [
            wire.RequestFlag.DECISION_SCORING,
            wire.RequestFlag.DECISION_SCORING | wire.RequestFlag.DECISION_SCORE_TYPE,
            wire.RequestFlag.DECISION_SCORING | wire.RequestFlag.DECISION_NOUL_TYPE,
        ]

        def request(kind=0, number=0, long=False):
            prompt = [bos, 8 + number, sep, marker, 10, marker, 11, sep]
            prompt += [12] * (8192 - len(prompt) - 1 if long else 1)
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
                score_tokens=(3, 5),
                generation_prompt_tokens=0,
                flags=flags[kind],
            )
            return GenerationRequest(frame, time.monotonic() + 120, None, None)

        serial = [runtime.submit(request(i % 3, i)).result(120) for i in range(4)]
        parallel = [runtime.submit(request(i % 3, i)) for i in range(4)]
        for baseline, pending in zip(serial, parallel):
            answer = pending.result(120)
            assert answer.reason is wire.FinishReason.STOP
            assert len(answer.option_logits) == 2
            assert all(
                abs(x - y) <= 1e-3 * max(1, abs(x))
                for x, y in zip(answer.option_logits, baseline.option_logits)
            )
        result["checks"].append(
            "four distinct requests and mixed types match serial at the existing 1e-3 bound"
        )

        invalid = request()
        invalid = replace(invalid, frame=replace(invalid.frame, score_tokens=(3, 4)))
        try:
            runtime.submit(invalid).result(120)
        except RequestFailed:
            pass
        else:
            raise AssertionError("non-marker score position was accepted")
        assert runtime.submit(request()).result(120).reason is wire.FinishReason.STOP
        assert process.pid == runtime._process.pid
        result["checks"].append(
            "invalid position fails one request and the same process remains usable"
        )

        pending = runtime.submit(request(long=True))
        status = runtime.status(timeout=2)
        assert json.loads(status.json)["ready"]
        pending.cancel()
        cancelled = pending.result(120)
        assert (
            cancelled.reason is wire.FinishReason.CANCELLED
            and not cancelled.option_logits
        )
        assert runtime.submit(request()).result(120).reason is wire.FinishReason.STOP
        result["checks"].append(
            "status during work, cancellation, no leaked scores and subsequent request"
        )
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
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
