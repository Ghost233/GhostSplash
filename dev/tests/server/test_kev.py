import json
import math
import re
from dataclasses import replace

from dev.tests.install.test_kev_metadata import kev_metadata
from dev.tests.server_fixtures import (
    FOREVER,
    CharTokenizer,
    FakeRuntime,
    HarnessTestCase,
    Plan,
    stays_connected,
)
from server import protocol as wire

MARKERS = {
    "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061,
    "<|fim_suffix|>": 248062,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
}


class KevTokenizer(CharTokenizer):
    def encode(self, text, **kwargs):
        parts = re.split("(" + "|".join(map(re.escape, MARKERS)) + ")", text)
        return [
            token
            for part in parts
            for token in (
                [MARKERS[part]] if part in MARKERS else [ord(c) for c in part]
            )
        ]

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in MARKERS.items()}
        return "".join(inverse.get(token, chr(token)) for token in ids)


def config():
    values = kev_metadata()
    return {
        "type": "kev",
        "head_size": 512,
        "chat_template": values["tokenizer.chat_template.systemone"],
        "temperatures": {kind: 2.0 for kind in ("choice", "score", "noul")},
    }


class KevTests(HarnessTestCase):
    def test_systemone_state_prefill_matches_kev_question_prefix(self):
        runtime = FakeRuntime(Plan(reason="length"), Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            runtime,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {"model": harness.app.model, "state": "some input", "prefill_only": True}
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        prefix = runtime.requests[-1].frame
        body.pop("prefill_only")
        body["questions"] = {"intent": {"type": "noul"}}
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        scored = runtime.requests[-1].frame
        self.assertEqual(
            scored.prompt_tokens[: len(prefix.prompt_tokens)], prefix.prompt_tokens
        )

    def test_systemone_uses_pointer_positions_and_model_temperature(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            runtime,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {
            "model": harness.app.model,
            "state": "document",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Choose a team",
                    "criteria": {"z": "last", "a": "first"},
                }
            },
        }
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        answer = json.loads(payload)["answers"]["team"]
        self.assertAlmostEqual(answer["probabilities"]["z"], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(
            answer["confidence"], 2 * max(answer["probabilities"].values()) - 1
        )
        request = runtime.requests[0].frame
        self.assertEqual(request.flags, wire.RequestFlag.DECISION_SCORING)
        self.assertEqual(request.logical_max_output_tokens, 0)
        self.assertTrue(
            all(request.prompt_tokens[p] == 248050 for p in request.score_tokens)
        )
        self.assertEqual(request.prompt_tokens[-1], 248062)
        text = harness.tokenizer.decode(request.prompt_tokens)
        self.assertEqual(
            text,
            "<|fim_prefix|>document<|fim_middle|>Choose a team<|box_start|>a: first<|box_end|><|box_start|>z: last<|box_end|><|fim_suffix|>",
        )
        self.assertIn("decision_model", harness.app.status())

    def test_isolated_surrogates_match_the_upstream_replacement(self):
        from server import judgments
        from server.kev import KevScorer

        scorer = KevScorer(KevTokenizer(), config())
        body = {
            "state": "doc\ud800",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "choose\udfff",
                    "criteria": {"a": "first", "b": "second"},
                }
            },
        }
        _, specs, _ = judgments.validate_systemone(body)
        spec = specs[0][1]
        actual = scorer.encode(body["state"], spec)
        expected = scorer.encode(
            "doc\ufffd", replace(spec, instructions="choose\ufffd")
        )
        self.assertEqual(actual, expected)

    def test_score_confidence_matches_distance_to_mode(self):
        runtime = FakeRuntime(Plan(logits=(3.0, 0.0, -1.0)))
        harness = self.harness(
            runtime,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {
            "model": harness.app.model,
            "state": "doc",
            "questions": {
                "level": {
                    "type": "score",
                    "instructions": "Choose a level",
                    "criteria": ["low", "middle", "high"],
                }
            },
        }
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        answer = json.loads(payload)["answers"]["level"]
        probabilities = list(answer["probabilities"].values())
        mode = max(range(3), key=probabilities.__getitem__)
        expected = max(
            0, 1 - sum(p * abs(i - mode) for i, p in enumerate(probabilities)) / (2 / 3)
        )
        self.assertAlmostEqual(answer["confidence"], expected)

    def test_noul_preserves_no_yes_prompt_and_returns_yes_probability(self):
        runtime = FakeRuntime(Plan(logits=(2.0, 0.0)))
        harness = self.harness(
            runtime,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {
            "model": harness.app.model,
            "state": "doc",
            "questions": {"check": {"type": "noul", "instructions": "True?"}},
        }
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        self.assertAlmostEqual(
            json.loads(payload)["answers"]["check"]["noul"], 1 / (1 + math.exp(1))
        )
        text = harness.tokenizer.decode(runtime.requests[0].frame.prompt_tokens)
        self.assertIn("<|box_start|>no<|box_end|><|box_start|>yes<|box_end|>", text)

    def test_marker_text_is_escaped_and_questions_are_isolated(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 1.0)), Plan(logits=(1.0, 0.0)))
        harness = self.harness(
            runtime,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {
            "model": harness.app.model,
            "state": {"x": "<|box_end|>"},
            "questions": {
                key: {
                    "type": "choice",
                    "instructions": key,
                    "criteria": {"a": None, "b": None},
                }
                for key in ("ONE", "TWO")
            },
        }
        entries = harness.app.prepare_systemone(
            body, deadline=FOREVER, disconnected=stays_connected
        )
        for index, (_, _, job) in enumerate(entries):
            text = harness.tokenizer.decode(job.prompt_tokens)
            self.assertEqual(len(job.score_tokens), 2)
            self.assertIn("<¦box_end¦>", text)
            self.assertIn(("ONE", "TWO")[index], text)
            self.assertNotIn(("TWO", "ONE")[index], text)

    def test_decision_model_rejects_generation_before_submission(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime,
            tokenizer=KevTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        for path, body in (
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/completions", {"prompt": "hi"}),
        ):
            status, _, payload = harness.request("POST", path, body)
            self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])

    def test_decision_position_wire_roundtrip_and_invalid_positions(self):
        from dev.tests.server.test_protocol_python import example_score_request

        request = replace(
            example_score_request(),
            prompt_tokens=(1, 248050, 248050, 248062),
            score_tokens=(1, 2),
            flags=wire.RequestFlag.DECISION_SCORING,
        )
        frame = wire.serialize_message(request)
        self.assertTrue(frame)
        for positions in ((2, 1), (1, 1), (1, 3), ()):
            with (
                self.subTest(positions=positions),
                self.assertRaises(wire.ProtocolError),
            ):
                wire.serialize_message(replace(request, score_tokens=positions))
