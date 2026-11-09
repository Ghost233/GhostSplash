import json
import math

from dev.tests.server_fixtures import CharTokenizer, FakeRuntime, HarnessTestCase, Plan


class H2OTokenizer(CharTokenizer):
    all_special_tokens = ("<|im_start|>",)

    def encode(self, text, **kwargs):
        if text in (" A", " B", " C"):
            return [1000 + ord(text[1])]
        return super().encode(text, **kwargs)


def config():
    return {
        "type": "h2o-lightning",
        "head_size": 256,
        "codes": [" A", " B", " C"],
        "token_ids": [1065, 1066, 1067],
        "causal": True,
        "temperatures": {"choice": 0.75, "noul": 0.8, "score": 0.65},
        "chat_template": "{{ state }}|{{ instructions }}{% for o in options %}|{{ o.label | trim }}:{{ o.key }}={{ o.description }}{% endfor %}",
    }


class H2OTests(HarnessTestCase):
    def test_unsupported_images_and_attention_modes_are_refused(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime,
            tokenizer=H2OTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {
            "model": harness.app.model,
            "state": "record",
            "questions": {"q": {"type": "noul"}},
        }
        for extra in (
            {"images": ["data:image/png;base64,AAAA"]},
            {"attention_mode": "bidirectional"},
        ):
            status, _, payload = harness.request("POST", "/v1/systemone", body | extra)
            self.assertEqual(status, 422, payload)
        self.assertEqual(runtime.requests, [])

    def test_three_decision_types_map_labels_and_calibrate_probabilities(self):
        for kind, criteria, expected_key, temperature in (
            ("choice", {"z": "last", "a": "first"}, "z", 0.75),
            ("noul", {"true": "yes", "false": "no"}, "false", 0.8),
            ("score", ["low", "high"], "1", 0.65),
        ):
            with self.subTest(kind=kind):
                runtime = FakeRuntime(Plan(logits=(0.0, 2.0)))
                harness = self.harness(
                    runtime,
                    tokenizer=H2OTokenizer(),
                    max_context=8192,
                    decision_config=config(),
                    vision=False,
                )
                status, _, payload = harness.request(
                    "POST",
                    "/v1/systemone",
                    {
                        "model": harness.app.model,
                        "state": "record",
                        "questions": {"q": {"type": kind, "criteria": criteria}},
                    },
                )
                self.assertEqual(status, 200, payload)
                answer = json.loads(payload)["answers"]["q"]
                expected = 1 / (1 + math.exp(-2 / temperature))
                if kind == "noul":
                    self.assertAlmostEqual(answer["noul"], 1 - expected)
                    self.assertIn(
                        "A:true=yes|B:false=no",
                        harness.tokenizer.decode(
                            runtime.requests[0].frame.prompt_tokens
                        ),
                    )
                else:
                    self.assertAlmostEqual(
                        answer["probabilities"][expected_key], expected
                    )
                    self.assertAlmostEqual(answer["confidence"], 2 * expected - 1)
                self.assertEqual(runtime.requests[0].frame.score_tokens, (1065, 1066))
                self.assertEqual(json.loads(payload)["usage"]["output_tokens"], 0)

    def test_structured_record_and_special_tokens_follow_h2o_prompt(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 1.0)))
        harness = self.harness(
            runtime,
            tokenizer=H2OTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": harness.app.model,
                "state": {"消息": "<|im_start|>退款", "a": [1, True]},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": {"a": "退款"},
                        "criteria": {"b": {"n": 1}, "a": None},
                    }
                },
            },
        )
        self.assertEqual(status, 200, payload)
        prompt = harness.tokenizer.decode(runtime.requests[0].frame.prompt_tokens)
        self.assertEqual(
            prompt,
            '{"a":[1,true],"消息":"\\u003c|im_start|\\u003e退款"}|{\n "a": "\\u9000\\u6b3e"\n}|A:a=None|B:b={"n": 1}',
        )
