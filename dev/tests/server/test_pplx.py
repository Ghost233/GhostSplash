import json
import math

from dev.tests.server_fixtures import CharTokenizer, FakeRuntime, HarnessTestCase, Plan


class PplxTokenizer(CharTokenizer):
    marker = "<|fim_suffix|>"

    def encode(self, text, **kwargs):
        return super().encode(text.replace(self.marker, "Ϩ"), **kwargs)

    def decode(self, token_ids, **kwargs):
        return super().decode(token_ids, **kwargs).replace("Ϩ", self.marker)

    def apply_chat_template(self, messages, **kwargs):
        rendered = "".join(
            f"{message['role']}:{message['content'][0]['text'] if isinstance(message['content'], list) else message['content']}\n"
            for message in messages
        )
        rendered += "assistant:<think>\n\n</think>\n\n"
        return rendered if kwargs.get("tokenize") is False else self.encode(rendered)


def config():
    return {
        "type": "pplx-decider-v3",
        "head_size": 256,
        "codes": ["A", "B"],
        "token_ids": [65, 66],
        "temperature": 2.0,
        "marker": "<|fim_suffix|>",
        "escape_tokens": ["<|fim_suffix|>"],
        "max_pixels": 1638400,
    }


class PplxApiTests(HarnessTestCase):
    def test_choice_is_read_at_the_inline_marker_with_calibration(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            runtime,
            tokenizer=PplxTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": harness.app.model,
                "state": "an example <|fim_suffix|>",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Which?",
                        "criteria": {"z": "first", "a": "second"},
                    }
                },
            },
        )
        self.assertEqual(status, 200, payload)
        answer = json.loads(payload)["answers"]["q"]
        expected = 1 / (1 + math.exp(-1))
        self.assertEqual(answer["choice"], "a")
        self.assertAlmostEqual(answer["probabilities"]["a"], expected)
        frame = runtime.requests[0].frame
        prompt = harness.tokenizer.decode(frame.prompt_tokens)
        self.assertIn("an example ＜|fim_suffix|＞", prompt)
        self.assertIn(
            "Question 0:\nWhich?\nOptions:\nA: z: first\nB: a: second\n<|fim_suffix|>",
            prompt,
        )
        self.assertEqual(frame.score_tokens, (65, 66))
        self.assertEqual(frame.generation_prompt_tokens, 0)
        self.assertEqual(frame.readout_positions, (frame.prompt_tokens.index(1000),))

    def test_questions_share_the_full_joint_prompt_and_have_separate_readouts(self):
        runtime = FakeRuntime(Plan(logits=(2.0, 0.0, 0.0, 2.0)))
        harness = self.harness(
            runtime,
            tokenizer=PplxTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": harness.app.model,
                "state": "document",
                "questions": {
                    "one": {
                        "type": "choice",
                        "criteria": {"first": None, "second": None},
                    },
                    "two": {"type": "noul"},
                },
            },
        )
        self.assertEqual(status, 200, payload)
        body = json.loads(payload)
        answers = body["answers"]
        self.assertEqual(answers["one"]["choice"], "first")
        self.assertAlmostEqual(answers["two"]["noul"], 1 / (1 + math.exp(-1)))
        self.assertEqual(len(runtime.requests), 1)
        frame = runtime.requests[0].frame
        prompt = harness.tokenizer.decode(frame.prompt_tokens)
        self.assertIn("Question 0:", prompt)
        self.assertIn("Question 1:", prompt)
        markers = tuple(
            i for i, token in enumerate(frame.prompt_tokens) if token == 1000
        )
        self.assertEqual(len(markers), 2)
        self.assertEqual(frame.readout_positions, markers)
        self.assertEqual(body["usage"]["input_tokens"], len(frame.prompt_tokens))
