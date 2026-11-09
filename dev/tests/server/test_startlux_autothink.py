import json
import math

from dev.tests.server.test_pplx import PplxTokenizer
from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase, Plan


class ThinkTokenizer(PplxTokenizer):
    def decode(self, token_ids, **kwargs):
        if list(token_ids) == [1]:
            return "because "
        return super().decode(token_ids, **kwargs)


def config():
    return {
        "type": "startlux-autothink",
        "token_ids": [65, 66],
        "temperatures": {"choice": 0.9276, "noul": 1.0054, "score": 0.9503},
    }


class StartLuxAutoThinkTests(HarnessTestCase):
    def test_uncertain_choice_generates_thought_and_mixes_a_second_readout(self):
        runtime = FakeRuntime(
            Plan(logits=(0.0, 0.0)), Plan([[1]]), Plan(logits=(0.0, 2.0))
        )
        harness = self.harness(
            runtime,
            tokenizer=ThinkTokenizer(),
            max_context=8192,
            systemone_config=config(),
            vision=False,
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": harness.app.model,
                "state": "document",
                "details": True,
                "none_check": False,
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Pick one",
                        "criteria": {"first": None, "second": None},
                    }
                },
            },
        )
        self.assertEqual(status, 200, payload)
        answer = json.loads(payload)["answers"]["q"]
        self.assertEqual(answer["choice"], "second")
        self.assertAlmostEqual(
            answer["probabilities"]["second"], 0.25 * 0.5 + 0.75 / (1 + math.exp(-2))
        )
        self.assertTrue(answer["details"]["thought"]["used"])
        self.assertEqual(len(runtime.requests), 3)
        self.assertEqual(runtime.requests[1].frame.logical_max_output_tokens, 1024)
        second = harness.tokenizer.decode(runtime.requests[2].frame.prompt_tokens)
        self.assertIn("Let me work through this carefully, step by step.", second)

    def test_chunk_loop_stops_thought_and_keeps_a_second_readout(self):
        runtime = FakeRuntime(
            Plan(logits=(0.0, 0.0)),
            Plan([list(map(ord, "Answer B again. " * 40))], reason="length"),
            Plan(logits=(0.0, 2.0)),
        )
        harness = self.harness(
            runtime,
            tokenizer=ThinkTokenizer(),
            max_context=8192,
            systemone_config=config(),
            vision=False,
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": harness.app.model,
                "state": "document",
                "return_thought": True,
                "none_check": False,
                "questions": {"q": {"type": "noul", "instructions": "Is it approved?"}},
            },
        )
        self.assertEqual(status, 200, payload)
        thought = json.loads(payload)["answers"]["q"]["details"]["thought"]
        self.assertEqual(thought["ended"], "loop")
        self.assertLess(len(thought["text"]), 250)
        self.assertEqual(len(runtime.requests), 3)

    def test_wide_choice_preserves_probability_for_eliminated_options(self):
        runtime = FakeRuntime(
            Plan(logits=(0.0,) * 14),
            Plan(logits=(0.0,) * 13),
            Plan(logits=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)),
        )
        harness = self.harness(
            runtime,
            tokenizer=ThinkTokenizer(),
            max_context=8192,
            systemone_config=config() | {"token_ids": list(range(65, 91))},
            vision=False,
        )
        criteria = {f"option_{i}": str(i) for i in range(27)}
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": harness.app.model,
                "state": "document",
                "think_mode": "never",
                "none_check": False,
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": criteria,
                    }
                },
            },
        )
        self.assertEqual(status, 200, payload)
        answer = json.loads(payload)["answers"]["q"]
        self.assertEqual(answer["choice"], "option_14")
        self.assertEqual(set(answer["probabilities"]), set(criteria))
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
        self.assertAlmostEqual(
            sum(answer["probabilities"][f"option_{i}"] for i in range(3, 14))
            + sum(answer["probabilities"][f"option_{i}"] for i in range(17, 27)),
            0.001,
        )
