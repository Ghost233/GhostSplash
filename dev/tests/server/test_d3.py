import json
import math

from dev.tests.server_fixtures import CharTokenizer, FakeRuntime, HarnessTestCase, Plan


def config():
    return {
        "type": "vllm-d3",
        "head_size": 256,
        "codes": ["A", "B", "C"],
        "token_ids": [65, 66, 67],
        "temperatures": dict.fromkeys(("choice", "score", "noul"), 2.0),
        "chat_template": "{{ state }}|{{ instructions }}{% for o in options %}|{{ o.label }}:{{ o.key }}={{ o.description }}{% endfor %}|readout",
    }


class D3Tests(HarnessTestCase):
    def test_state_prefill_is_refused_without_affecting_scoring(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            decision_config=config(),
            vision=False,
        )
        body = {"model": harness.app.model, "state": "document", "prefill_only": True}
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 422, payload)
        self.assertEqual(runtime.requests, [])
        body.pop("prefill_only")
        body["questions"] = {"team": {"type": "noul"}}
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)

    def test_structured_unicode_matches_model_json_template(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 2.0)))
        cfg = config() | {
            "chat_template": "{{ state | tojson }}|{{ instructions | tojson }}"
        }
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            decision_config=cfg,
            vision=False,
        )
        body = {
            "model": harness.app.model,
            "state": {"消息": "请退款 <已付款>", "enabled": True, "a": [1, 2]},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": {"提示": "选择部门"},
                    "criteria": {"billing": "退款", "shipping": "物流"},
                }
            },
        }
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            harness.tokenizer.decode(runtime.requests[0].frame.prompt_tokens),
            '{"消息": "请退款 <已付款>", "enabled": true, "a": [1, 2]}|{"提示": "选择部门"}',
        )

    def test_systemone_code_mapping_and_temperature(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 2.0)))
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
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
                    "instructions": "Choose",
                    "criteria": {"z": "last", "a": "first"},
                }
            },
        }
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        result = json.loads(payload)
        self.assertAlmostEqual(
            result["answers"]["team"]["probabilities"]["z"], 1 / (1 + math.exp(-1))
        )
        request = runtime.requests[0].frame
        self.assertEqual(request.score_tokens, (65, 66))
        self.assertEqual(request.flags, 0)
        self.assertEqual(request.logical_max_output_tokens, 0)
        self.assertEqual(
            harness.tokenizer.decode(request.prompt_tokens),
            "document|Choose|A:a=first|B:z=last|readout",
        )
        self.assertEqual(result["usage"]["output_tokens"], 0)

    def test_invalid_tokenizer_mapping_is_rejected(self):
        from server.d3 import D3Scorer

        with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
            D3Scorer(CharTokenizer(), config() | {"token_ids": [67, 66, 65]})
