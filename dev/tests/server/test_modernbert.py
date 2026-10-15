import unittest
from unittest.mock import Mock

from server import judgments, protocol
from server.modernbert import ModernBertScorer


class ModernBertScorerTests(unittest.TestCase):
    def scorer(self):
        tokenizer = Mock()
        tokenizer.convert_ids_to_tokens.return_value = "[MASK]"
        tokenizer.encode.return_value = [4]
        return ModernBertScorer(
            tokenizer,
            {
                "type": "laya",
                "head_size": 1,
                "temperatures": {"choice": 2.0, "choice.2": 4.0, "score.3_5": 1.25},
                "max_head_tokens": 192,
                "mask_token_id": 4,
                "sep_token_id": 1,
                "chat_template": "[CLS]{{ type }}[SEP]{% for o in options %}[MASK]{{ o.key }}{% endfor %}[SEP]{{ state }}[SEP]",
            },
        )

    def test_per_type_and_option_count_temperature(self):
        scorer = self.scorer()
        self.assertEqual(
            scorer.temperature(Mock(kind="choice", labels=("a", "b"))), 4.0
        )
        self.assertEqual(
            scorer.temperature(Mock(kind="choice", labels=tuple("abcdef"))), 2.0
        )
        self.assertEqual(
            scorer.temperature(Mock(kind="score", labels=("0", "1", "2"))), 1.25
        )
        self.assertEqual(
            scorer.temperature(Mock(kind="noul", labels=("false", "true"))), 1.0
        )

    def test_order_and_type_are_not_inferred_from_token_ids(self):
        scorer = self.scorer()
        spec = Mock(kind="choice", labels=("shipping", "billing"))
        self.assertEqual(scorer.order(spec), (0, 1))
        self.assertEqual(scorer.flags(spec), protocol.RequestFlag.DECISION_SCORING)
        self.assertEqual(
            scorer.flags(Mock(kind="score")),
            protocol.RequestFlag.DECISION_SCORING
            | protocol.RequestFlag.DECISION_SCORE_TYPE,
        )
        self.assertEqual(
            scorer.flags(Mock(kind="noul")),
            protocol.RequestFlag.DECISION_SCORING
            | protocol.RequestFlag.DECISION_NOUL_TYPE,
        )

    def test_option_budget_preserves_markers_and_state_tail(self):
        scorer = self.scorer()
        _, specs, _ = judgments.validate_systemone(
            {
                "state": "document",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "choose",
                        "criteria": {"a": "first", "b": "second"},
                    }
                },
            }
        )
        scorer.tokenizer.encode.return_value = [
            2,
            *([8] * 160),
            1,
            4,
            *([9] * 80),
            4,
            *([10] * 80),
            1,
            11,
            1,
        ]
        tokens, positions = scorer.encode("document", specs[0][1])
        self.assertEqual(len(positions), 2)
        self.assertTrue(all(tokens[p] == 4 for p in positions))
        self.assertEqual(tokens[-3:], [1, 11, 1])
        self.assertLessEqual(positions[-1] + 49, 195)

    def test_isolated_surrogates_use_the_upstream_text_normalization(self):
        scorer = self.scorer()
        scorer.template = scorer.template.environment.from_string(
            "[CLS]{{ instructions }}{{ type }}[SEP]{% for o in options %}[MASK]{{ o.key }}{% endfor %}[SEP]{{ state }}[SEP]"
        )
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

        def strict_encode(text, **kwargs):
            text.encode("utf-8")
            return [2, 1, 4, 8, 4, 9, 1, 12, 1]

        scorer.tokenizer.encode.side_effect = strict_encode
        scorer.encode(body["state"], specs[0][1])
        prompt = scorer.tokenizer.encode.call_args.args[0]
        self.assertIn("doc\ufffd", prompt)
        self.assertIn("choose\ufffd", prompt)

    def test_missing_marker_is_a_failure(self):
        scorer = self.scorer()
        _, specs, _ = judgments.validate_systemone(
            {
                "state": "doc",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "choose",
                        "criteria": {"a": None, "b": None},
                    }
                },
            }
        )
        scorer.tokenizer.encode.return_value = [2, 8, 1, 4, 9, 1, 10, 1]
        with self.assertRaisesRegex(ValueError, "layout"):
            scorer.encode("doc", specs[0][1])
