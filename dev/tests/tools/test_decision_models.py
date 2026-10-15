import copy
import unittest

from dev.benchmarks.decision_models import check_response, percentile


class DecisionBenchmarkValidationTests(unittest.TestCase):
    def test_percentile_interpolates_small_sample_sets(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 0.95), 4.8)

    def test_valid_decision_has_exact_usage_and_normalized_distribution(self):
        check_response(self.response(), 128, self.body())

    @staticmethod
    def body():
        return {
            "questions": {
                "team": {"type": "choice", "criteria": {"a": "one", "b": "two"}}
            }
        }

    @staticmethod
    def response():
        return {
            "usage": {"input_tokens": 128, "output_tokens": 0},
            "answers": {
                "team": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "confidence": 0.4,
                }
            },
        }

    def test_invalid_or_incomplete_results_never_count_as_samples(self):
        changes = [
            lambda x: x["usage"].update(output_tokens=1),
            lambda x: x["usage"].update(input_tokens=127),
            lambda x: x.update(answers={}),
            lambda x: x["answers"]["team"].update(type="score"),
            lambda x: x["answers"]["team"].update(choice="missing"),
            lambda x: x["answers"]["team"]["probabilities"].update(a=float("nan")),
            lambda x: x["answers"]["team"]["probabilities"].update(a=-0.1, b=1.1),
            lambda x: x["answers"]["team"]["probabilities"].update(a=0.5, b=0.1),
        ]
        for change in changes:
            bad = copy.deepcopy(self.response())
            change(bad)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                check_response(bad, 128, self.body())
