import copy
import json
import unittest
from pathlib import Path

from dev.benchmarks import expert_ple_calibration as calibration


class ExpertPleCalibrationTests(unittest.TestCase):
    def status(self):
        # This tracked fixture shares the public status schema; configure the
        # few fixed-quota trial fields rather than loading local build output.
        status = json.loads(
            (Path(__file__).parent / "engine/status_golden.json").read_text()
        )
        status["transport"] = {"restarts": 0}
        status["identity"]["kv"].update(
            format="q4h", quantization="symmetric_int4_hadamard", scale_type="float32"
        )
        return status

    def test_abba_does_not_promote_a_small_or_noisy_difference(self):
        for times in ([100, 99.5, 99.5, 100], [100, 90, 100, 110]):
            rows = [
                dict(ple_mib=p, wall_seconds=t, interference=[])
                for p, t in zip((512, 256, 256, 512), times)
            ]
            verdict = calibration.paired_verdict(rows)
            self.assertIsNone(verdict["winner_ple_mib"])
            self.assertTrue(verdict["keep_baseline"])

    def test_abba_finds_a_repeatable_win_and_rejects_interference(self):
        rows = [
            dict(ple_mib=p, wall_seconds=t, interference=[])
            for p, t in zip((512, 256, 256, 512), (100, 90, 90, 100))
        ]
        self.assertEqual(calibration.paired_verdict(rows)["winner_ple_mib"], 256)
        rows[1]["interference"] = [123]
        with self.assertRaisesRegex(ValueError, "uncontaminated"):
            calibration.paired_verdict(rows)

    def test_abba_rejects_unpaired_or_duplicate_quotas(self):
        rows = [dict(ple_mib=512, wall_seconds=100, interference=[])] * 4
        with self.assertRaisesRegex(ValueError, "distinct"):
            calibration.paired_verdict(rows)
        with self.assertRaisesRegex(ValueError, "four ABBA"):
            calibration.paired_verdict(rows[:3])

    def test_quota_cannot_be_silently_clamped(self):
        status = self.status()
        status["memory_plan"]["budget"]["expert_ple_pool"] = dict(
            limit_bytes=calibration.POOL,
            expert_planned_bytes=calibration.POOL - 256 * calibration.MIB,
            ple_planned_bytes=256 * calibration.MIB,
        )
        with self.assertRaisesRegex(ValueError, "quota was changed"):
            calibration.allocation(status, 512 * calibration.MIB)

    def test_exact_output_and_cold_prefix_contract(self):
        response = dict(
            usage=dict(prompt_tokens=32768, completion_tokens=512),
            metrics=dict(cache=dict(matched_tokens=0)),
            choices=[dict(message=dict(role="assistant", content="reference"))],
        )
        after = dict(
            state=dict(active_lanes=0),
            kv=dict(pages_active=0),
            scheduler=dict(queued=0, prefilling=0, decoding=0),
        )
        reference = calibration.validate_response(response, after)
        self.assertEqual(
            calibration.validate_response(response, after, reference), reference
        )
        other = copy.deepcopy(response)
        other["choices"][0]["message"]["content"] = "different"
        with self.assertRaisesRegex(ValueError, "output differs"):
            calibration.validate_response(other, after, reference)
        response["metrics"]["cache"]["matched_tokens"] = 32
        with self.assertRaisesRegex(ValueError, "prefix reuse"):
            calibration.validate_response(response, after)


if __name__ == "__main__":
    unittest.main()
