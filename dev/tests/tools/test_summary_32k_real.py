import unittest

from dev.tests.summary_32k_real import FACTS, missing_summary_facts


class SummaryFactTests(unittest.TestCase):
    def summary(self):
        return "\n".join("、".join(facts) for facts in FACTS.values())

    def test_accepts_equivalent_number_and_date_spacing(self):
        text = (
            self.summary()
            .replace("480万元", "４８０ 万元")
            .replace("2027年3月18日", "2027 年 3 月 18 日")
        )
        self.assertEqual(missing_summary_facts(text), {})

    def test_rejects_changed_amount_without_matching_its_suffix(self):
        text = self.summary().replace("480万元", "1480万元")
        self.assertEqual(missing_summary_facts(text), {"start": ["480万元"]})

    def test_rejects_wrong_owner_count_and_date(self):
        text = (
            self.summary()
            .replace("周衡", "周恒")
            .replace("19项", "18项")
            .replace("2027年6月9日", "2027年6月10日")
        )
        self.assertEqual(
            missing_summary_facts(text),
            {"middle": ["2027年6月9日", "周衡"], "end": ["19项"]},
        )
