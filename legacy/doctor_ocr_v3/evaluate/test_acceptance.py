"""
수용기준 판정 단위테스트 (표준 unittest).
실행: <venv>/bin/python -m unittest test_acceptance -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from acceptance import overall_cer, acceptance, acceptance_report


class TestOverallCer(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(overall_cer([]))

    def test_perfect(self):
        preds = [{'true': 'Tablet', 'pred': 'Tablet'},
                 {'true': 'Napa', 'pred': 'Napa'}]
        self.assertEqual(overall_cer(preds), 0.0)

    def test_mixed(self):
        # Tablet(0오류) + femotol->femotoll(1/7) → 평균 (0 + 1/7)/2 = 1/14
        preds = [{'true': 'Tablet', 'pred': 'Tablet'},
                 {'true': 'femotol', 'pred': 'femotoll'}]
        self.assertAlmostEqual(overall_cer(preds), 1 / 14)


class TestAcceptance(unittest.TestCase):
    def test_pass(self):
        # 고빈도 100%, 전체 CER 0% → PASS
        high_summary = {'total': 10, 'correct': 10, 'acc': 1.0, 'avg_cer': 0.0}
        preds = [{'true': 'Tablet', 'pred': 'Tablet'}] * 10
        passed, details = acceptance(high_summary, preds)
        self.assertTrue(passed)
        self.assertTrue(details['high_ge_90'])
        self.assertTrue(details['cer_le_20'])

    def test_fail_high_acc(self):
        # 고빈도 80% → 고빈도 기준 FAIL (전체 CER은 0%)
        high_summary = {'total': 10, 'correct': 8, 'acc': 0.8, 'avg_cer': 0.0}
        preds = [{'true': 'Tablet', 'pred': 'Tablet'}] * 10
        passed, details = acceptance(high_summary, preds)
        self.assertFalse(passed)
        self.assertFalse(details['high_ge_90'])
        self.assertTrue(details['cer_le_20'])

    def test_fail_overall_cer(self):
        # 고빈도 100%지만 전체 CER이 50% → CER 기준 FAIL
        high_summary = {'total': 10, 'correct': 10, 'acc': 1.0, 'avg_cer': 0.0}
        preds = [{'true': 'aaaa', 'pred': 'bbbb'}] * 10  # CER 100%
        passed, details = acceptance(high_summary, preds)
        self.assertFalse(passed)
        self.assertTrue(details['high_ge_90'])
        self.assertFalse(details['cer_le_20'])

    def test_missing_high(self):
        # 고빈도 없음 → 판정 불가 (PASS 아님)
        passed, details = acceptance(None, [{'true': 'a', 'pred': 'a'}])
        self.assertFalse(passed)
        self.assertIsNone(details['high_acc'])


class TestAcceptanceReport(unittest.TestCase):
    def test_report_pass(self):
        passed, details = acceptance({'acc': 0.95, 'total': 10, 'correct': 9},
                                     [{'true': 'a', 'pred': 'a'}] * 10)
        rep = acceptance_report(passed, details)
        self.assertIn("PASS", rep)

    def test_report_fail(self):
        passed, details = acceptance({'acc': 0.5, 'total': 10, 'correct': 5},
                                     [{'true': 'a', 'pred': 'b'}] * 10)
        rep = acceptance_report(passed, details)
        self.assertIn("FAIL", rep)


if __name__ == '__main__':
    unittest.main()
