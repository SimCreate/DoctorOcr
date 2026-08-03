"""
빈도그룹별 예측 집계 단위테스트 (표준 unittest).
실행: <venv>/bin/python -m unittest test_aggregate -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aggregate import group_predictions, summarize


class TestGroupPredictions(unittest.TestCase):
    def test_grouping(self):
        label_counts = {'Tablet': 146, 'Napa': 114, 'Dextr': 2, 'x': 1}
        preds = [
            {'true': 'Tablet', 'pred': 'Tablet', 'label': 'Tablet'},  # high, correct
            {'true': 'Napa',   'pred': 'Napa',   'label': 'Napa'},    # high, correct
            {'true': 'Dextr',  'pred': 'Dextt',  'label': 'Dextr'},   # mid, wrong
            {'true': 'x',      'pred': 'y',      'label': 'x'},       # low, wrong
        ]
        acc = group_predictions(preds, label_counts)
        self.assertEqual(acc['high']['total'], 2)
        self.assertEqual(acc['high']['correct'], 2)
        self.assertEqual(acc['mid']['total'], 1)
        self.assertEqual(acc['mid']['correct'], 0)
        self.assertEqual(acc['low']['total'], 1)
        self.assertEqual(acc['low']['correct'], 0)

    def test_missing_label_defaults_low(self):
        # label_counts에 없는 라벨은 low로 처리 (안전 기본값)
        preds = [{'true': 'zzz', 'pred': 'zzx', 'label': 'zzz'}]
        acc = group_predictions(preds, {})
        self.assertEqual(acc['low']['total'], 1)

    def test_cer_sum(self):
        # 'femotol'->'femotoll': cer = 1/7
        preds = [{'true': 'femotol', 'pred': 'femotoll', 'label': 'femotol'}]
        acc = group_predictions(preds, {'femotol': 1})
        self.assertAlmostEqual(acc['low']['cer_sum'], 1 / 7)


class TestSummarize(unittest.TestCase):
    def test_summary(self):
        label_counts = {'Tablet': 146}
        preds = [{'true': 'Tablet', 'pred': 'Tablet', 'label': 'Tablet'}]
        s = summarize(group_predictions(preds, label_counts))
        self.assertEqual(s['high']['total'], 1)
        self.assertEqual(s['high']['correct'], 1)
        self.assertEqual(s['high']['acc'], 1.0)
        self.assertEqual(s['high']['avg_cer'], 0.0)

    def test_empty_group(self):
        # 샘플 없는 그룹은 acc/avg_cer None
        s = summarize({'high': {'total': 0, 'correct': 0, 'cer_sum': 0.0},
                       'mid': {'total': 0, 'correct': 0, 'cer_sum': 0.0},
                       'low': {'total': 0, 'correct': 0, 'cer_sum': 0.0}})
        self.assertIsNone(s['high']['acc'])
        self.assertIsNone(s['high']['avg_cer'])


if __name__ == '__main__':
    unittest.main()
