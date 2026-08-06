"""
CER/WER/빈도그룹 지표 단위테스트 (표준 unittest 사용 — 추가 설치 없음).
실행: <venv>/bin/python -m unittest test_metrics -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metrics import levenshtein, cer, wer, exact_match, freq_group


class TestLevenshtein(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(levenshtein('', ''), 0)
        self.assertEqual(levenshtein('', 'abc'), 3)
        self.assertEqual(levenshtein('abc', ''), 3)

    def test_basic(self):
        self.assertEqual(levenshtein('kitten', 'sitting'), 3)
        self.assertEqual(levenshtein('femotol', 'femotoll'), 1)
        self.assertEqual(levenshtein('Tablet', 'Tablet'), 0)
        self.assertEqual(levenshtein('Dexter', 'Dextr'), 1)

    def test_word_list(self):
        # 단어 리스트에도 동작 (WER용)
        self.assertEqual(levenshtein(['Naprox', 'plus'], ['Naprox']), 1)
        self.assertEqual(levenshtein(['a', 'b', 'c'], ['a', 'x', 'c']), 1)


class TestCER(unittest.TestCase):
    def test_cer_values(self):
        self.assertEqual(cer('femotol', 'femotoll'), 1 / 7)
        self.assertEqual(cer('', 'abc'), 1.0)
        self.assertEqual(cer('', ''), 0.0)
        self.assertEqual(cer('Dexter', 'Dextr'), 1 / 6)
        self.assertEqual(cer('Tablet', 'Tablet'), 0.0)


class TestWER(unittest.TestCase):
    def test_wer_values(self):
        self.assertEqual(wer('Naprox plus', 'Naprox plus'), 0.0)
        self.assertEqual(wer('Naprox plus', 'Naprox'), 0.5)
        self.assertEqual(wer('', 'x'), 1.0)
        self.assertEqual(wer('', ''), 0.0)


class TestExactMatch(unittest.TestCase):
    def test_match(self):
        self.assertTrue(exact_match('Tablet', 'Tablet'))
        self.assertFalse(exact_match('Tablet', 'Tablett'))
        self.assertTrue(exact_match('', ''))


class TestFreqGroup(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(freq_group(146), 'high')
        self.assertEqual(freq_group(10), 'high')
        self.assertEqual(freq_group(9), 'mid')
        self.assertEqual(freq_group(2), 'mid')
        self.assertEqual(freq_group(1), 'low')
        self.assertEqual(freq_group(0), 'low')


if __name__ == '__main__':
    unittest.main()
