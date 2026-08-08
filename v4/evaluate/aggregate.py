"""
예측 결과를 빈도그룹별(고/중/저)로 집계.
predictions: [{'true': str, 'pred': str, 'label': str}, ...]
"""
from collections import defaultdict

from metrics import cer, exact_match, freq_group, beam_oracle

GROUPS = ('high', 'mid', 'low')


def group_predictions(predictions, label_counts):
    """predictions를 빈도그룹별로 집계.

    label_counts: {label: 전체 빈도} — 각 예측의 label이 어느 그룹인지 결정.
    반환: {group: {'total': int, 'correct': int, 'cer_sum': float}}
    """
    acc = {g: {'total': 0, 'correct': 0, 'cer_sum': 0.0} for g in GROUPS}
    for p in predictions:
        # label_counts에 없으면 1로 취급 (low) — 안전 기본값
        g = freq_group(label_counts.get(p['label'], 1))
        acc[g]['total'] += 1
        if exact_match(p['true'], p['pred']):
            acc[g]['correct'] += 1
        acc[g]['cer_sum'] += cer(p['true'], p['pred'])
    return acc


def summarize(acc):
    """집계 dict -> 요약 dict {group: {total, correct, acc, avg_cer}}.

    acc: group_predictions()의 반환값.
    """
    out = {}
    for g in GROUPS:
        d = acc[g]
        out[g] = {
            'total': d['total'],
            'correct': d['correct'],
            'acc': d['correct'] / d['total'] if d['total'] else None,
            'avg_cer': d['cer_sum'] / d['total'] if d['total'] else None,
        }
    return out


# ============================================================
# v3_1 전용 확장: beam oracle 집계
# ============================================================

def oracle_group_predictions(predictions, label_counts):
    """빈도그룹별 beam oracle 정확도 집계.

    predictions: [{'true','label', 'candidates': [str,...]}, ...]
      - candidates = beam search top-k 후보 문자열 리스트
      - oracle = candidates에 정답이 포함되면 correct
    반환: summarize()와 동일 형태 {group: {total, correct, acc, avg_cer}}
    """
    acc = {g: {'total': 0, 'correct': 0, 'cer_sum': 0.0} for g in GROUPS}
    for p in predictions:
        g = freq_group(label_counts.get(p['label'], 1))
        acc[g]['total'] += 1
        if beam_oracle(p['true'], p.get('candidates', [])):
            acc[g]['correct'] += 1
        acc[g]['cer_sum'] += cer(p['true'], p.get('pred', ''))
    return acc
