"""
수용기준 판정.
규칙: 고빈도(10회↑) exact match >= 90% AND 전체 CER <= 20% 이면 PASS.
"""
from metrics import cer


def overall_cer(predictions):
    """전체 CER (고/중/저 합산, 샘플 단위 평균).

    predictions: [{'true','pred',...}, ...]
    """
    if not predictions:
        return None
    return sum(cer(p['true'], p['pred']) for p in predictions) / len(predictions)


def acceptance(high_group_summary, predictions):
    """수용기준 판정.

    high_group_summary: summarize() 결과의 'high' dict (acc 포함).
    반환: (passed: bool, details: dict)
    """
    high_acc = high_group_summary.get('acc') if high_group_summary else None
    ov_cer = overall_cer(predictions)
    high_ge_90 = high_acc is not None and high_acc >= 0.90
    cer_le_20 = ov_cer is not None and ov_cer <= 0.20
    passed = high_ge_90 and cer_le_20
    return passed, {
        'high_acc': high_acc,
        'overall_cer': ov_cer,
        'high_ge_90': high_ge_90,
        'cer_le_20': cer_le_20,
    }


def acceptance_report(passed, details):
    """사람이 읽을 판정 문자열."""
    high_acc = details['high_acc']
    ov_cer = details['overall_cer']
    lines = ["=== 수용기준 판정 ==="]
    if high_acc is None:
        lines.append("고빈도 샘플 없음 — 판정 불가")
    else:
        lines.append(f"고빈도 exact match: {high_acc:.1%} (기준 ≥90%) → "
                     f"{'PASS' if details['high_ge_90'] else 'FAIL'}")
    if ov_cer is None:
        lines.append("평가 샘플 없음 — 판정 불가")
    else:
        lines.append(f"전체 CER: {ov_cer:.1%} (기준 ≤20%) → "
                     f"{'PASS' if details['cer_le_20'] else 'FAIL'}")
    lines.append(f"최종: {'PASS — 실사용 시작 가능' if passed else 'FAIL — 개선 필요'}")
    return "\n".join(lines)
