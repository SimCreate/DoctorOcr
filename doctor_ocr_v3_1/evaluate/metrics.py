"""
평가 지표: CER(문자 오류율), WER, exact_match, 빈도그룹 분류.
표준 라이브러리만 사용 (추가 의존성 없음).
"""
from typing import Sequence, TypeVar

T = TypeVar('T')


def levenshtein(a: Sequence[T], b: Sequence[T]) -> int:
    """Levenshtein 편집거리 (표준 동적프로그래밍 구현).

    한 시퀀스를 다른 시퀀스로 바꾸는 데 필요한 최소 편집 횟수
    (삽입/삭제/대체 각 1회). str뿐 아니라 리스트(단어 단위)에도 동작.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def cer(true: str, pred: str) -> float:
    """문자 오류율 (Character Error Rate).

    틀린 글자 수 ÷ 정답 글자 수.
    0.0 = 완벽, 1.0 = 전혀 다름.
    정답이 비어있으면: 예측도 비면 0.0, 아니면 1.0.
    """
    if not true:
        return 1.0 if pred else 0.0
    return levenshtein(true, pred) / len(true)


def wer(true: str, pred: str) -> float:
    """단어 오류율 (Word Error Rate). 단어 단위 편집거리 / 참조 단어 수."""
    t_words = true.split()
    p_words = pred.split()
    if not t_words:
        return 1.0 if p_words else 0.0
    return levenshtein(t_words, p_words) / len(t_words)


def exact_match(true: str, pred: str) -> bool:
    """엄밀 일치: 정확히 같으면 True."""
    return true == pred


def freq_group(count: int) -> str:
    """라벨 빈도 그룹.

    count >= 10  -> 'high'  (자주 보는 단어)
    count  2~9   -> 'mid'   (가끔 보는 단어)
    count  1     -> 'low'   (한 번만 본 단어)
    """
    if count >= 10:
        return 'high'
    elif count >= 2:
        return 'mid'
    else:
        return 'low'
