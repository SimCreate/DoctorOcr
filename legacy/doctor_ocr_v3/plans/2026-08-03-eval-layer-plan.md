# 평가 레이어 (CER + 빈도그룹) 구현 계획

> **For Hermes:** 이 프로젝트는 서브에이전트 환경 없음 → 에이전트(본인)가 절차형으로 직접 구현.
> TDD 기반, bite-sized task, task마다 커밋.

**Goal:** 기존 v2.2의 exact-match 평가를 **CER(문자 오류율) + 라벨 빈도그룹(고/중/저) 분리**로 확장한 평가 레이어를 만들어 v3의 baseline 재측정 및 실험군 비교의 공통 기준으로 사용하게 한다.

**Architecture:** 기존 `evaluate.py`(v2_2)의 평가 로직(모델 로드, val split, CTC decode)을 재사용하고, 결과 집계 부분만 CER·빈도그룹 지표로 확장한다. 독립 평가 도구를 v3 폴더에 배치하되, v2_2 모델/체크포인트/데이터를 읽는다. 핵심 지표 계산(CER, 빈도그룹)은 순수 함수로 분리해 단위 테스트 가능하게 한다.

**Tech Stack:** Python 3.13, torch 2.13.0+cu132, pandas, numpy, cv2 (기존 v2_2 venv 재사용). FastDTW/rapidfuzz 등 **추가 설치 없음** — CER은 Levenshtein distance 표준 구현 사용.

**환경:**
- venv: `/home/dev/DoctorOcr/doctor_ocr_v3/venv/` (torch 2.13.0+cu132, py3.13.7)
- 모델: `/home/dev/doctor_ocr_v2_2/model/model_v2_2.py` (CTCModel — CNN SEBlock + BiLSTM + CTC Head)
- 데이터: `/home/dev/doctor_ocr_v2_2/dataset/` (심볼릭 링크 → v2 dataset)
- val split: seed 42, 80/20, torch.manual_seed(42) (기존 evaluate.py와 동일)
- GPU: CUDA_VISIBLE_DEVICES='1' (GPU1 Max-Q)

---

## 사전 검증된 사실 (계획 근거)

- 고빈도/중빈도/저빈도: 2,582 / 1,849 / 1,148 샘플, 고유라벨 105 / 535 / 1,148
- 80/20 스플릿(seed42) 시 val에 고빈도 520샘플(102 고유라벨) → **고빈도 105개 중 102개 평가 가능** (실측)
- val 총 1,115샘플 — 기존 v2의 val 1,116과 실질 동일
- 기존 evaluate.py는 exact match만 출력, CER/빈도그룹 없음 (재사용 지점 확인)

---

## Task 1: 평가 레이어 폴더 구조 + 지표 계산 모듈 (순수 함수)

**Objective:** CER(Levenshtein)와 빈도그룹 분류를 순수 함수로 구현해 테스트 가능하게 만든다.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/metrics.py`
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/__init__.py`
- Test: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/test_metrics.py`

**Step 1: metrics.py 작성 (표준 라이브러리만, 추가 의존성 없음)**

```python
"""평가 지표: CER(문자 오류율), WER, 빈도그룹 분류. 표준 라이브러리만 사용."""

def levenshtein(a: str, b: str) -> int:
    """Levenshtein 편집거리 (표준 구현)."""
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
    """문자 오류율: 편집거리 / 참조 길이. 참조가 비면 1.0."""
    if not true:
        return 1.0 if pred else 0.0
    return levenshtein(true, pred) / len(true)


def wer(true: str, pred: str) -> float:
    """단어 오류율: 단어 단위 편집거리 / 참조 단어 수."""
    t_words = true.split()
    p_words = pred.split()
    if not t_words:
        return 1.0 if p_words else 0.0
    return levenshtein(t_words, p_words) / len(t_words)


def exact_match(true: str, pred: str) -> bool:
    return true == pred


def freq_group(count: int) -> str:
    """라벨 빈도 그룹: count>=10 → 'high', 2~9 → 'mid', 1 → 'low'."""
    if count >= 10:
        return 'high'
    elif count >= 2:
        return 'mid'
    else:
        return 'low'
```

**Step 2: test_metrics.py 작성 (TDD — 실패 먼저)**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from metrics import levenshtein, cer, wer, exact_match, freq_group


def test_levenshtein_empty():
    assert levenshtein('', '') == 0
    assert levenshtein('', 'abc') == 3
    assert levenshtein('abc', '') == 3


def test_levenshtein_basic():
    assert levenshtein('kitten', 'sitting') == 3
    assert levenshtein('femotol', 'femotoll') == 1
    assert levenshtein('Tablet', 'Tablet') == 0
    assert levenshtein('Dexter', 'Dextr') == 1


def test_cer():
    assert cer('femotol', 'femotoll') == 1/7
    assert cer('', 'abc') == 1.0
    assert cer('', '') == 0.0
    assert cer('Dexter', 'Dextr') == 1/6


def test_wer():
    assert wer('Naprox plus', 'Naprox plus') == 0.0
    assert wer('Naprox plus', 'Naprox') == 0.5
    assert wer('', 'x') == 1.0


def test_exact_match():
    assert exact_match('Tablet', 'Tablet') is True
    assert exact_match('Tablet', 'Tablett') is False


def test_freq_group():
    assert freq_group(146) == 'high'
    assert freq_group(10) == 'high'
    assert freq_group(9) == 'mid'
    assert freq_group(2) == 'mid'
    assert freq_group(1) == 'low'
```

**Step 3: 테스트 실행 — 실패 확인 (의도적)**

Run: `cd /home/dev/DoctorOcr/doctor_ocr_v3/evaluate && ../venv/bin/python -m pytest test_metrics.py -v`
Expected: FAIL — "ModuleNotFoundError: No module named 'metrics'" (아직 파일 없음)

**Step 4: 실제 구현 파일 생성 후 pass 확인**

위 metrics.py를 실제로 저장하고:
Run: `venv/bin/python -m pytest test_metrics.py -v`
Expected: 6 passed

**Step 5: 커밋**
```bash
cd /home/dev/DoctorOcr
git add doctor_ocr_v3/evaluate/
git commit -m "feat(v3): CER/WER/빈도그룹 지표 순수 함수 + 단위테스트"
```

---

## Task 2: 예측 결과 집계 모듈 (빈도그룹별 통계)

**Objective:** 개별 예측(사전의 리스트)을 받아 빈도그룹별로 exact match / 평균 CER / 샘플수를 집계한다.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/aggregate.py`
- Test: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/test_aggregate.py`

**Step 1: aggregate.py 작성**

```python
"""예측 결과를 빈도그룹별로 집계."""
from collections import defaultdict
from metrics import cer, exact_match, freq_group


def group_predictions(predictions, label_counts):
    """predictions: [{'true':str,'pred':str,'label':str}, ...]
       label_counts: {label: 전체 빈도}
       -> {group: {'total':int,'correct':int,'cer_sum':float}}"""
    acc = {g: {'total': 0, 'correct': 0, 'cer_sum': 0.0}
           for g in ('high', 'mid', 'low')}
    for p in predictions:
        g = freq_group(label_counts.get(p['label'], 1))
        acc[g]['total'] += 1
        if exact_match(p['true'], p['pred']):
            acc[g]['correct'] += 1
        acc[g]['cer_sum'] += cer(p['true'], p['pred'])
    return acc


def summarize(acc):
    """집계 dict -> 요약 dict {group: {total, correct, acc, avg_cer}}"""
    out = {}
    for g, d in acc.items():
        out[g] = {
            'total': d['total'],
            'correct': d['correct'],
            'acc': d['correct'] / d['total'] if d['total'] else None,
            'avg_cer': d['cer_sum'] / d['total'] if d['total'] else None,
        }
    return out
```

**Step 2: test_aggregate.py (TDD)**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from aggregate import group_predictions, summarize


def test_group_predictions_groups():
    label_counts = {'Tablet': 146, 'Napa': 114, 'Dextr': 2, 'x': 1}
    preds = [
        {'true': 'Tablet', 'pred': 'Tablet', 'label': 'Tablet'},   # high, correct
        {'true': 'Napa', 'pred': 'Napa', 'label': 'Napa'},          # high, correct
        {'true': 'Dextr', 'pred': 'Dextt', 'label': 'Dextr'},       # mid, wrong
        {'true': 'x', 'pred': 'y', 'label': 'x'},                   # low, wrong
    ]
    acc = group_predictions(preds, label_counts)
    assert acc['high']['total'] == 2
    assert acc['high']['correct'] == 2
    assert acc['mid']['total'] == 1
    assert acc['mid']['correct'] == 0
    assert acc['low']['total'] == 1


def test_summarize():
    label_counts = {'Tablet': 146}
    preds = [{'true': 'Tablet', 'pred': 'Tablet', 'label': 'Tablet'}]
    s = summarize(group_predictions(preds, label_counts))
    assert s['high']['acc'] == 1.0
    assert s['high']['avg_cer'] == 0.0
```

**Step 3: 테스트 실패 확인 → Step 4: pass 확인 → Step 5: 커밋**
```bash
cd /home/dev/DoctorOcr && git add doctor_ocr_v3/evaluate/ && git commit -m "feat(v3): 빈도그룹별 예측 집계 + 테스트"
```

---

## Task 3: 수용기준 판정 모듈

**Objective:** 집계 결과가 수용기준(고빈도 exact ≥90% AND 전체 CER ≤20%) 충족 여부를 계산한다.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/acceptance.py`

**Step 1: acceptance.py 작성**

```python
"""수용기준 판정: 고빈도 exact≥90% AND 전체 CER≤20%"""
from metrics import cer


def overall_cer(predictions):
    """전체 CER (고/중/저 합산)"""
    if not predictions:
        return None
    return sum(cer(p['true'], p['pred']) for p in predictions) / len(predictions)


def acceptance(high_group_summary, predictions):
    """high_group_summary: summarize()의 'high' dict
       returns (passed: bool, details: dict)"""
    high_acc = high_group_summary.get('acc')
    ov_cer = overall_cer(predictions)
    passed = (high_acc is not None and high_acc >= 0.90 and
              ov_cer is not None and ov_cer <= 0.20)
    return passed, {
        'high_acc': high_acc,
        'overall_cer': ov_cer,
        'high_ge_90': high_acc is not None and high_acc >= 0.90,
        'cer_le_20': ov_cer is not None and ov_cer <= 0.20,
    }


def acceptance_report(passed, details):
    """사람이 읽을 판정 문자열"""
    lines = ["=== 수용기준 판정 ==="]
    lines.append(f"고빈도 exact match: {details['high_acc']:.1%} (기준 ≥90%) → {'PASS' if details['high_ge_90'] else 'FAIL'}")
    lines.append(f"전체 CER: {details['overall_cer']:.1%} (기준 ≤20%) → {'PASS' if details['cer_le_20'] else 'FAIL'}")
    lines.append(f"최종: {'PASS — 실사용 시작 가능' if passed else 'FAIL — 개선 필요'}")
    return "\n".join(lines)
```

**Step 2-4: TDD cycle** (test_acceptance.py — passed case, failed case 2개). 커밋.

---

## Task 4: 평가 메인 (v2_2 통합 실행)

**Objective:** 기존 v2_2 evaluate.py의 모델 로드/val split/CTC decode를 재사용해 전체 val에 대해 예측을 만들고, Task 1-3 모듈로 집계·판정·CSV 저장을 수행하는 메인 스크립트.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/run_eval.py`
- Modify: 기존 v2_2 evaluate.py는 건드리지 않음 (재사용만)

**Step 1: run_eval.py 작성 — 기존 evaluate.py의 로드/스플릿/디코드 로직을 import 재사용**

```python
#!/usr/bin/env python3
"""v3 평가 레이어: CER + 빈도그룹 + 수용기준 판정 (v2_2 모델/체크포인트 기준)"""
import os, sys, pickle
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# v2_2 모델 및 evaluate 로직 재사용
V2_2 = Path("/home/dev/doctor_ocr_v2_2")
sys.path.insert(0, str(V2_2 / "model"))
sys.path.insert(0, str(V2_2))
from model_v2_2 import CRNN, decode_sequence, CTC_BLANK
from evaluate import HandwritingDataset, ctc_collate_fn, evaluate_model  # v2_2의 평가 함수

# v3 지표 모듈
CUR = Path(__file__).parent
sys.path.insert(0, str(CUR))
from metrics import cer
from aggregate import group_predictions, summarize
from acceptance import acceptance, acceptance_report, overall_cer

DATA_ROOT = V2_2 / "dataset"
LABEL_CSV = DATA_ROOT / "combined_labels.csv"
IMG_DIR = DATA_ROOT / "img" / "img"
CHAR_DICT = V2_2 / "working" / "char_dict.pkl"
CKPT = V2_2 / "working" / "checkpoints" / "best_model.pth"
BATCH_SIZE = 8
NUM_WORKERS = 4

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device} (CUDA_VISIBLE_DEVICES=1)")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    vocab_size = ckpt.get('vocab_size')
    with open(CHAR_DICT, 'rb') as f:
        char_dict = pickle.load(f)
    idx2char = char_dict['idx2char']

    model = CRNN(vocab_size=vocab_size,
                 hidden_size=config.get('hidden_size', 256),
                 dropout=config.get('dropout', 0.3),
                 use_gradient_checkpointing=config.get('use_gradient_checkpointing', True)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[MODEL] epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full = HandwritingDataset(LABEL_CSV, IMG_DIR, char_dict['char2idx'], transform, augment=False)
    torch.manual_seed(42)
    train_size = int(0.8 * len(full))
    _, val = torch.utils.data.random_split(full, [train_size, len(full) - train_size])
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=ctc_collate_fn)

    _, _, predictions = evaluate_model(model, val_loader, device, idx2char)

    # 실제 라벨 빈도 계산
    df = pd.read_csv(LABEL_CSV)
    label_counts = df['label'].value_counts().to_dict()

    acc = group_predictions(predictions, label_counts)
    summ = summarize(acc)
    passed, details = acceptance(summ['high'], predictions)

    print("\n=== 빈도그룹별 결과 ===")
    for g in ('high', 'mid', 'low'):
        s = summ[g]
        print(f"  {g:5s} total={s['total']:4d} acc={s['acc']:.1%} avg_cer={s['avg_cer']:.1%}")
    print(f"\n{acceptance_report(passed, details)}")

    # 결과 저장
    out = pd.DataFrame(predictions)
    out['group'] = [freq_group of label]
    out.to_csv("/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/result_val.csv", index=False)
    print("[SAVED] result_val.csv")

if __name__ == '__main__':
    main()
```

**Step 2: 검증 실행**
Run: `venv/bin/python run_eval.py`
Expected:
- val 1,115샘플 평가 완료
- 빈도그룹별 acc/CER 출력
- 수용기준 판정 (현재 v2.2 best 기준 — 통과 못할 가능성 높음, baseline으로 기록)
- result_val.csv 저장

**Step 3: baseline 기록**
- result_val.csv와 출력값을 `doctor_ocr_v3/reports/baseline_v2_2.md`로 저장
- Checkpoint: val_loss, epoch 포함

**Step 4: 커밋**

---

## Task 5: 검증 및 문서화

**Objective:** 평가 레이어가 v3 DESIGN.md의 수용기준과 일치하는지 확인하고 사용법 문서화.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/README.md`

**Step 1: README에 사용법, 지표 정의, 수용기준, 실행 명령 기록**

**Step 2: 전체 테스트 재실행**
Run: `cd /home/dev/DoctorOcr/doctor_ocr_v3/evaluate && ../venv/bin/python -m pytest -v`
Expected: 전부 pass

**Step 3: 커밋** (README + 테스트)

---

## 가드레일 (전 구간)

- v2_2의 evaluate.py / best_model.pth / 모델 정의는 **수정하지 않음** (읽기 전용)
- 평가 레이어는 v3 폴더에 신규 파일로만 추가
- CER은 표준 라이브러리 구현 (추가 pip 설치 없음 — GPU1에서 vLLM과 공존하므로 pip으로 시스템 건드리지 않음)
- GPU: CUDA_VISIBLE_DEVICES='1' (Max-Q) 고정, GPU0 vLLM 간섭 없음
- val split은 기존 (seed 42, 80/20) 재사용 — 재현성 보장
- 예산: 구현은 짧은 작업 (~30분), 실행은 val 1,115샘플 1회

## 성공 기준

- [ ] metrics / aggregate / acceptance 단위테스트 전부 pass
- [ ] run_eval.py가 v2_2 best_model로 val 1,115샘플 평가 완료
- [ ] 빈도그룹(고/중/저)별 acc/CER 출력 확인
- [ ] 수용기준 판정 (고빈도≥90%, 전체CER≤20%) PASS/FAIL 출력
- [ ] baseline_v2_2.md 기록 완료
- [ ] 커밋 + 푸시
