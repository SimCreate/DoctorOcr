# v3 클린 스플릿 재실험 — 리키지 제거 후 결과 (2026-08-05)

## TL;DR

**8/4의 "증강 = 압도적 레버리지 (98.8%)" 결론은 리키지 아티팩트였다.**

클린 split(원본 80/20 → train에만 증강/합성)으로 재학습한 결과, 증강 효과는 **+2.2p에 불과**하다.
수용기준(고빈도 ≥90%, 전체 CER ≤20%)은 모든 실험군에서 여전히 **FAIL**이다.

| 실험군 | 데이터 | exact | 고빈도 | 중빈도 | 저빈도 | 전체 CER | 판정 |
|---|---|---|---|---|---|---|---|
| exp1_clean | 원본만 (4,462) | **35.8%** | 56.3% | 28.0% | 7.6% | 22.1% | FAIL |
| exp2_clean | +실사증강 2배 (13,386) | **37.9%** | 56.5% | 33.5% | 8.0% | 20.6% | FAIL |
| exp3_clean | +합성 12.5% (13,943) | **36.4%** | 54.2% | 30.0% | 11.4% | 20.7% | FAIL |

- exp1 vs exp2 증강 효과: **+2.2p** (35.8 → 37.9)
- exp3 합성 추가 효과: **-1.5p** (37.9 → 36.4), 저빈도만 +3.4p 소폭 개선
- 증강은 중빈도(28→33.5)에서만 유의미, 고빈도는 사실상 변화 없음

## 배경 — 왜 재실험했나

8/4 비교 보고서의 "원본 val 공정 비교 98.8%"는 **이미지 단위 리키지** 때문에 부풀려졌다.

- 기존 파이프라인은 **증강 포함 전체 데이터셋**을 seed 42로 80/20 split
- run_eval이 val로 쓰는 원본 1,116장 중 **~80% (약 890장)가 exp2/3 train에 그대로 포함**
- → 98.8%는 "eval 이미지를 이미 학습한" 모델을 측정한 수치

이번 재실험은 이를 구조적으로 차단한다.

## 수정된 파이프라인

```
[원본 5,579장]
   │  seed 42, 80/20 (HandwritingDataset 유효 이미지 기준 = 5,578)
   ▼
train 4,462 ──▶ exp1_clean (원본만)
   │  + 증강 2배 ──▶ exp2_clean (real 4,462 + aug 8,924 = 13,386)
   │  + 저빈도 합성 ──▶ exp3_clean (13,386 + synth 557 = 13,943)
   ▼
val 1,116 (고정, train과 구조적 분리 — 8/4의 same val)
```

- **val은 어떤 실험군의 train에도 포함되지 않음** (증강 쌍둥이 포함 0장 확정)
- 클린 val 1,116장은 **8/4의 run_eval val과 100% 동일** (`verify_split_repro.py`로 대조) → 수치 직접 비교 가능

## 검증 내역

| 검증 | 결과 |
|---|---|
| 클린 val 1,116 == 8/4 run_eval val | 100% 일치 (스크립트 대조) |
| val ∩ exp{1,2,3}_clean train | **0장** (증강 쌍둥이 `__augN` 포함 0) |
| exp{1,2,3}_clean 무결성 (파일/라벨/중복/충돌) | ALL PASS |
| 학습 중 리키지 가드 (`train_exp.py --clean`) | train∩val >0 이면 중단 |

## 산출물/경로

```
data/clean_split/{train.csv, val.csv}         # 원본 80/20 (1회 생성, 공용)
data/experiment_{1,2,3}_clean/                # train만으로 재구성
working/exp{1,2,3}_clean/checkpoints/          # 새 체크포인트 (gitignore)
evaluate/result_exp{1,2,3}_clean.csv          # 클린 val 평가 결과
reports/exp_comparison_20260805_clean.md      # 본 문서
```

## 재현 방법

```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3

# 1. 클린 split + 실험군 데이터 재구성 (train만 증강/합성)
venv/bin/python scripts/build_clean_split.py

# 2. 재학습 (GPU1 Max-Q, `--clean` = 리키지 가드 + 고정 val)
CUDA_VISIBLE_DEVICES=1 venv/bin/python -u scripts/train_exp.py --exp 1 --clean
CUDA_VISIBLE_DEVICES=1 venv/bin/python -u scripts/train_exp.py --exp 2 --clean
CUDA_VISIBLE_DEVICES=1 venv/bin/python -u scripts/train_exp.py --exp 3 --clean

# 3. 클린 val 정식 평가 (고정 split, split 없이 전체 평가)
CUDA_VISIBLE_DEVICES=1 venv/bin/python evaluate/run_eval.py \
  --ckpt working/exp2_clean/checkpoints/best_model.pth \
  --char-dict working/exp2_clean/char_dict.pkl \
  --csv /home/dev/doctor_ocr_v2/dataset/combined_labels.csv \
  --img-dir /home/dev/doctor_ocr_v2/dataset/img/img \
  --fixed-val-csv data/clean_split/val.csv \
  --out evaluate/result_exp2_clean.csv
```

## 학습 세부

| 실험군 | 종료 | best val_loss | 최종 val acc |
|---|---|---|---|
| exp1_clean | epoch 49 (early stop) | 0.9779 | 43.2% (학습 로그) |
| exp2_clean | epoch 26 (early stop) | 0.9067 | 47.6% (학습 로그) |
| exp3_clean | epoch 25 (early stop) | 0.8716 | 47.0% (학습 로그) |

> 학습 로그 val acc(43~48%)보다 run_eval 재측정(35~38%)이 낮다 — 로그는 학습 중 augment 변형이 섞인 val, run_eval은 원본 그대로라 디코드 조건이 달라서 생긴 차이. **결론 수치는 run_eval(원본 그대로) 기준을 쓴다.**

## 결론 & 한계 (정직한 기록)

1. **8/4 "증강 압도" 결론 폐기** — 리키지가 만든 착시. 클린 기준 증강은 중빈도만 소폭 개선.
2. **한계 1 — early stop이 빠름**: exp2/3이 epoch 25-26에서 멈춤 (patience 15). 최대 성능을 못 냈을 가능성.
3. **한계 2 — 온디스크 증강 강도**: 2배 하드 증강 + 런타임 augment 동시 적용에도 효과 미미 → 데이터 증강만으로는 롱테일 극복 안 됨을 시사.
4. **함의**: 수용기준 통과하려면 데이터 전략이 아니라 **아키텍처/디코더 개선** 또는 **라벨 오류 정제**(필기체 라벨 오타가 정확도 하한을 결정)가 우선일 수 있음.

## 변경 파일

- `scripts/build_clean_split.py` (신규) — 클린 split + train만 증강/합성 재구성
- `scripts/verify_split_repro.py` (신규) — 클린 val == 8/4 val 대조 검증
- `scripts/train_exp.py` — `--clean` 옵션 (고정 val, 리키지 가드, 원본 char_dict)
- `evaluate/run_eval.py` — `--fixed-val-csv` 옵션 (고정 split 평가)
- `reports/exp_comparison_20260805_clean.md` (본 문서)
