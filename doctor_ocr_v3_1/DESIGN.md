# v3_1 — v2.1(Attention 디코더) + v3 증강 데이터셋 재학습

> 기간: 2026-08-06 ~
> 목적: "가장 진보된 아키텍처(v2.1 multi-head attention + beam search)가
>       데이터 부족으로 학습에 실패했다"는 진단을, 커진 데이터셋으로 재검증

## TL;DR

| 항목 | 내용 |
|---|---|
| 아키텍처 | v2.1 CRNN — SEBlock CNN 5블록 + BiLSTM 3층(hidden 384) + **Multi-head Attention 8-head + Beam Search** |
| 데이터 | v3 exp2_clean (실사 증강 2배, 13,386장) — train |
| val | v3 클린 고정 split (1,116장, v2 원본 이미지) — v3 exp2_clean과 **동일 기준** |
| venv | `doctor_ocr_v3_1/venv/` (v2_2 venv site-packages 물리 복사, torch 2.13.0+cu132) |
| 핵심 가설 | 증강 데이터(13,386)로 attention 디코더가 이전(원본 5,578 및 v2.1 실패)과 다르게 수렴하는가 |

## 배경 — 왜 v2.1인가

### 모델 아키텍처 계보 (v3_1 설계 시점 기준)

| 버전 | 디코더 | 실측 정확도 | 비고 |
|---|---|---|---|
| v1 | Attention (4-head, LSTM 1) | 0% | 기준선 |
| v2 | Attention (4-head, TF 0.5) | 20% | BATCH 96 + AMP |
| v2.1 | **Attention (8-head, LSTM 2) + Beam Search** | **20%** | hidden 384, 가장 진보 |
| v2.2 | **CTC Head** (Attention 제거) | **40%** | v3 베이스 |
| v3 | v2.2 + 데이터 실험 | 35.8% (클린, 증강 +2.2p) | 데이터 증강만으론 한계 |

- **가장 진보 = v2.1** (8-head attention + beam search, hidden 384)
- **그러나 v2.1은 성능이 안 나왔다** (20%). 원인은 데이터 부족이라는 진단:
  - RxHandBD 라벨 1,788 고유 중 64%가 1회 등장(롱테일)
  - attention 디코더는 학습 데이터가 적으면 alignment/attention 학습에 실패
  - v2.1 실패 근거: `Deliiiii` 같은 반복 토큰 (DESIGN.md §8, 알려진 이슈)
- **v2.2(CTC)로 전환한 건 "더 나아서"가 아니라 "attention이 실패해서"** — 회귀적 선택
- CTC는 attention보다 *구식*이지만 데이터가 적을 때 더 견고. 다만 롱테일 극복엔 한계 확정(v3 클린 실험)

### 그래서 v3_1

v3에서 만든 **커진 증강 데이터셋(13,386) = attention 학습에 필요한 데이터를 갖춘 상태**.
"데이터가 부족해 실패했던 가장 진보된 아키텍처"에 "충분한 데이터"를 주면
CTC(v3 exp2 37.9%)를 넘어설 수 있는지가 실험의 질문.

## 설계 결정

1. **데이터 = exp2_clean (13,386)**: v3에서 만든 실사 증강(회전±5°, 스케일 0.9~1.1, 밝기, 노이즈) 2배. 온디스크 증강.
2. **런타임 증강 OFF**: v2.1 스크립트는 `augment=True` 기본인데, 이미 온디스크로 증강돼 있으므로 중복 증강 방지. (v3 exp2와 동일 데이터 조건)
3. **val = 고정 클린 split**: v3 `data/clean_split/val.csv` 1,116장, v2 원본 이미지에서 읽음.
   - v3 exp2_clean과 **100% 동일한 평가 기준** → 직접 비교 가능
   - 학습(train)에 val 이미지가 들어가지 않도록 구조적 분리 (리키지 방지)
   - ⚠️ exp2_clean 이미지 디렉토리엔 **train 원본+증강만** 있고 val 원본은 없음 → val은 v2 원본 디렉토리 사용
4. **checkpoint config 버그 수정**: v2.1은 hidden 384 학습인데 config에 256 저장 → v3_1은 384로 정확히 저장
5. **하이퍼파라미터**: v2.1 기본값 유지 (LR 3e-4, warmup 5, cosine, batch 40, accum 4 = 효과적 배치 160, early stop patience 20)

## 데이터 구성

```
train: doctor_ocr_v3/data/experiment_2_clean/   (13,386 = 원본 4,462 + 증강 8,924)
  ├── combined_labels.csv   (filename, label, source)  13,386행
  └── img/img/*.jpg         13,386장 (원본 + P####__aug0/1.jpg)
val:   doctor_ocr_v3/data/clean_split/val.csv   (1,116장, v2 원본에서 읽음)
  - v2 원본 이미지: /home/dev/doctor_ocr_v2/dataset/img/img/ (5,578장)
  - val에 aug는 0장 포함 (train과 구조적 분리)
```

> v3_1의 `data/exp2_clean`, `data/clean_split`은 v3 디렉토리로의 **심볼릭 링크** (데이터 중복 저장 방지)

## 실행 방법

```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_1

# 본학습 (GPU1 Max-Q, 배치 40)
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_1.py \
  > logs/train_v3_1.log 2>&1 &
# 로그 확인
tail -f logs/train_v3_1.log
```

## 비교 기준 (v3 exp2_clean vs v3_1)

| 상태 | exact | 고빈도 | 중빈도 | 저빈도 | CER | 판정 |
|---|---|---|---|---|---|---|
| v3 exp2_clean (CTC) | 37.9% | 56.5% | 33.5% | 8.0% | 20.6% | FAIL |
| **v3_1 (Attention)** | **재학습 후 측정** | | | | | ? |

- 목표: v3_1이 exp2_clean(CTC)을 넘는가? 특히 고빈도(≥90% 기준)와 저빈도 개선 확인
- 평가는 v3 evaluate/run_eval.py 재사용 (CER + 빈도그룹 + 수용기준) — 단, v3_1 체크포인트는 v2.1 아키텍처라 디코더 방식이 다름에 주의

## 파일 구조

```
doctor_ocr_v3_1/
├── model/model_v2_1.py      # v2.1 모델 복제본 (원본 수정 금지)
├── scripts/train_v3_1.py    # 재학습 스크립트 (v2.1 기반 + val 고정 + 증강 off)
├── data/{exp2_clean, clean_split}   # v3 심볼릭 링크
├── working/                 # 체크포인트 (gitignore)
├── logs/                    # 학습 로그 (gitignore)
├── venv/                    # 전용 venv (gitignore)
├── evaluate/                # (v3 run_eval 재사용 예정)
└── reports/                 # 결과 보고서
```

## 아직 미정 / 리스크

- **평가 어댑터**: v2.1 체크포인트는 attention 디코더(beam search)라, v3 run_eval(CTC decode)과 호환 안 될 수 있음 → v2.1의 `decode_sequence`/beam search로 평가하거나 어댑터 작성 필요
- **시간**: 배치 40, ~8 it/s → 에폭 ~45초, 80에폭 ≈ 1시간 (vLLM과 GPU1 공존)
- **VRAM**: vLLM(93.7GB)과 공존, 여유 ~3.5GB에서 배치 40 동작 확인 (AMP + gradient checkpointing)
