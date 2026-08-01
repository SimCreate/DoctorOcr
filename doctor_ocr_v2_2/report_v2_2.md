---
name: doctor-ocr-v2-2-report
category: research
description: v2_2 (doctor_ocr_v2_2) 프로젝트 상세 보고 — v2_2 아키텍처, E38+ best 1.0216, 추론 40%, evaluate.py 포함
---

# v2_2 (doctor_ocr_v2_2) — 상세 보고

[확인됨] 학습 완료 — Epoch 38+, best val_loss=1.0216 (proc_f089b9d6fb89 exit 0, 로그 3,456,958바이트)
[확인됨] 추론 완료 — ACCURACY 4/10 = 40.00% (proc_93c913b6e97c exit 0, 최상)
[확인됨] evaluate.py 수정 완료 (CUDA_VISIBLE_DEVICES='0')

## 아키텍처
- v2_2 전용 모델 (model_v2_2.py)
- BATCH_SIZE=8, ACCUM_STEPS=16 (effective batch 128), NUM_WORKERS=4
- USE_AMP=True, USE_GRADIENT_CHECKPOINTING=True, DROPOUT=0.3
- DEVICE=cuda:0 (Blackwell)

## 학습 세부
- Epoch 2 val_loss=3.6855
- Epoch 3 LR=0.000180, ~28-44 it/s (558 batches/epoch)
- Best: 1.0216 (4개 중 최상)

## 추론 세부
- 40% (4/10) — 최고 정확도
- evaluate.py 포함 — 평가/추론 통합

## 핵심
- 가장 좋은 성능
- 큰 유효 배치(8×16=128) 효율적
- evaluate.py 추가로 평가 기능 강화

## 한계점
- 다른 버전과 아키텍처 달라 직접 비교 어려움
- ACCUM_STEPS=16 (유효 배치 128) — 그래디언트 누적 많아 학습 속도 느릴 수 있음
