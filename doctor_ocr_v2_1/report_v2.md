---
name: doctor-ocr-v2-report
category: research
description: v2 (doctor_ocr_v2) 프로젝트 상세 보고 — 개선 CRNN, E50 best 2.3205, 추론 20%
---

# v2 (doctor_ocr_v2) — 상세 보고

[확인됨] 학습 완료 — Epoch 50, best val_loss=2.3205 (proc_b40dad367bcc exit 0, 로그 446,691바이트)
[확인됨] 추론 완료 — ACCURACY 2/10 = 20.00% (proc_24c772884a50 exit 0)
[확인됨] 소스: local_train.py/local_infer.py 수정 (CUDA_VISIBLE_DEVICES='0')

## 아키텍처
- 개선 CRNN (v2 기준)
- BATCH_SIZE=96, AMP=True, NUM_EPOCHS=50
- DEVICE=cuda:0 (Blackwell, CUDA_VISIBLE_DEVICES='0')
- NUM_WORKERS=8, PIN_MEMORY=True

## 학습 세부
- Epoch 12 시점 val_loss=2.7880, Acc=0.45% (검증 중)
- Epoch 49 val_loss=2.3278, Acc=8.15% (90/1116)
- 최종 Epoch 50 완료, best=2.3205
- 저장: working/checkpoints/best_model.pth

## 추론 세부
- 정확도 20% (2/10)
- 모델 로드 성공, DEVICE=cuda:0 확인
- 오류 없음

## 핵심
- v1 대비 val_loss 0.48 개선 (2.7976→2.3205)
- 배치 96으로 효율적 — VRAM ~18% 사용
- 추론 20% — v1(0%) 대비 개선

## 한계점
- 정확도 20%는 아직 낮음
- Blackwell sm_120 호환성 — PyTorch 2.11+cu128 사용 중
