---
name: doctor-ocr-v2-report
category: research
description: v2 (doctor_ocr_v2) 프로젝트 상세 보고 — 개선 CRNN, E50 best 2.3205, 추론 20%
---

# v2 (doctor_ocr_v2) — 상세 보고

[확인됨] 학습 완료 — Epoch 50, best val_loss=2.3205 (proc_b40dad367bcc exit 0, 로그 446,691바이트)
[확인됨] 추론 완료 — ACCURACY 2/10 = 20.00% (proc_24c772884a50 exit 0)
[확인됨] 소스: local_train.py/local_infer.py 수정 (CUDA_VISIBLE_DEVICES='0')

## 아키텍처 — 개선 CRNN (local_train.py)

v1과 동일한 3컴포넌트 CRNN 구조(CNN + BiLSTM + Attention Decoder)를 유지하되,
**대용량 배치(96) + AMP**로 학습 효율 개선.

### CNNEncoder (7 Conv Block)
- Conv2d-BN-ReLU 7블록, MaxPool(2,2)×2 + MaxPool(2,1)×2 + 마지막 Conv(2,p=0)
- 출력 채널 512, 출력 특징맵 (B, 512, 3, 63)
- **논문**: CRNN — https://arxiv.org/abs/1507.05717 (An End-to-End Trainable Neural Network for Image-based Sequence Recognition, Shi et al.)

### BiLSTM (2층, hidden=256)
- hidden_size=256, num_layers=2, bidirectional, dropout=0.2
- 출력 차원: hidden×2 = 512

### AttentionDecoder (4-head, LSTM 1층)
- **4-head** MultiheadAttention + LSTM 1층(hidden 256)
- Teacher forcing 0.5
- `encoder_proj`를 `__init__`에서 생성해 forward마다 재생성하던 **메모리 누수 버그 수정** (`[FIX]`)
- **논문**: Attention — https://arxiv.org/abs/1409.0473 (Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau)

### 학습 설정
- BATCH_SIZE=96, ACCUM_STEPS=1 (유효 배치 96), NUM_WORKERS=8, PIN_MEMORY=True
- LR=3e-4, USE_AMP=True, NUM_EPOCHS=50
- DEVICE=cuda:0 (Blackwell, CUDA_VISIBLE_DEVICES='0')

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
