---
name: doctor-ocr-v1-report
category: research
description: v1 (doctor_ocr) 프로젝트 상세 보고 — 기준 CRNN, 학습 E50 best 2.7976, 추론 0%
---

# v1 (doctor_ocr) — 상세 보고

[확인됨] 학습 완료 — Epoch 50, best val_loss=2.7976 (로그: /home/dev/doctor_ocr/train_bg_1341.log, [DONE] 확인)
[확인됨] 추론 완료 — ACCURACY 0/10 = 0.00% (로그: /home/dev/doctor_ocr/infer_bg_1403.log, proc_64741a6f4a15 exit 0)
[확인됨] 소스 수정: CUDA_VISIBLE_DEVICES='0' (default='0')

## 아키텍처 — 기준 CRNN (local_train.py)

3컴포넌트 구조: CNN Encoder + BiLSTM + **Attention Decoder** (CTC 아님).
이후 버전(v2_1/v2_2)의 비교 기준선.

### CNNEncoder (7 Conv Block)
- Conv2d-BN-ReLU 7블록: Conv(3,64,3) → MaxPool(2,2) → Conv(64,128) → MaxPool(2,2) → Conv(128,256) → Conv(256,256)+MaxPool(2,1) → Conv(256,512) → Conv(512,512)+MaxPool(2,1) → Conv(512,512,2)
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
- BATCH_SIZE=8, ACCUM_STEPS=2 (유효 배치 16), NUM_WORKERS=2
- LR=1e-4, NUM_EPOCHS=50, WEIGHT_DECAY=1e-5
- 손실: CrossEntropy(ignore_index=<PAD>) — Attention seq2seq 기반
- DEVICE=cuda:0 (Blackwell, CUDA_VISIBLE_DEVICES='0')

## 학습 세부 결과
- 총 50 epoch 완료
- 최종 val_loss: 2.7976
- 저장: /home/dev/doctor_ocr/working/checkpoints/best_model.pth
- 로그 83,898바이트 — 전체 epoch 진행 확인

## 추론 세부 결과
- 정확도 0% (0/10)
- 테스트 데이터: /home/dev/doctor_ocr/dataset/img/img + label CSV
- 모델: best_model.pth 로드 성공
- 오류 없음 — 단순히 정확도가 0

## 핵심
- 기준선(baseline) 역할
- 50 epoch에서 val_loss 2.8 수준 — 개선 여지 큼
- 추론 0%는 데이터 품질 또는 모델 용량 문제 가능

## 한계점
- BATCH=8로 작음 — VRAM 여유가 있음에도 작은 배치
- 정확도 0% — 실제 예측 품질 확인 필요
- 다른 버전(v2/v2_2)과 비교 시 성능 차이 명확
