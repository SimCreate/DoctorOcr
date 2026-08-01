---
name: doctor-ocr-v2-1-report
category: research
description: v2_1 (doctor_ocr_v2_1) 프로젝트 상세 보고 — BiLSTM 3층+Attention, E77 best 2.2795, 추론 20%(수정 후). 체크포인트 config 오류 원인 분석 포함.
---

# v2_1 (doctor_ocr_v2_1) — 상세 보고 (수정 포함)

[확인됨] 학습 완료 — Epoch 77 (best 체크포인트), best val_loss=2.2795 (proc_494adea73f46 exit 0, 로그 1,868,186바이트)
[확인됨] 추론 완료(수정 후) — ACCURACY 2/10 = 20.00% (proc_f2f2ff542d5e exit 0, infer_bg_1403_fixed.log)
[확인됨] 수정: local_infer_v2_1.py:106 hidden_size=384 강제 (체크포인트 config 256 ≠ 모델 384)

## 아키텍처 — 개선 CRNN (model_v2_1.py)

3컴포넌트 구조로, v1 대비 CNN에 SEBlock 추가 + BiLSTM 3층(384) + Attention 강화.

### CNNEncoder (SEBlock 5블록)
- Conv-BN-ReLU 5블록, 각 블록 끝에 **SEBlock** 채널 어텐션
- 풀링: MaxPool(2,2)×2 + MaxPool(2,1)×2 + 마지막 Conv(2,p=0) → 출력 512채널
- Gradient checkpointing 지원 (`use_checkpointing`)
- **논문**: SE Block — https://arxiv.org/abs/1709.01507 (Squeeze-and-Excitation Networks)

### BiLSTM (3층, hidden=384)
- hidden_size=384, num_layers=3, bidirectional, dropout=0.3
- 출력 차원: hidden×2 = 768
- **논문**: BiLSTM 기반 OCR — https://arxiv.org/abs/1507.05717 (CRNN, Shi et al.)

### AttentionDecoder (8-head, LSTM 2층)
- **8-head** MultiheadAttention (v1은 4-head) + LSTM 2층(hidden 384)
- Teacher forcing 0.5
- **Beam Search** 지원 (`beam_search`, beam_width=5) — 추론 시 greedy 대신 beam decoding 사용
- **논문**: Attention — https://arxiv.org/abs/1409.0473 (Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau)

### 학습 설정
- BATCH_SIZE=40, ACCUM_STEPS=4 (effective batch 160), NUM_WORKERS=4
- LR=3e-4, USE_AMP=True, Gradient Checkpointing
- DEVICE=cuda:0 (Blackwell)

## 학습 세부
- Epoch 4 val_loss=3.0362 (초기)
- Epoch 5부터 LR=0.0003, TF_Ratio=0.39
- Epoch 77 (best 체크포인트) best=2.2795
- 저장: working/checkpoints/best_model.pth (272MB)

## 추론 — 오류 원인 및 수정

### 오류 (수정 전)
- `RuntimeError: size mismatch for rnn.lstm.weight_ih_l0`
- checkpoint shape: [1536, 1536] (384*4)
- current model: [1024, 1536] (256*4)

### 원인
- 학습 시 `model_v2_1.py`: `hidden_size=384`
- 체크포인트 저장: `config['hidden_size']=256` (잘못 저장)
- 추론: `local_infer_v2_1.py`가 `config.get('hidden_size', 256)` 사용 → 256으로 생성

### 수정
- `/home/dev/doctor_ocr_v2_1/local_infer_v2_1.py:106`
- `hidden_size=384` 강제 (config 무시)
- 결과: ACCURACY 20% 정상 완료

## 핵심
- BiLSTM 3층 + Attention 구조로 개선 시도
- 체크포인트 저장 버그 발견 — 추론 시 구조 불일치
- 수정 후 20% 정상 작동

## 한계점
- 체크포인트 저장 로직 수정 필요 (`local_train_v2_1.py`의 save 부분)
- vllm-env PyTorch 2.11+cu130 — Blackwell sm_120 호환성 확인됨(작동함)
- 추론 정확도 20% — 개선 여지 있음
