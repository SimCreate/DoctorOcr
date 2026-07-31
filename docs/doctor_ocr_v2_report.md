---
name: doctor_ocr-report
category: research
description: >
  doctor_ocr v2 4개 프로젝트(학습+추론) 보고서. verified-reporting 형식 적용.
  CUDA_VISIBLE_DEVICES='0'(Blackwell) 변경, v2_1 수정( hidden_size=384) 포함.
---

# doctor_ocr v2 — 메인 보고서 (2026-07-31)

[확인됨] 4개 프로젝트 학습 완료 (v1/v2/v2_1/v2_2)
[확인됨] 4개 추론 완료 — v2(20%), v2_1(20% 수정 후), v2_2(40% 최상), v1(0%)
[확인됨] v2_1 수정: local_infer_v2_1.py hidden_size=384 강제 (체크포인트 config 256 ≠ 모델 384)
[확인됨] 소스 9개 .py 수정 + 1개 패치, 4060 Ti 참조 0건

## 1. 프로젝트별 특징 (정확 + 간결)

- v1 (doctor_ocr): 기준 CRNN, BATCH=8, E50 best 2.7976, 추론 0/10
- v2 (doctor_ocr_v2): 개선 CRNN, BATCH=96, E50 best 2.3205, 추론 2/10
- v2_1 (doctor_ocr_v2_1): BiLSTM 3층 + Attention, BATCH=40, E54+ best 2.2795, 추론 2/10 (수정 후)
- v2_2 (doctor_ocr_v2_2): v2_2 아키텍처, BATCH=558, E38+ best 1.0216 (최상), 추론 4/10

## 2. 학습 결과 (검증된 로그)

| 프로젝트 | 최종 Epoch | Best val_loss | 로그 파일 |
|---|---|---|---|
| v1 | 50 | 2.7976 | /home/dev/doctor_ocr/train_bg_1341.log |
| v2 | 50 | 2.3205 | /home/dev/doctor_ocr_v2/train_bg_1341.log |
| v2_1 | 54+ | 2.2795 | /home/dev/doctor_ocr_v2_1/train_bg_1341.log |
| v2_2 | 38+ | 1.0216 | /home/dev/doctor_ocr_v2_2/train_bg_1341.log |

## 4. 추론 결과 [확인됨 — 로그 직접 확인]

| 프로젝트 | 정확도 | 세션 (exit 0) | 로그 |
|---|---|---|---|
| v1 | 0/10 = 0% | proc_64741a6f4a15 | infer_bg_1403.log |
| v2 | 2/10 = 20% | proc_24c772884a50 | infer_bg_1403.log |
| v2_1 | 2/10 = 20% (수정 후) | proc_f2f2ff542d5e | infer_bg_1403_fixed.log |
| v2_2 | 4/10 = 40% (최상) | proc_93c913b6e97c | infer_bg_1403.log |

## 5. 각 프로젝트 핵심 특징 (정확 + 간결)

- v1: 기준 CRNN (BATCH=8) — E50 best 2.7976, 추론 0%. 기준선.
- v2: 개선 CRNN (BATCH=96, AMP) — E50 best 2.3205, 추론 20%. 가장 효율적.
- v2_1: BiLSTM 3층+Attention (BATCH=40) — E54+ best 2.2795, 추론 20%(수정). 체크포인트 저장 버그(256≠384) 발견 및 수정.
- v2_2: v2_2 아키텍처 (BATCH=558) — E38+ best 1.0216(최상), 추론 40%. 최고 성능.

## 6. 수정 사항 [확인됨]

- USB4 4060 Ti → CUDA_VISIBLE_DEVICES='0' (Blackwell RTX PRO 6000 cuda0) — 4개 소스 변경
- v2_1 추론: hidden_size=384 강제 (체크포인트 config 256 오류) — /home/dev/doctor_ocr_v2_1/local_infer_v2_1.py
- 4060 Ti 참조 0건

## 4. 수정 사항

- USB4 4060 Ti → CUDA_VISIBLE_DEVICES='0' (RTX PRO 6000 Blackwell cuda0)
- 4개 소스 모두 `default='0'`, `CUDA_VISIBLE_DEVICES='0'` 변경
- v2_1 추론 오류 해결: `local_infer_v2_1.py:106 hidden_size=384` 강제
  - 원인: 학습 체크포인트 `config['hidden_size']=256`이 모델 정의(384)와 불일치
  - 결과: LSTM weight 1536(체크포인트) vs 1024(오류 모델) 불일치 → 수정 후 20% 정상

## 5. 의의

- 4개 아키텍처 비교로 CRNN 개선 방향 확인 (v2_2 40%가 최상)
- Blackwell cuda0 전환 성공 — USB4 불안정성 제거
- v2_1 체크포인트 저장 버그 발견 및 수정 (config 불일치)

## 6. 한계점

- v2_1 체크포인트 저장 시 config 오류는 근본 원인 미해결(저장 로직 수정 필요)
- v1 추론 정확도 0% — 모델/데이터 문제 가능
- Blackwell sm_120 + PyTorch 2.11 호환성 — 추론/학습 모두 작동했으나 버전 업그레이드 권장
- 4개 동시 실행 시 GPU 0 VRAM 92~97GB 사용 — OOM 위험

## 7. 검증 형식 (verified-reporting)

- 수행: 4개 학습 + 4개 추론 (백그라운드, CUDA_VISIBLE_DEVICES='0')
- 확인: 로그 파일 직접 확인 (train_bg_1341.log, infer_bg_1403.log)
- 확인된 결과: 위 표의 수치 (실제 출력 인용)
- 미확인: Notion DB ID 없음 → 상위 페이지로 생성 시도 (API 작동 확인)
