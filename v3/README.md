# DoctorOcr v3_1 — v2.1(Attention 디코더) + 증강 데이터 재학습

의사 손글씨 처방전 OCR — **v2.1 아키텍처 + v3 증강 데이터셋** 실험 프로젝트.

- 아키텍처: v2.1 CRNN (SEBlock CNN 5블록 + BiLSTM 3층·hidden 384 + **Multi-head Attention 8-head + Beam Search**)
- 데이터: v3 `exp2_clean` (실사 증강 2배, 13,386장) — 프로젝트 내 가장 진보된 디코더를 충분한 데이터로 재검증
- val: v3 클린 고정 split (1,116장, v2 원본) — v3 exp2_clean과 동일 기준 비교
- 전용 venv: `doctor_ocr_v3_1/venv/`

상세: `DESIGN.md` 참조

## 실행

```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_1
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_1.py > logs/train_v3_1.log 2>&1 &
```
