# v2.2 Baseline 평가 기록 (2026-08-03)

평가 레이어(v3, CER + 빈도그룹 + 수용기준)로 재측정한 v2.2 baseline.

## 대상 모델
- 체크포인트: `/home/dev/doctor_ocr_v2_2/working/checkpoints/best_model.pth`
- 메타데이터: **epoch 32, val_loss 1.0676**, vocab 74
- 아키텍처: CTC 기반 CRNN (SEBlock CNN 5블록 + BiLSTM 3층 + CTC Head)
- config: hidden 256, dropout 0.3, batch 8, accum 16, grad checkpointing

> 참고: report_v2_2.md는 val_loss 1.0216, 추론 40% 주장. 라이브 best는 1.0676.
> "40%"가 어느 체크포인트인지 불확실 → 여기선 디스크의 best(1.0676)를 baseline으로.

## 평가 설정
- val split: 전체 5,579행 중 80/20 랜덤 (seed 42, torch.manual_seed(42)) → **val 1,116샘플**
- GPU: GPU1 Max-Q (CUDA_VISIBLE_DEVICES=1)
- 평가일: 2026-08-03
- 저장: `result_val.csv` (1,116행)

## 결과

### 빈도그룹별 (수용기준 정의 기준: 10회↑ / 2~9회 / 1회)
| 그룹 | 샘플 | exact_acc | avg_cer |
|------|------|-----------|---------|
| 전체 | 1,116 | 32.7% (365) | 23.7% |
| 고빈도(10↑) | 476 | **52.7%** | **14.7%** |
| 중빈도(2~9) | 403 | 24.6% | 26.7% |
| 저빈도(1) | 237 | 6.3% | 36.6% |

- 기존 채점 방식 exact match: 365/1116 = 32.7%
- CTC val_loss: 1.0080

### 수용기준 판정
- 고빈도 exact ≥ 90%? → 52.7% → **FAIL**
- 전체 CER ≤ 20%? → 23.7% → **FAIL**
- 최종: **FAIL — 개선 필요** (실사용 불가)

## 관찰
1. 빈도가 높을수록 정확도↑, CER↓ → 모델은 "자주 본 단어"는 꽤 읽고, "드문 단어"는 거의 못 읽음
2. 저빈도 avg_cer 36.6% → 완전히 엉뚱한 출력보다는 "부분적" 오류 (단, 여전히 개선 필요)
3. 고빈도가 52.7%로 시작점 — 실험군 3(실사 증강 + light-touch 합성)이 이걸 90%로 끌어올리는 게 목표

## 실행 명령
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3/evaluate
/home/dev/doctor_ocr_v2_2/venv/bin/python run_eval.py
```
