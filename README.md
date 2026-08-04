# DoctorOcr

의사 손글씨 처방전 OCR — CRNN 계열 모델 학습·추론 파이프라인
Kaggle RxHandBD 데이터셋(5,578장) 기반, v1 → v2.2 → v3 순차 실험 (모노레포)

- 현재 메인 스트림: **doctor_ocr_v3** (자가개선 시나리오 + 실험군 비교 + 평가 레이어)
- 최고 성능: v3 **exp2 (실사 증강 2배)** — 원본 val 98.8% / 수용기준 PASS

## 프로젝트 구조

모노레포: `/home/dev/DoctorOcr/` (git 저장소)

```
DoctorOcr/
├── README.md                     # 상위 인덱스 (본 파일)
├── .gitignore
├── doctor_ocr/                   # v1 — 기준 CRNN (Attention Decoder)
│   ├── local_train.py / local_infer.py
│   └── doctor_ocr_v1.md
├── doctor_ocr_v2/                # v2 — 개선 CRNN (대용량 배치 + AMP)
│   ├── local_train.py / local_infer.py
│   └── PROJECT_LOG.md
├── doctor_ocr_v2_1/              # v2.1 — BiLSTM 3층 + Attention
│   ├── local_train_v2_1.py / local_infer_v2_1.py
│   ├── model/model_v2_1.py
│   └── report_v2_1.md
├── doctor_ocr_v2_2/              # v2.2 — CTC 기반 CRNN (v3의 베이스)
│   ├── local_train_v2_2.py / local_infer_v2_2.py
│   ├── evaluate.py
│   ├── model/model_v2_2.py
│   └── report_v2_2.md
└── doctor_ocr_v3/                # v3 — 데이터 실험 + 평가 레이어 (메인)
    ├── DESIGN.md                 # 자가개선 시나리오 설계
    ├── DATASETS.md               # 데이터셋 구조/재생성 가이드
    ├── scripts/                  # 데이터 가공 + 학습 스크립트
    │   ├── augment_dataset.py    #   실사 증강 (exp2)
    │   ├── synthesize_labels.py  #   저빈도 합성 (exp3)
    │   ├── verify_datasets.py    #   데이터셋 무결성 검증
    │   └── train_exp.py          #   실험군별 학습 (exp 1/2/3)
    ├── evaluate/                 # 평가 레이어 (CER + 빈도그룹 + 수용기준)
    │   ├── run_eval.py           #   평가 실행
    │   ├── metrics.py / aggregate.py / acceptance.py
    │   └── test_*.py             #   단위 테스트
    ├── plans/                    # 구현 계획 문서
    ├── reports/                  # baseline / 실험 비교 보고서
    ├── logs/                     # 학습 로그 (gitignore)
    ├── data/                     # 실험군 1/2/3 데이터 (gitignore)
    └── working/                  # 체크포인트 (gitignore)
```

> 데이터(`data/`, `working/`, `logs/`)는 `.gitignore` — 저장소에는 소스·문서·보고서만.
> 원본(v2 dataset)은 절대 수정하지 않고, v3 실험군 데이터는 복사/증강/합성물만 사용.

## 버전별 결과 요약

| 항목 | v1 `doctor_ocr` | v2 `_v2` | v2.1 `_v2_1` | v2.2 `_v2_2` | v3 `_v3` |
|---|---|---|---|---|---|
| 아키텍처 | CRNN (7CNN+BiLSTM2+Attn) | 개선 CRNN | BiLSTM 3층+Attention | CTC 기반 CRNN | v2.2 + 데이터 실험 |
| 손실 함수 | CrossEntropy | CrossEntropy | CrossEntropy | **CTC Loss** | CTC Loss |
| Best val_loss | 2.7976 | 2.3205 | 2.2795 | 1.0216 | 0.0825 (exp2) |
| 추론 정확도 | 0% | 20% | 20% | 40% | **98.8%** (원본 val) |

**v3 실험군별 결과 (2026-08-04, 수용기준: 고빈도 exact ≥90% AND 전체 CER ≤20%)**

| 실험군 | 데이터 구성 | val acc (로그) | 원본 val 공정비교 | 수용기준 |
|---|---|---|---|---|
| exp1 | 원본만 (5,578) | 41.9% @49 | 35.8% (고56/중28/저8) | FAIL |
| **exp2** | **원본+증강 2배 (16,735)** | **93.7% @80** | **98.8%** (고99.6/중99/저97) | **PASS** |
| exp3 | exp2+저빈도 합성 12.5% (17,432) | 92.9% @80 | 98.8% (고99.8/중98.3/저97.9) | PASS |

- **핵심 결론**: 증강이 압도적 효과 (+63p), 저빈도 합성 추가는 미미/소폭 하락 → 실사용 기본은 exp2
- 상세: `doctor_ocr_v3/reports/exp_comparison_20260804.md`

## 아키텍처 진화

4개 버전 모두 CRNN 3컴포넌트(CNN Encoder + BiLSTM + Decoder) 골격을 유지하되, 디코더와 학습 전략이 단계적으로 개선:

| 버전 | CNN Encoder | BiLSTM | Decoder | 개선 포인트 |
|---|---|---|---|---|
| v1 | 7 Conv Block | 2층, hidden 256, dropout 0.2 | Attention (4-head, LSTM 1층), TF 0.5 | 기준선 |
| v2 | 7 Conv Block | 2층, hidden 256 | Attention (4-head), TF 0.5 | BATCH 96 + AMP |
| v2.1 | SEBlock 5블록 | 3층, hidden 384, dropout 0.3 | Attention (8-head, LSTM 2층) + Beam Search | SEBlock + 층수 증가 + 8-head |
| v2.2 | SEBlock 5블록 | 3층, hidden 256, dropout 0.3 | **CTC Head** | CTC 전환 (Attention 제거) |
| v3 | v2.2 동일 | v2.2 동일 | v2.2 동일 (12.68M) | 데이터 증강·합성 실험 + CER/빈도그룹 평가 |

## 참고 논문

| 기술 | 논문 | 링크 |
|---|---|---|
| CRNN | An End-to-End Trainable Neural Network for Image-based Sequence Recognition (Shi et al., 2015) | https://arxiv.org/abs/1507.05717 |
| CTC | Connectionist Temporal Classification (Graves, 2006) | https://www.cs.toronto.edu/~graves/icml_2006.pdf |
| Attention | Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014) | https://arxiv.org/abs/1409.0473 |
| SE Block | Squeeze-and-Excitation Networks (Hu et al., 2017) | https://arxiv.org/abs/1709.01507 |

## 데이터셋

- **소스**: Kaggle RxHandBD — 의사 처방전 손글씨 5,578장 (Train 5,000 + Test 400)
- **고유 문자**: 73자 (알파벳, 숫자, 특수문자)
- **구조**: `dataset/img/img/` 이미지 + `combined_labels.csv` (filename, label)
- **v3 실험군**:
  - exp1: 원본만 (source=real)
  - exp2: 원본 + 실사 증강 2배 (회전±5°, 스케일, 밝기, 노이즈 — source=real/aug)
  - exp3: exp2 + 저빈도(1~2회) 라벨 폰트 합성 12.5% 캡 (source=synth)
  - 상세: `doctor_ocr_v3/DATASETS.md`
- 원본 데이터/라벨 CSV는 `.gitignore` 제외 (용량 문제)

## 하드웨어 / 환경

- **GPU**: 2× RTX PRO 6000 Blackwell (192GB VRAM, GPU0 Workstation 450W + GPU1 Max-Q 300W)
- **GPU 할당 (현재)**:
  - vLLM (DeepSeek V4 Flash, TP2): GPU0 + GPU1 공존
  - **doctor_ocr 학습/평가: GPU1 (Max-Q) 타겟팅** — `CUDA_VISIBLE_DEVICES=1` 강제
  - Blackwell sm_120 미지원엔 cu132 PyTorch 빌드 필요 (torch 2.13.0+cu132 사용)
- **venv**: v1/v2/v2_1 → `/home/dev/doctor_ocr/venv`, `/home/dev/vllm-env`
  v2_2 → `/home/dev/doctor_ocr_v2_2/venv` (torch 2.13.0+cu132) ★ v3 학습·평가도 이 venv 사용

## 실행 방법

> 스크립트 내 경로는 라이브 디렉토리(`/home/dev/doctor_ocr_*`) 기준
> 모노레포 사본 실행 시 경로 선수정 필요

### v3 (메인) — 학습
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3
# 실험군별 학습 (GPU1 Max-Q, 백그라운드 권장)
CUDA_VISIBLE_DEVICES=1 /home/dev/doctor_ocr_v2_2/venv/bin/python scripts/train_exp.py --exp 2 > logs/train_exp2.log 2>&1 &
# exp: 1(원본) / 2(증강2배) / 3(증강+합성)
```

### v3 — 평가 레이어 (CER + 빈도그룹 + 수용기준)
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3
VENV=/home/dev/doctor_ocr_v2_2/venv/bin/python
# 공정 비교 (모든 모델 → 같은 원본 val)
CUDA_VISIBLE_DEVICES=1 $VENV evaluate/run_eval.py \
  --ckpt working/exp2/checkpoints/best_model.pth \
  --char-dict working/exp2/char_dict.pkl \
  --csv /home/dev/doctor_ocr_v2/dataset/combined_labels.csv \
  --img-dir /home/dev/doctor_ocr_v2/dataset/img/img \
  --out evaluate/result_exp2_orig.csv
# 자기 데이터 재현은 --csv/--img-dir을 data/experiment_N/으로 변경
```

### v2.2 (베이스)
```bash
cd /home/dev/doctor_ocr_v2_2 && ./venv/bin/python local_train_v2_2.py
cd /home/dev/doctor_ocr_v2_2 && ./venv/bin/python local_infer_v2_2.py
cd /home/dev/doctor_ocr_v2_2 && ./venv/bin/python evaluate.py  # 검증셋 정확도 + 분석
```

- 학습: val_loss 최소 시 `working/checkpoints/best_model.pth` 저장 (epoch/optimizer/config 포함)
- 문자 사전: `working/char_dict.pkl` (char2idx, idx2char)

## 알려진 이슈 / 교훈

- **v2.1 체크포인트 config 버그**: 학습 시 `hidden_size=384`인데 config엔 `256` 저장 — 추론 시 강제 우회
- **Attention 반복 토큰**: teacher forcing 0.5에도 `Deliiiii` 같은 반복 생성 → CTC로 해결 (v2.2 전환 이유)
- **DataParallel 비효율**: 소규모 모델(12M)에선 단일 GPU + 대용량 배치가 3.3× 빠름
- **GPU 공존**: vLLM(93GB×2)과 학습 동시 진행 시 GPU1 free ~3.6GB. v3 학습 VRAM ~1.6GB로 공존 가능
- **저빈도 합성 한계**: 폰트 합성(exp3)은 저빈도 개선에 미미 — 실데이터 증강 추가가 더 유효

## Git

```bash
cd /home/dev/DoctorOcr
git add -A && git commit -m "..." && git push origin main
```

- 저장소: SimCreate/DoctorOcr (main)
- v3 데이터/체크포인트/로그는 gitignore — 소스·문서·보고서만 커밋
