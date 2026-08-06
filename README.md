# DoctorOcr

의사 손글씨 처방전 OCR — CRNN 계열 모델 학습·추론 파이프라인
Kaggle RxHandBD 데이터셋(5,578장) 기반, v1 → v3_1 순차 실험 (모노레포)

- 현재 메인: **doctor_ocr_v3_1** (attention 디코더 + 증강 데이터, 자립 패키지)
- v2.2 / v3 (CTC 계열)은 평가 완료 → **`legacy/` 로 보관**
- **현재 상태**: 클린 스플릿 기준 전 실험 **FAIL** — 수용기준(고빈도 ≥90%, CER ≤20%) 미충족
  - v3(CTC): exp2_clean 37.9% · v3_1(Attention): 30.4% — CTC가 실질 우위
  - 8/4 "98.8% PASS"는 리키지 착시로 폐기 → `legacy/doctor_ocr_v3/reports/exp_comparison_20260805_clean.md`

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
├── doctor_ocr_v3_1/              # v3_1 — ★ 현재 메인 (attention 디코더 + 증강 데이터, 자립)
│   ├── DESIGN.md                 # 설계/가설/데이터 구성
│   ├── scripts/
│   │   ├── train_v3_1.py         #   학습 (v2.1 attention + exp2_clean 13,386장)
│   │   ├── eval_v3_1.py          #   beam search 평가 (클린 val 1,116장)
│   │   └── make_presentation.py  #   발표 PPT 생성
│   ├── evaluate/                 # 자체 평가 레이어 (v3에서 복제)
│   │   ├── metrics.py / aggregate.py / acceptance.py
│   │   └── result_*.csv          #   평가 결과 (v3_1 + v3 클린)
│   ├── utils/result_viewer.py    #   Streamlit 정성 분석 뷰어
│   ├── model/model_v2_1.py       # v2.1 모델 복제 (자립)
│   ├── data/                     # exp2_clean + clean_split (로컬 복사본)
│   ├── reports/                  # 평가 보고서
│   └── venv/                     # 전용 venv (torch 2.13.0+cu132)
└── legacy/                       # ★ 보관 (CTC 기반, 평가 완료/종료)
    ├── doctor_ocr_v2_2/          # v2.2 — CTC 기반 CRNN
    └── doctor_ocr_v3/            # v3 — CTC 데이터 실험 + 평가 레이어 원본
```

> 데이터(`data/`, `working/`, `logs/`)는 `.gitignore` — 저장소에는 소스·문서·보고서만.
> 원본(v2 dataset)은 절대 수정하지 않고, v3 실험군 데이터는 복사/증강/합성물만 사용.

## 버전별 결과 요약 (클린 val 1,116장, 동일 기준)

| 항목 | v1 | v2 | v2.1 | v2.2 | v3 | v3_1 |
|---|---|---|---|---|---|---|
| 아키텍처 | CRNN (7CNN+BiLSTM2+Attn) | 개선 CRNN | BiLSTM 3층+Attention | CTC 기반 CRNN | v2.2 + 데이터 실험 | v2.1 + 증강 데이터 |
| 손실 함수 | CrossEntropy | CrossEntropy | CrossEntropy | **CTC Loss** | CTC Loss | CrossEntropy (Attention) |
| Best val_loss | 2.7976 | 2.3205 | 2.2795 | 1.0216 | 0.8716 (exp3_clean) | 1.3886 (v3_1) |
| 추론 정확도 (클린) | 0% | 20% | 20% | 40% | **37.9%** (exp2_clean) | 30.4% (beam=5) |
| CER (클린) | - | - | - | - | 20.6% | 38.2% |
| 상태 | 활성 | 활성 | 활성 | legacy | legacy | **★ 메인** |

> v2.2 / v3는 2026-08-06 `legacy/`로 보관 (CTC 계열, 평가 완료). v3_1이 현재 메인.

**v3 실험군 결과 (보관 기록 — 클린 스플릿 재실험, 2026-08-05)**

> ⚠️ 8/4 1차 실험의 "98.8% 원본 val 공정비교"는 이미지 단위 리키지(증강 포함 전체 split, val 상당수가 train에 포함)로 부풀려진 수치 → **폐기**.
> 클린 split(원본 80/20 → train에만 증강/합성, val과 구조적 분리) 기준 재측정 결과 전 실험군 FAIL.

| 실험군 | 데이터 구성 | val acc (로그) | 원본 val 공정비교 (클린) | 수용기준 |
|---|---|---|---|---|
| exp1_clean | 원본만 (4,462) | 43.2% @49 | **35.8%** (고56.3/중28.0/저7.6, CER 22.1%) | FAIL |
| exp2_clean | +실사 증강 2배 (13,386) | 47.6% @26 | **37.9%** (고56.5/중33.5/저8.0, CER 20.6%) | FAIL |
| exp3_clean | +합성 12.5% (13,943) | 47.0% @25 | **36.4%** (고54.2/중30.0/저11.4, CER 20.7%) | FAIL |

- **v3 결론**: 8/4 "증강 = 압도적 효과(+63p)"는 리키지 착시. 클린 기준 증강 효과는 **+2.2p**에 불과(35.8→37.9). 저빈도 합성은 -1.5p.
- 상세: `legacy/doctor_ocr_v3/reports/exp_comparison_20260805_clean.md`

**v3_1 결과 (2026-08-06, attention + 증강 데이터, 현재 메인)**

| 방식 | exact | CER | 고빈도 | 중빈도 | 저빈도 | 판정 |
|---|---|---|---|---|---|---|
| v3 exp2_clean (CTC) | **37.9%** | **20.6%** | 56.5% | 33.5% | 8.0% | FAIL |
| v3_1 (Attention, beam=5) | 30.4% | 38.2% | 54.8% | 3.2% | 0.4% | FAIL |

- **v3_1 결론**: attention은 학습(loss 수렴)은 개선됐으나, 추론에서 CTC 미달 — 고빈도만 작동 중·저빈도 붕괴 (중 3.2% / 저 0.4%).
- "데이터가 attention을 살린다"는 부분 검증, "attention > CTC"는 **기각**.
- 학습 로그 acc(43.1%)가 실제 디코딩(30.4%)보다 과대평가됨 — 평가는 항상 디코더 재측정으로 판단.
- 상세: `doctor_ocr_v3_1/reports/eval_v3_1_20260806.md`

> 공통: 수용기준(고빈도 ≥90%, CER ≤20%) 미충족. 통과하려면 데이터 전략이 아니라 **아키텍처/디코더 개선** 또는 **라벨 오류 정제** 우선.

## 아키텍처 진화

6개 버전 모두 CRNN 3컴포넌트(CNN Encoder + BiLSTM + Decoder) 골격을 유지하되, 디코더와 학습 전략이 단계적으로 개선:

| 버전 | CNN Encoder | BiLSTM | Decoder | 개선 포인트 |
|---|---|---|---|---|
| v1 | 7 Conv Block | 2층, hidden 256, dropout 0.2 | Attention (4-head, LSTM 1층), TF 0.5 | 기준선 |
| v2 | 7 Conv Block | 2층, hidden 256 | Attention (4-head), TF 0.5 | BATCH 96 + AMP |
| v2.1 | SEBlock 5블록 | 3층, hidden 384, dropout 0.3 | Attention (8-head, LSTM 2층) + Beam Search | SEBlock + 층수 증가 + 8-head |
| v2.2 | SEBlock 5블록 | 3층, hidden 256, dropout 0.3 | **CTC Head** | CTC 전환 (Attention 제거) |
| v3 | v2.2 동일 | v2.2 동일 | v2.2 동일 (12.68M) | 데이터 증강·합성 실험 + CER/빈도그룹 평가 |
| v3_1 | v2.1 동일 (SEBlock) | v2.1 동일 (3층, hidden 384) | **Attention 8-head + Beam Search (22.7M)** | 증강 데이터로 attention 재검증 → CTC 미달로 기각 |

## 참고 논문

| 기술 | 논문 | 링크 |
|---|---|---|
| CRNN | An End-to-End Trainable Neural Network for Image-based Sequence Recognition (Shi et al., 2015) | https://arxiv.org/abs/1507.05717 |
| CTC | Connectionist Temporal Classification (Graves, 2006) | https://www.cs.toronto.edu/~graves/icml_2006.pdf |
| SE Block | Squeeze-and-Excitation Networks (Hu et al., 2017) | https://arxiv.org/abs/1709.01507 |

## 데이터셋

- **소스**: Kaggle RxHandBD — 의사 처방전 손글씨 5,578장 (Train 5,000 + Test 400)
- **고유 문자**: 73자 (알파벳, 숫자, 특수문자)
- **구조**: `dataset/img/img/` 이미지 + `combined_labels.csv` (filename, label)
- **v3 실험군**:
  - exp1: 원본만 (source=real)
  - exp2: 원본 + 실사 증강 2배 (회전±5°, 스케일, 밝기, 노이즈 — source=real/aug)
  - exp3: exp2 + 저빈도(1~2회) 라벨 폰트 합성 12.5% 캡 (source=synth)
  - 상세: `legacy/doctor_ocr_v3/DATASETS.md`
- 원본 데이터/라벨 CSV는 `.gitignore` 제외 (용량 문제)

## 하드웨어 / 환경

- **GPU**: 2× RTX PRO 6000 Blackwell (192GB VRAM, GPU0 Workstation 450W + GPU1 Max-Q 300W)
- **GPU 할당 (현재)**:
  - vLLM (DeepSeek V4 Flash, TP2): GPU0 + GPU1 공존
  - **doctor_ocr 학습/평가: GPU1 (Max-Q) 타겟팅** — `CUDA_VISIBLE_DEVICES=1` 강제
  - Blackwell sm_120 미지원엔 cu132 PyTorch 빌드 필요 (torch 2.13.0+cu132 사용)
- **venv**: v1/v2/v2_1 → `/home/dev/doctor_ocr/venv`, `/home/dev/vllm-env`
  v2_2 코드 → `legacy/doctor_ocr_v2_2`, venv 원본 → `/home/dev/doctor_ocr_v2_2/venv` (torch 2.13.0+cu132)
  **v3_1 → `/home/dev/DoctorOcr/doctor_ocr_v3_1/venv`** (전용 venv, v2_2 venv site-packages 물리 복사 — torch 2.13.0+cu132)
- **v3_1 코드 자립**: `model/model_v2_1.py`, `evaluate/{metrics,aggregate,acceptance}.py`는 원본(v2.1/v3)의 복제본. v3_1 파이프라인은 v3_1 내부 모듈만 사용 — legacy와 독립.
  > venv 재생성: `python3 -m venv venv && rsync -a /home/dev/doctor_ocr_v2_2/venv/lib/python3.13/site-packages/ venv/lib/python3.13/site-packages/`

## 실행 방법

> 스크립트 내 경로는 라이브 디렉토리(`/home/dev/DoctorOcr/`) 기준. GPU1(Max-Q) 타겟팅.

### v3_1 (메인) — 학습
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_1
# attention 디코더 (v2.1) + exp2_clean(증강 13,386장), GPU1 백그라운드 권장
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_1.py > logs/train_v3_1.log 2>&1 &
```

### v3_1 — 평가 (beam search, 클린 val 1,116장)
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_1
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/eval_v3_1.py \
  --ckpt working/checkpoints/best_model.pth
# 결과: evaluate/result_v3_1_clean.csv + 빈도그룹/CER/수용기준 판정
```

### v3_1 — 결과 뷰어 (Streamlit, 정성 분석)
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_1
venv/bin/python -m streamlit run utils/result_viewer.py
```

### legacy (v2.2 / v3 — 보관, 참고용)
```bash
# v3(CTC) 실험 학습
cd /home/dev/DoctorOcr/legacy/doctor_ocr_v3
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_exp.py --exp 2 > logs/train_exp2.log 2>&1 &
# v3 평가
CUDA_VISIBLE_DEVICES=1 venv/bin/python evaluate/run_eval.py --ckpt working/exp2/checkpoints/best_model.pth
```

### v2.2 (베이스, legacy)
```bash
cd /home/dev/DoctorOcr/legacy/doctor_ocr_v2_2 && ./venv/bin/python local_train_v2_2.py
cd /home/dev/DoctorOcr/legacy/doctor_ocr_v2_2 && ./venv/bin/python local_infer_v2_2.py
cd /home/dev/DoctorOcr/legacy/doctor_ocr_v2_2 && ./venv/bin/python evaluate.py  # 검증셋 정확도 + 분석
```

- 학습: val_loss 최소 시 `working/checkpoints/best_model.pth` 저장 (epoch/optimizer/config 포함)
- 문자 사전: `working/char_dict.pkl` (char2idx, idx2char)

## 알려진 이슈 / 교훈

- **v2.1 체크포인트 config 버그**: 학습 시 `hidden_size=384`인데 config엔 `256` 저장 — 추론 시 강제 우회
- **Attention 반복 토큰**: teacher forcing 0.5에도 `Deliiiii` 같은 반복 생성 → CTC로 해결 (v2.2 전환 이유)
- **DataParallel 비효율**: 소규모 모델(12M)에선 단일 GPU + 대용량 배치가 3.3× 빠름
- **GPU 공존**: vLLM(93GB×2)과 학습 동시 진행 시 GPU1 free ~3.6GB. v3/v3_1 학습 VRAM ~1.6GB로 공존 가능
- **저빈도 합성 한계**: 폰트 합성(exp3)은 저빈도 개선에 미미 — 실데이터 증강 추가가 더 유효
- **학습 로그 acc는 추론 성능이 아님** (v3_1): argmax 로그 acc 43.1%였으나 beam search 실제 디코딩 30.4% — teacher forcing·자기생성 혼재. 최종 판단은 반드시 디코더 재평가
- **attention vs CTC 오류 특성** (v3_1): attention은 고빈도만 작동(54.8%)·중저빈도 붕괴(중3.2/저0.4) → CER 급증. CTC는 중저 빈도도 부분 일치 → CER 우위. "어떤 오류가 더 치명적인가" 고려 필요

## Git

```bash
cd /home/dev/DoctorOcr
git add -A && git commit -m "..." && git push origin main
```

- 저장소: SimCreate/DoctorOcr (main)
- 데이터(`data/`)·체크포인트(`working/`)·로그는 gitignore — 소스·문서·보고서만 커밋
