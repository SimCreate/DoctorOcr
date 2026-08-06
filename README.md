# DoctorOcr

의사 손글씨 처방전 OCR — CRNN 계열 모델 학습·추론 파이프라인
Kaggle RxHandBD 데이터셋(5,578장) 기반, v1 → v3_1 순차 실험 (모노레포)

- 현재 메인: **doctor_ocr_v3_1** (attention 디코더 + 증강 데이터, 자립 패키지)
- **현재 상태**: 클린 스플릿 기준 **FAIL** — 수용기준(고빈도 ≥90%, CER ≤20%) 미충족 (exact 30.4%, CER 38.2%)

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
│   ├── evaluate/                 # 평가 레이어 (기본 지표 + v3_1 고유 확장)
│   │   ├── metrics.py / aggregate.py / acceptance.py   # + beam oracle / 반복토큰
│   │   └── result_*.csv          #   평가 결과
│   ├── utils/result_viewer.py    #   Streamlit 정성 분석 뷰어 (v3_1 고유)
│   ├── model/model_v2_1.py       # v2.1 모델 + beam 후보 반환 확장 (v3_1 고유)
│   ├── data/                     # exp2_clean + clean_split (로컬 복사본)
│   ├── reports/                  # 평가 보고서
│   └── venv/                     # 전용 venv (torch 2.13.0+cu132)
└── legacy/                       # ★ 보관 — 상세는 하단 [Legacy 기록]
    ├── doctor_ocr_v2_2/
    └── doctor_ocr_v3/
```

> 데이터(`data/`, `working/`, `logs/`)는 `.gitignore` — 저장소에는 소스·문서·보고서만.
> 원본(v2 dataset)은 절대 수정하지 않고, 실험군 데이터는 복사/증강/합성물만 사용.

## 결과 요약

**현재 결과 (v3_1 — attention + 증강 데이터, 클린 val 1,116장)**

| 방식 | exact | CER | 고빈도 | 중빈도 | 저빈도 | 판정 |
|---|---|---|---|---|---|---|
| v3_1 (Attention, beam=5) | 30.4% | 38.2% | 54.8% | 3.2% | 0.4% | FAIL |

- 고빈도 단어는 그럭저럭 인식하나(54.8%), 중·저빈도 붕괴 (중 3.2% / 저 0.4%)로 CER이 높음.
- 학습 로그 acc(43.1%) ≠ 실제 디코딩(30.4%) — 평가는 항상 디코더 재측정으로 판단.
- **beam oracle** (top-5 후보에 정답 포함): 42.6% — top-1보다 +12.2p. 고빈도는 후보까지 72.3%지만, 중·저빈도는 후보에도 없음 (인코더가 못 읽음).
- 수용기준(고빈도 ≥90%, CER ≤20%) 미충족. 통과하려면 **라벨 오류 정제**나 아키텍처 개선 우선.
- 상세: `doctor_ocr_v3_1/reports/eval_v3_1_20260806.md`

**버전 계보 (활성: v1 → v2 → v2.1 → v3_1)**

| 항목 | v1 | v2 | v2.1 | v3_1 |
|---|---|---|---|---|
| 아키텍처 | CRNN (7CNN+BiLSTM2+Attn) | 개선 CRNN | BiLSTM 3층+Attention | v2.1 + 증강 데이터 |
| 손실 함수 | CrossEntropy | CrossEntropy | CrossEntropy | CrossEntropy (Attention) |
| 추론 정확도 (클린) | 0% | 20% | 20% | 30.4% (beam=5) |
| 상태 | 활성 | 활성 | 활성 | **★ 메인** |

## 아키텍처 진화

활성 버전들(v1 → v2 → v2.1 → v3_1)은 CRNN 3컴포넌트(CNN Encoder + BiLSTM + Decoder) 골격을 유지하되, 디코더와 학습 전략이 단계적으로 개선:

| 버전 | CNN Encoder | BiLSTM | Decoder | 개선 포인트 |
|---|---|---|---|---|
| v1 | 7 Conv Block | 2층, hidden 256, dropout 0.2 | Attention (4-head, LSTM 1층), TF 0.5 | 기준선 |
| v2 | 7 Conv Block | 2층, hidden 256 | Attention (4-head), TF 0.5 | BATCH 96 + AMP |
| v2.1 | SEBlock 5블록 | 3층, hidden 384, dropout 0.3 | Attention (8-head, LSTM 2층) + Beam Search | SEBlock + 층수 증가 + 8-head |
| v3_1 | v2.1 동일 (SEBlock) | v2.1 동일 (3층, hidden 384) | **Attention 8-head + Beam Search (22.7M)** | 증강 데이터로 attention 재검증 |

## 참고 논문

| 기술 | 논문 | 링크 |
|---|---|---|
| CRNN | An End-to-End Trainable Neural Network for Image-based Sequence Recognition (Shi et al., 2015) | https://arxiv.org/abs/1507.05717 |
| SE Block | Squeeze-and-Excitation Networks (Hu et al., 2017) | https://arxiv.org/abs/1709.01507 |

## 평가 지표

v3_1의 평가 레이어(`evaluate/`)가 사용하는 지표. 의료 처방전 OCR은 단어 단위 평가가 실제 활용에 맞다 — 이미지가 아니라 **출력된 단어 전체가 정확한가**를 본다.

| 지표 | 정의 | 용도 |
|---|---|---|
| **exact match** | 예측 단어 == 정답 단어 (전체 일치) | 기본 정확도. 단어 하나라도 틀리면 오답 |
| **CER** (Character Error Rate) | (Levenshtein 편집거리) ÷ 정답 글자 수 | "얼마나 가까웠는가" — `Napa`→`Npa`는 1/4=0.25, 완전 오답은 1.0 |
| **WER** (Word Error Rate) | 단어 단위 편집거리 ÷ 정답 단어 수 | 문장/여러 단어 처리 시 유용 (현 단어 데이터에선 보조) |
| **빈도그룹** (고/중/저) | 라벨 전체 빈도 기준: 고≥10회, 중 2~9회, 저 1회 | 롱테일 분석 — 고빈도만 잘 되고 저빈도가 무너지는지 확인 |
| **수용기준** (PASS/FAIL) | 고빈도 exact ≥90% AND 전체 CER ≤20% | 실사용 가능 여부의 게이트. **현재 전 실험 FAIL** |

**평가 방식의 계보** (코드 구조):
- v1/v2/v2.1: `local_infer*.py` 안에 인라인 — "pred == true" 정확도만 계산 (모듈 없음)
- **v3_1: `evaluate/` 폴더 모듈화** — exact/CER/WER/빈도그룹/수용기준 + **beam oracle** + **반복토큰(repetition)**

**v3_1 고유 확장 지표** (attention beam search 특성 분석):

| 지표 | 정의 | 현재 결과 |
|---|---|---|
| **beam oracle** | top-k 후보(beam)에 정답 포함 여부 | 42.6% (top-1 30.4%보다 +12.2p) |
| **반복토큰** | 연속 동일문자 ≥3 (예: `Raviiil`) | 1.8% (20/1116) |

- beam oracle은 "top-1은 틀렸지만 후보엔 정답이 있는" 케이스(12.2%)를 드러내어, 실패가 디코딩(순위) 문제인지 인코더가 못 읽은 문제인지 구분하게 한다. 현재 중·저빈도는 후보에도 없어 **인코더 문제**로 판정.
- 반복토큰은 attention 디코더의 전형적 mode collapse(`Raviiil`식)를 탐지 — v3_1에선 1.8%로 낮은 수준.

## 데이터셋

- **소스**: Kaggle RxHandBD — 의사 처방전 손글씨 5,578장 (Train 5,000 + Test 400)
- **고유 문자**: 73자 (알파벳, 숫자, 특수문자)
- **원본 정리·공용**: v2부터 원본 5,578장을 `dataset/img/img/` + `combined_labels.csv`로 정리 (v2/v2.1/v2.2/v3_1 공용)

**버전별 데이터셋 차이**:

| 구분 | v1 | v2 / v2.1 / v2.2 | v3_1 |
|---|---|---|---|
| **실학습 데이터** | **89장** | 5,578장 (원본 정리) | 13,386장 |
| 이미지 디렉토리 | 4,769장 (오염·무라벨 포함) | 5,578장 | 13,386장 |
| 라벨 파일 | `doctor_handwriting_labels.csv` | `combined_labels.csv` | `exp2_clean/` |
| split | random_split (80/20) | random_split | **클린 고정 split** |
| train / val | 71 / **18장** | ~4,462 / ~1,116장 | **13,386 / 1,116장** |

- **v1 부실 원인**: 이미지 디렉토리에 4,769장이 있었지만, 학습이 참조한 라벨 CSV(`doctor_handwriting_labels.csv`)는 **89장만 매핑** — 즉 실학습 89장 (train 71 / val 18). 여기에 Windows 중복(`- Copy.jpg`), 휴지통(`.trashed-`), `rxhandbd/train·test` 하위폴더가 마구 섞인 채 random_split이라 val 18장으론 평가 신뢰도가 사실상 없었다.
- **v2에서 정비**: 원본 5,578장 전체를 `combined_labels.csv`로 정리·통일 → 5,578장으로 재학습 (v2/v2.1/v2.2/v3_1 공용 원본).
- **v2 → v3_1 (증강)**: v3 실험에서 원본 4,462장(clean train)에 실사 증강 2배(회전±5°, 스케일, 밝기, 노이즈)를 더해 `exp2_clean` 13,386장. v3_1은 이 **전체(13,386)를 train**으로 사용. 원본만 쓴 `exp1_clean`과 대조해 증강 효과(+2.2p) 측정 — 상세는 [Legacy 기록](#legacy-기록).
- **클린 split**: v3_1부터 원본 80/20을 먼저 고정 분리 후 train에만 증강 — val 1,116장이 어떤 train에도 포함되지 않게 보장 (리키지 차단).
- 원본 데이터/라벨 CSV는 `.gitignore` 제외 (용량 문제)

## 실행 방법

> 스크립트 내 경로는 라이브 디렉토리(`/home/dev/DoctorOcr/`) 기준.

### v3_1 (메인) — 학습
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_1
# attention 디코더 (v2.1) + exp2_clean(증강 13,386장), 백그라운드 권장
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

- **v3_1 학습 산출물**: val_loss 최소 시 `working/checkpoints/best_model.pth` 저장 (epoch/optimizer/config 포함)
- 문자 사전: `working/char_dict.pkl` (char2idx, idx2char)

## 알려진 이슈 / 교훈

- **v2.1 체크포인트 config 버그**: 학습 시 `hidden_size=384`인데 config엔 `256` 저장 — 추론 시 강제 우회 (v3_1에선 수정됨)
- **DataParallel 비효율**: 소규모 모델에선 단일 GPU + 대용량 배치가 더 빠름
- **학습 로그 acc는 추론 성능이 아님** (v3_1): argmax 로그 acc 43.1%였으나 beam search 실제 디코딩 30.4% — teacher forcing·자기생성 혼재. 최종 판단은 반드시 디코더 재평가
- **저빈도 붕괴** (v3_1): attention은 고빈도만 작동(54.8%)·중저빈도 급락(중3.2/저0.4) → CER 상승. 중심 단어 생성 강화 필요

## Legacy 기록

`legacy/doctor_ocr_v2_2` (v2.2, CTC 기반 CRNN), `legacy/doctor_ocr_v3` (v3, CTC 데이터 실험) — 2026-08-06 보관.

**v3 실험군 결과 (클린 스플릿 재실험, 2026-08-05)**

> ⚠️ 8/4 1차 실험의 "98.8% 원본 val 공정비교"는 이미지 단위 리키지(증강 포함 전체 split, val 상당수가 train에 포함)로 부풀려진 수치 → **폐기**.
> 클린 split(원본 80/20 → train에만 증강/합성, val과 구조적 분리) 기준 재측정 결과 전 실험군 FAIL.

| 실험군 | 데이터 구성 | 원본 val 공정비교 (클린) | 판정 |
|---|---|---|---|
| exp1_clean | 원본만 (4,462) | **35.8%** (고56.3/중28.0/저7.6, CER 22.1%) | FAIL |
| exp2_clean | +실사 증강 2배 (13,386) | **37.9%** (고56.5/중33.5/저8.0, CER 20.6%) | FAIL |
| exp3_clean | +합성 12.5% (13,943) | **36.4%** (고54.2/중30.0/저11.4, CER 20.7%) | FAIL |

- 8/4 "증강 = 압도적 효과(+63p)"는 리키지 착시. 클린 기준 증강 효과는 **+2.2p**. 저빈도 합성은 -1.5p.
- v2.2(CTC) 도입 배경: v2.1 Attention의 `Deliiiii` 반복 토큰 실패 → CTC로 해결 (추론 40%)
- 실행: `legacy/doctor_ocr_v3` → `scripts/train_exp.py` · `evaluate/run_eval.py`
- 데이터셋 재생성: `legacy/doctor_ocr_v3/DATASETS.md` · 상세 보고: `legacy/doctor_ocr_v3/reports/`

## Git

```bash
cd /home/dev/DoctorOcr
git add -A && git commit -m "..." && git push origin main
```

- 저장소: SimCreate/DoctorOcr (main)
- 데이터(`data/`)·체크포인트(`working/`)·로그는 gitignore — 소스·문서·보고서만 커밋
