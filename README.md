# DoctorOcr

의사 손글씨 처방전 OCR — CRNN 계열 모델 학습·추론 파이프라인
Kaggle RxHandBD 데이터셋(5,578장) 기반, v1 → v2 → v3 → v4 스토리라인 (모노레포)

- 현재 메인: v4 (resnet18 + BiLSTM3 + 하이브리드 Attention/CTC, 라벨 정제 적용, 자립 패키지)
- 현재 상태: 클린 스플릿 기준 FAIL — 수용기준(고빈도 ≥90%, CER ≤20%) 미충족 (CTC greedy exact 48.2%, CER 24.3%)

## 스토리라인 (v1 → v4)

| 단계 | 디렉토리 | 핵심 내용 | 결과 (클린 val exact) |
|---|---|---|---|
| v1 — 시작 | `v1/` | 첫 CRNN (7CNN+BiLSTM2+Attention). 데이터 89장 부실 | 0% |
| v2 — 데이터 확장 + 아키텍처 개선 | `v2/` | 원본 5,578장 정리, BiLSTM 3층 + Attention 8-head + Beam | ~20% |
| v3 — 데이터 가공·증량 | `v3/` | 클린 split, 증강 → 13,386장, resnet18 백본 전환 | 35.8~37.9% |
| v4 — 하이브리드 (현재) ★ | `v4/` | resnet18 + Attention + CTC 이중헤드, 라벨 정제 | 62.3% (attn beam5) |

- v1→v2: 데이터셋을 89장 → 5,578장으로 확장하고 아키텍처를 개선
- v2→v3: 데이터 가공(클린 split·증강)으로 학습량 13,386장까지 확대, 백본을 resnet18로 전환
- v3→v4: CTC+Attention 하이브리드 구조로 전환 + 라벨 정제

## 프로젝트 구조

모노레포: `/home/dev/DoctorOcr/` (git 저장소)

```
DoctorOcr/
├── README.md                     # 상위 인덱스 (본 파일)
├── DoctorOcr_발표_20260807.pptx  # 발표 PPT (정적 산출물)
├── .gitignore
├── v1/                           # v1 — 시작 (첫 CRNN + Attention)
│   ├── local_train.py / local_infer.py
│   └── doctor_ocr_v1.md
├── v2/                           # v2 — 데이터 확장 + 아키텍처 개선
│   ├── local_train.py / local_infer.py
│   ├── PROJECT_LOG.md
│   └── v2_1/                     # v2.1 — BiLSTM3 + Attention 8-head + Beam (v2 하위)
├── v3/                           # v3 — 데이터 가공·증량 (resnet18 백본)
│   ├── DESIGN.md / README.md
│   ├── scripts/                  # train/eval_v3_1, train/eval_v3_2
│   ├── model/                    # model_v3_1, model_v3_2, preprocess_v3_2
│   ├── evaluate/                 # metrics·aggregate·acceptance
│   └── utils/result_viewer.py    # Streamlit 정성 분석 뷰어
├── v4/                           # v4 — ★ 현재 메인 (하이브리드 resnet18+Attention+CTC, 자립)
│   ├── DESIGN.md                 # 설계/가설/데이터 구성/학습 설정
│   ├── scripts/                  # train_v3_3.py(학습) · eval_v3_3.py(평가)
│   ├── evaluate/                 # metrics·aggregate·acceptance (공용 평가 모듈)
│   ├── model/                    # model_v3_3.py(CRNN+이중헤드) · preprocess_v3_3.py
│   └── utils/result_viewer.py    # Streamlit 정성 분석 뷰어
├── legacy/                       # ★ 보관 — CTC 부수 실험 — 상세는 하단 [Legacy 기록]
│   ├── doctor_ocr_v2_2/
│   └── doctor_ocr_v3/
└── docs/                         # 교차검증·발표용 기록
```

> 데이터(`data/`, `working/`, `logs/`)는 `.gitignore` — 저장소에는 소스·문서·보고서만.
> v1/v2/v3는 기록 보존 용도 (실행 원본은 DoctorOcr 밖 별도 repo에 존재). 현재 실행은 v4만.
> 원본(v2 dataset)은 절대 수정하지 않고, 실험군 데이터는 복사/증강/합성물만 사용.

## 결과 요약

현재 결과 (v4 = 메인, 하이브리드 + 라벨 정제, 클린 val 1,116장, v3_3e 체크포인트 정식 eval)

| 디코더 | exact | CER | 고빈도 | 중빈도 | 저빈도 |
|---|---|---|---|---|---|
| Attention beam5 | 62.3% | 22.9% | 83.9% | 55.1% | 0.9% |
| CTC greedy | 48.2% | 24.3% | 70.3% | 23.6% | 2.8% |

- Attention beam5 정식 eval로 62.3% 확정 (기존 "67.0%는 로그값" → 이제 beam5로 재측정. 학습 로그의 67.0%는 더 넓은 beam(200)이라 차이).
- 라벨 정제(소문자 통일·하이픈→공백)로 CTC greedy 45.5% → 48.2% (+2.7p). CER 25.8% → 24.3%.
- attention이 CTC보다 전반 우세: 전체 62.3 vs 48.2, 중빈도 55.1 vs 23.6. 고빈도 83.9%까지.
- 저빈도는 두 디코더 모두 붕괴 (0.9~2.8%) — 디코더가 아니라 데이터/라벨 문제로 확정.
- ⚠️ 라벨 정제로 빈도그룹 경계가 이동해, 이전 버전과 그룹별 수치는 직접 비교 불가 (전체 exact만 동일 val 기준 비교).
- 상세: `v4/DESIGN.md`

버전 계보 (v1 → v4 스토리라인)

인코더 구성

| 단계 | CNN 인코더 | 시퀀스 인코더(BiLSTM) | 데이터 (train) |
|---|---|---|---|
| v1 | 7 Conv Block (→512ch) | 2층, hidden 256 | 89장 |
| v2 | SEBlock 5블록 (→512ch) | 3층, hidden 384 | 5,578장 |
| v3 | v3_1: SEBlock → v3_2: resnet18 | 3층, hidden 384 | 13,386장 |
| v4 | resnet18 (ImageNet pretrained, layer3) | 3층, hidden 384 | 13,386장 (라벨 정제) |

디코더 구성

| 단계 | 디코더 | loss | exact | 주요 개선 |
|---|---|---|---|---|
| v1 | AttentionDecoder 4-head (LSTM1) | CE | 0% | 시작점. 데이터·평가 부실 |
| v2 | AttentionDecoder 8-head (LSTM2) + Beam | CE | ~20% | 데이터 확장 + 아키텍처 개선 |
| v3 | AttentionDecoder 8-head + Beam | CE | 30~35% | 클린 split·증강, 백본 전환 |
| v4 | AttentionDecoder 8-head + CTCHead (하이브리드) | λ·L_ctc + (1-λ)·β·L_ce | 48.2% (attn 62.3%) | CTC+Attention 하이브리드 |

## 아키텍처

현재 메인: v4 — 하이브리드 (resnet18 + Attention + CTC)

```
입력 [B, 3, 256, 128]  (비율유지 패딩 + ImageNet 정규화)
  │
  ▼ resnet18 layer1~3 (ImageNet pretrained)     ← stride 16, 특징맵 [B,256,16,8] (64열)
  ▼ BiLSTM 3층 (hidden 384)                     ← [B, 16, 768]
  │
  ├─▶ AttentionDecoder (8-head, LSTM2, beam5) → attn_logits [B, L, 73]
  └─▶ CTCHead (Linear → 70, blank=0)          → ctc_logits  [B, 16, 70]

손실:  L = λ·L_ctc + (1-λ)·β·L_ce     (최종: λ=0.3, β=3.0, label_smoothing=0.1)
```

- CTC 헤드: 인코더를 시각 정렬에 강하게 밀어줌 → 모든 디코더의 기반 (v2_2에서 검증된 구현 재사용)
- Attention: 언어 prior → 고빈도/중빈도 패턴 인식 (beam5 62.3%로 CTC 48.2%보다 우세)
- 공유 인코더: resnet18+BiLSTM이 두 손실 모두에 의해 "시각적으로 정확 + 언어적으로 그럴듯"하게 학습

### ⚠️ 시퀀스 시프트 버그 (이 프로젝트 최대 교훈)
- 증상: attention beam이 epoch 40까지 0% (v3_3/v3_3b/v3_3c)
- 원인: 학습 CE 정답이 `targets[:, :-1]` (입력과 동일) → 모델이 자기복사만 학습, `<SOS>`→`<SOS>` 출력
- 수정: 정답을 `targets[:, 1:]`로 한 칸 시프트 → attention 0% → 56.5% (전체 +11p)
- 교훈: attention(순차 모델)은 입력/정답 시프트 일치가 생명.

## 평가 지표

의료 처방전 OCR은 단어 단위 평가가 실제 활용에 맞다 — 이미지가 아니라 출력된 단어 전체가 정확한가를 본다.

| 지표 | 정의 | 용도 |
|---|---|---|
| exact match | 예측 단어 == 정답 단어 (전체 일치) | 기본 정확도. 단어 하나라도 틀리면 오답 |
| CER (Character Error Rate) | (Levenshtein 편집거리) ÷ 정답 글자 수 | "얼마나 가까웠는가" — `Napa`→`Npa`는 1/4=0.25, 완전 오답은 1.0 |
| WER (Word Error Rate) | 단어 단위 편집거리 ÷ 정답 단어 수 | 문장/여러 단어 처리 시 유용 (현 단어 데이터에선 보조) |
| 빈도그룹 (고/중/저) | 라벨 전체 빈도 기준: 고≥10회, 중 2~9회, 저 1회 | 롱테일 분석 — 고빈도만 잘 되고 저빈도가 무너지는지 확인 |
| 수용기준 (PASS/FAIL) | 고빈도 exact ≥90% AND 전체 CER ≤20% | 실사용 가능 여부의 게이트. 현재 전 실험 FAIL |

## 데이터셋

- 소스: Kaggle RxHandBD — 의사 처방전 손글씨 5,578장 (Train 5,000 + Test 400)
- 고유 문자: 73자 (알파벳, 숫자, 특수문자)
- 원본 정리·공용: v2부터 원본 5,578장을 `dataset/img/img/` + `combined_labels.csv`로 정리 (전 버전 공용)

라벨 정제 (v4, 2026-08-07)
- 변경 규모: val 1,116장 중 817장(73.1%), train 13,386건 중 10,152건(75.8%)
- 내용: 대문자→소문자 통일 (`Devixil`→`devixil`), 하이픈→공백 (`nalbun-1`→`nalbun 1`, `Coragen-D`→`coragen d`)
- 원본 보관: `v4/backups/val_ORIGINAL_20260807_170138.csv`, `combined_labels_ORIGINAL_20260807_170138.csv`
- 효과: exact +2.7p (45.5→48.2), CER -1.5p (25.8→24.3)

클린 split: v3부터 원본 80/20을 먼저 고정 분리 후 train에만 증강 — val 1,116장이 어떤 train에도 포함되지 않게 보장 (리키지 차단). v4는 이 클린 val 1,116 고정.

## 실행 방법

### v4 (메인) — 학습
```bash
cd /home/dev/DoctorOcr/v4
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_3.py   # 하이브리드 + 라벨 정제 (백그라운드 권장)
```

### v4 — 평가 (CTC greedy, val 1,116)
```bash
cd /home/dev/DoctorOcr/v4
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/eval_v3_3.py \
  --ckpt working/checkpoints/best_model_v3_3.pth --out evaluate/result_v3_3e_clean.csv
```

### v4 — 결과 뷰어 (Streamlit, 정성 분석)
```bash
cd /home/dev/DoctorOcr/v4
venv/bin/python -m streamlit run utils/result_viewer.py
```

- 발표 PPT: `/home/dev/DoctorOcr/DoctorOcr_발표_20260807.pptx` (정적 산출물)
- v4 학습 산출물: best `working/checkpoints/best_model_v3_3.pth` (epoch/optimizer/config 포함), 최신 실험 `best_model_v3_3e.pth` 보관
- 문자 사전: `working/char_dict_v3_3.pkl` (char2idx, idx2char)

## 알려진 이슈 / 교훈

- 저빈도 붕괴가 핵심 한계: attention 0.9% / CTC 2.8%. 디코더/beam이 아니라 데이터/라벨/외부 손글씨 문제. 다음 단계 핵심 타깃
- 빈도그룹 경계 이동: 라벨 정제로 train 빈도가 바뀌어 이전 버전과 그룹별 비교 불가 (전체 exact만 비교)
- 시퀀스 시프트 버그: attention의 입력/정답 시프트 일치가 생명 (위 상세)
- 학습 로그 acc ≠ 실제 디코딩: 최종 판단은 반드시 디코더 재평가 (v3에서 검증된 교훈)

## Legacy 기록

### 보관 트리 (v1/v2/v3)
- `v1/` — 시작 CRNN (실학습 89장 한계) — 기록 보존
- `v2/` — 데이터 5,578장 + 아키텍처 개선 — v2/v2_1 포함
- `v3/` — 클린 split·증강 → 13,386장, resnet18 백본 — v3_1/v3_2 통합본
- `legacy/doctor_ocr_v2_2` (v2.2, CTC 기반), `legacy/doctor_ocr_v3` (v3, CTC 데이터 실험) — 2026-08-06 보관

v3 실험군 결과 (클린 스플릿 재실험, 2026-08-05)

> ⚠️ 8/4 1차 실험의 "98.8% 원본 val 공정비교"는 이미지 단위 리키지(증강 포함 전체 split, val 상당수가 train에 포함)로 부풀려진 수치 → 폐기.
> 클린 split 기준 재측정 결과 전 실험군 FAIL.

| 실험군 | 데이터 구성 | 원본 val 공정비교 (클린) | 판정 |
|---|---|---|---|
| exp1_clean | 원본만 (4,462) | 35.8% (고56.3/중28.0/저7.6, CER 22.1%) | FAIL |
| exp2_clean | +실사 증강 2배 (13,386) | 37.9% (고56.5/중33.5/저8.0, CER 20.6%) | FAIL |
| exp3_clean | +합성 12.5% (13,943) | 36.4% (고54.2/중30.0/저11.4, CER 20.7%) | FAIL |

- 8/4 "증강 = 압도적 효과(+63p)"는 리키지 착시. 클린 기준 증강 효과는 +2.2p. 저빈도 합성은 -1.5p.
- v2.2(CTC) 도입 배경: v2.1 Attention의 `Deliiiii` 반복 토큰 실패 → CTC로 해결 (추론 40%)

## Git

```bash
cd /home/dev/DoctorOcr
git add -A && git commit -m "..." && git push origin main
```

- 저장소: SimCreate/DoctorOcr (main)
- 데이터(`data/`)·체크포인트(`working/`)·로그는 gitignore — 소스·문서·보고서만 커밋
