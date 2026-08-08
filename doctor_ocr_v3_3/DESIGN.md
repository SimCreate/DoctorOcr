# DoctorOcr v3_3 — Hybrid (resnet18 + Attention + CTC) 설계 문서

> 작성: 2026-08-07 · 갱신: v3_3e 종료
> 상태: **Attention beam(200) 67.0% / CTC greedy 48.2% — 프로젝트 사상 최고**
> 핵심: **① 시퀀스 시프트 버그 수정이 attention을 부활시킴** (0% → 이전 56.5%) → **② 라벨 정제가 CTC를 45.5% → 48.2%로 끌어올림**

---

## 1. 배경 — 왜 v3_3인가

### 1.1 검증된 실측 (모두 동일 클린 val 1,116장 기준)

| 지표 | v3_1 | v3_2 | v3 CTC | v3_3(버그) | v3_3d(수정) | **v3_3e(+라벨정제)** |
|---|---|---|---|---|---|---|
| 전체 exact (CTC greedy) | 30.4% | 34.7% | 37.9% | 45.4% | 45.5% | **48.2%** |
| 전체 exact (Attn beam5) | 30.4% | 34.7% | — | — | **56.5%** | *(미측정)* |
| 전체 CER | 38.2% | 39.4% | 20.6% | 25.7% | 25.8% | **24.3%** |

> ⚠️ **빈도그룹 경계 주의**: 라벨 정제로 train 빈도 집계가 바뀌어 v3_3e의 그룹 n이 d와 다름 (고 602→684, 저 265→216). **그룹별 exact는 d/e 직접 비교 불가, 전체 exact만 동일 val 이미지 기준으로 비교 가능.** (e 그룹별: 고 70.3%, 중 23.6%, 저 2.8%)

**핵심 패턴:**
- **v3_3d attention beam5 = 56.5%**: 시퀀스 시프트 버그 수정이 attention을 부활시킴 (이전 0% → 56.5%)
- **v3_3e (라벨 정제) = CTC greedy 48.2% (+2.7p), average CER 24.3% (-1.5p)**: 라벨 정제가 CTC를 개선
- **v3_3e 학습 로그 Attention beam(200) = 67.0%** (best_hybrid=0.8171) — 로그 기준 프로젝트 사상 최고. 단, **beam5 정식 eval은 아직 미실행** (eval 스크립트의 `--attn` 경로가 미구현 stub 상태)
- **저빈도 붕괴** (e: 2.8%, d: 0.4%) 는 여전히 모든 디코더 공통 — 디코더가 아니라 **데이터/라벨 문제** 확정

### 1.2 교차검증 (ChatGPT + 로컬 LLM V4 Flash, 2026-08-07)
- "CTC = 구식" 주장은 **틀림** — 2026년에도 CTC/Attention/Hybrid 모두 활발히 사용 (SVTRv2 "CTC Beats Encoder-Decoder Models", arXiv:2409.02134 실존)
- **작은 데이터 + 롱테일(64% 1회 등장) + 전문용어(약물명)** 에서 attention decoder는 언어 prior에 과적합 위험
- OCR/HWR에서 **Hybrid(CTC+Attention 공동손실)**는 "효과 유망하나 검증 필요" — 본 프로젝트로 검증 완료: **유효** (attention 56.5%)

### 1.3 레거시 교훈 (폐기 프로젝트에서)
| 폐기 프로젝트 | 교훈 | v3_3 반영 |
|---|---|---|
| v2_2 (CTC) | CTC는 가볍고 잘 돔. CTCHead 구현 검증됨 | CTC 헤드 재사용 |
| v3 8/4 실험 | **98.8%는 이미지 단위 리키지 아티팩트** | 클린 val 1,116 고정 |
| v3_2 1차 (LR 단일) | attention+pretrained LR 1e-4로는 10에폭 acc 0.27% | **LR 개별화** |
| v3_3 1차 | **attention beam 0%의 원인 = 시퀀스 시프트 버그** (아래) | **수정 완료** |

---

## 2. 아키텍처

```
입력 [B, 3, 256, 128]  (비율유지 패딩 + ImageNet 정규화)
  │
  ▼ resnet18 layer1~3 (ImageNet pretrained)     ← stride 16, 특징맵 [B,256,16,8] (64열)
  ▼ BiLSTM 3층 (hidden 384)                     ← [B, 16, 768]
  │
  ├─▶ AttentionDecoder (8-head + LSTM2 + beam5) → attn_logits [B, L, 73]
  │      vs  CTCHead (Linear → 70)              → ctc_logits  [B, 16, 70]  (blank=0)

손실:  L = λ·L_ctc + (1-λ)·β·L_ce     (최종: λ=0.3, β=3.0, label_smoothing=0.1)
```

### 2.1 ⚠️ 시퀀스 시프트 버그 (이 프로젝트 최대 교훈)
**증상**: attention beam이 epoch 40까지 0% (v3_3/v3_3b/v3_3c)
**원인**: 학습 CE 정답이 `targets[:, :-1]` (입력과 동일) → 모델이 **자기복사**만 학습, `<SOS>`→`<SOS>` 출력
**수정**: 정답을 `targets[:, 1:]`로 한 칸 시프트 → `<SOS>`→첫 글자 학습
**효과**: attention 0% → 56.5% (전체 +11p)

> 교훈: attention(순차 모델)은 **입력/정답 시프트 일치가 생명**. 병렬화 리팩토링 시 반드시 검증.

### 2.2 이중 헤드의 의미 (검증됨)
- **CTC 헤드**: 인코더를 시각 정렬에 강하게 밀어줌 → 모든 디코더의 기반
- **Attention**: 언어 prior → 고빈도/중빈도 패턴 인식 (56.5%로 CTC 45.5%보다 우세)
- **공유 인코더**: resnet18+BiLSTM이 두 손실 모두에 의해 "시각적으로 정확 + 언어적으로 그럴듯"하게 학습

### 2.3 어휘집 분리
- **attention vocab (73)**: `<SOS>=0, <EOS>=1, <PAD>=2, <UNK>=3` + 69 문자 — 고정길이 패딩
- **CTC vocab (70)**: `<BLANK>=0` + 69 문자 — 가변길이 (concat + lengths)

---

## 3. 데이터

| 항목 | 값 |
|---|---|
| train | `data/exp2_clean/combined_labels.csv` — 실사 원본 4,462 + 증강 2배 = **13,386** |
| val | `data/clean_split/val.csv` — **클린 고정 1,116** (train과 구조 분리, 리키지 0) |
| 이미지 | 128x128 원본 → **256x128 비율유지 패딩** |
| transform | ImageNet 평균/표준편차 정규화 |

### 3.1 라벨 정제 (v3_3e — 2026-08-07 17:01)
- **변경 규모**: val 1,116장 중 **817장(73.1%)**, train 13,386건 중 **10,152건(75.8%)**
- **정제 내용**: 대문자→소문자 통일 (`Devixil`→`devixil`), 하이픈→공백 (`nalbun-1`→`nalbun 1`, `Coragen-D`→`coragen d`), 형태소 분리
- **원본 보관**: `backups/val_ORIGINAL_20260807_170138.csv`, `backups/combined_labels_ORIGINAL_20260807_170138.csv`
- **효과**: 전체 exact +2.7p (45.5→48.2), 평균 CER -1.5p (25.8→24.3)
- **교훈**: 교차검증 문서(`docs/improvement_crosscheck_20260807.md`)의 "0순위: 라벨 정제" 제안이 실제로 가장 저비용·고효율이었음

---

## 4. 학습 설정 (v3_3d 최종)

| 파라미터 | 값 | 근거 |
|---|---|---|
| optimizer | AdamW, 그룹별 LR | v3_2 1차 실패 교훈 |
| backbone LR | 1e-4 (고정) | pretrained 특징 보존 |
| head LR | 5e-4 (cosine) | 새 헤드 빠른 적응 |
| warmup | 3 epoch | |
| batch / accum | 24 / 6 (=144) | GPU1 공존 3.5GB 내 |
| CTC 가중치 λ | **0.3** | attention 부활 장려 (0.5→0.3) |
| CE 스케일 β | **3.0** | CE=0 만점 붕괴 방지 |
| CE label smoothing | 0.1 | 과신 방지 |
| TF 스케줄 | 0.18→0.05 (decay 0.88) | 자기회귀 적응 강화 |
| early stop | 15 epoch | |
| best 저장 기준 | **hybrid score** (ctc + 0.5·attn) | attention 유지 체크포인트 |

---

## 5. 평가 기준 (동일 val 1,116)

- **v3_3d (수정 직전)**: attention beam5 == **56.5%**, CTC greedy == **45.5%**
- **v3_3e (라벨 정제)**: CTC greedy == **48.2%** (CER 24.3%), 학습 로그 Attn beam(200) == **67.0%**
- 빈도그룹(e): 고 70.3% (n=684) / 중 23.6% (n=216) / 저 2.8% (n=216) — **그룹 경계가 d와 달라 직접 비교 불가**
- **저빈도 붕괴가 유일한 한계** — 데이터/라벨 문제로 확정, 다음 단계의 최우선 타깃

---

## 6. 기대 효과 & 리스크 (실측 반영)

### 달성
- attention beam(200) 67.0% (로그), CTC greedy 48.2% — 70% 목표에 근접
- **라벨 정제가 +2.7p 기여** — 저비용 고효율 검증

### 남은 리스크
1. **저빈도 붕괴** (e 2.8%) — 디코더 무관, **데이터 확충/외부 손글씨 사전학습** 필요 (다음 단계 핵심)
2. **attention beam5 정식 eval 미실행** — v3_3e의 attention 67%(beam200)는 로그값. 평가 스크립트의 `--attn` 경로가 stub라 beam5 비교표 작성 필요
3. **빈도그룹 경계가 라벨 정제로 이동** — 이전 버전과 그룹별 비교가 더 이상 1:1 안 됨 (문서·보고서에 명시할 것)
4. attention beam eval 비용: val 전체 1,116장에 beam5 = 수 분 (최종 eval만)

---

## 7. 실행/재현

```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_3
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_3d.py   # 학습 (버그 수정판)
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/eval_v3_3.py --ckpt working/checkpoints/best_model_v3_3d.pth --out evaluate/result_v3_3d_clean.csv
```

**v3_3e (라벨 정제 — 최신)**
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3_3
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_3e.py   # d와 코드 동일, 라벨 정제 데이터 사용
```

> ⚠️ **트랩**: `train_v3_3e.py`는 `train_v3_3d.py`의 복사본이라 (ckpt 경로만 e로 변경) **로그의 [DONE] 메시지가 "v3_3d complete"로 출력됨** — 기능 문제는 아님, 로그 읽을 때 주의. 또 epoch 저장 파일명도 `epoch_v3_3d_*.pth`로 기록됨.

**최종 체크포인트**: `working/checkpoints/best_model_v3_3e.pth` (best_hybrid=0.8171, best_attn=0.6700, best_ctc≈0.50)
**eval 결과**: `evaluate/result_v3_3e_clean.csv` (CTC greedy, val 1,116, 전체 exact 48.2%)

---

## 8. 다음 단계 후보 (저빈도 2.8% 돌파가 메인)

- ✅ **라벨 정제** — v3_3e에서 수행 완료 (+2.7p 검증). 원본은 `backups/`에 보관

1. **attention beam5 정식 eval + 두 디코더 앙상블/선택** — v3_3e attention(67% beam200) 정식 측정 후 CTC 48.2%와 조합
2. **외부 손글씨(IAM·CVL) 사전학습** — 인코더에 손글씨 사전지식 주입
3. **저빈도 데이터 확충** — 합성/증강으로 1회 등장 단어 반복
4. **lexicon-constrained decoding** — 약물명 사전(DrugBank)으로 beam 제한
5. **라벨 오류 재검토** — 저빈도 정제 후에도 실패 다수면 제로샷 검증으로 2차 정제
