# DoctorOcr v3_1 개선 교차검증 결과 (2026-08-07)

**주제**: attention CRNN(30.4%)의 병목 = 인코더 82% 미인식, 개선 방법은?
**방법**: 내 실측(oracle 분석) + ChatGPT(웹검색) + 로컬 V4 Flash(독립 팩트체크)

## 📌 결론: 병목은 인코더, 답은 사전학습 인코더 + 손글씨 특화 증강 + 라벨 정제

### 두 LLM이 일치한 것 (신뢰 높음)
1. **병목 = encoder 표현력** (디코더/beam 아님) — 단, "인코더 단독 실패"라는 표현은 로컬 LLM이 지적, "인코더+디코더 결합"이 정확
2. **저빈도는 디코더/beam으로 해결 불가** — oracle 분석(저빈도 후보에도 없음) 근거로 두 LLM 모두 동의
3. **가장 강력한 단일 개선 = 사전학습 backbone** (ConvNeXt-Tiny / EfficientNetV2-S / Swin-Tiny / TrOCR encoder)
4. **손글씨·롱테일 특화**: IAM/Bentham/CVL(영어 손글씨)로 encoder 선학습 후 Rx 미세조정
5. **증강 부족**: 현재 rotation/scale/brightness/noise만 → Elastic distortion + stroke erosion/dilation이 손글씨에 핵심
6. **라벨 정제가 저비용 고효율**: Valix CR / Valix-CR / Valix CR 표기 통일, 대소문자·공백
7. **하드 예제 마이닝**: oracle=False 641장 oversampling (단, 라벨 오류 아닌 것만)

### ChatGPT vs 로컬 LLM 충돌 지점 (판단 필요)
| 항목 | ChatGPT | 로컬 LLM(V4 Flash) |
|---|---|---|
| oracle=False 해석 | "인코더가 다른 단어를 봄" (단정) | "인코더+디코더 결합 실패 — 단정은 오류" (수정) |
| beam/LM 가치 | "Decoder < LM < Beam ≪ Encoder" | "고빈도 16.8%p는 beam 선택 오류 — 무시 못함" |
| RIMES | 영어 손글씨 후보로 추천 | "RIMES는 프랑스어 — 아님" ⚠️ |
| CER 예상 | "38%→31~34% 보장" | "근거 없는 추정" |
| ViT | "훨씬 강력 권장" | "13K에선 과적합, CNN이 안정" |
| convnext 수정 난이도 | "기존 코드 수정 적음" | "입력/stride/loading 다 바꿔야 — 과장" |

### 실측 검증 필요 (아직 확인 못함)
- IAM, CVL, Bentham, TrOCR 사전학습 가중치 존재/라이선스 — 직접 확인 필요
- ConvNeXt/EfficientNet backbone 교체 시 입력 사이즈 제약
- 의사 손글씨는 1024 입력이 아니라 256x64 리사이즈 — 해상도 전처리부터 재검토 필요

## 🎯 최종 제안: 실용 개선 순서 (사용자 판단용)

### 0순위 (즉시 가능, 비용 ~0)
- **라벨 정제**: `diagnose_labels.py`가 이미 만들었던 시그널(3중복 글자 lll, 한글 섞임 hansol, 전소문) — 실제로 oracle=False 641장에 라벨 오류가 섞여 있는지 **육안 검토 100장**
- **전처리 검토**: 256x64 무조건 리사이즈 → 가로세로비 유지 패딩으로 변경 (+높이 48/64 실험)

### 1순위 (낮은 비용, 효과 큼)
- **사전학습 backbone 교체**: ConvNeXt-Tiny (ImageNet→Rx) — 기존 CRNN 뼈대 유지하며 encoder만 교체
- **손글씨 특화 증강**: Elastic distortion + stroke erosion/dilation 추가 (가장 손글씨다운 개선)

### 2순위 (중간 비용)
- **영어 손글씨 pretraining**: IAM 다운로드 → encoder만 선학습 → Rx fine-tune
- **하드 예제 마이닝**: oracle=False 중 라벨 정확한 것만 oversampling

### 3순위 (고비용, 장기)
- **multi-scale feature (FPN/BiFPN)**: 얇은 stroke 보존
- **CNN 뒤 2~4층 Transformer encoder**: 멀리 떨어진 stroke 연결
- **lexicon-constrained decoding**: 약물명 사전 → beam 후보 제한 (롱테일 근본 대응)

### 하지 말 것 (두 LLM 공통 합의)
- ~~beam search 넓히기~~ (저빈도에 무효)
- ~~Attention 디코더 고집~~ (CTC가 이미 더 좋음 — 단, 하이브리드 CTC+Attn은 후보)
- ~~디코더/LM 튜닝에 올인~~ (병목 아님)
