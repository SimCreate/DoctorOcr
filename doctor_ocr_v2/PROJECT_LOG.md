# doctor_ocr_v2 프로젝트 작업 로그

## 프로젝트 개요
- **목적**: 의사 손글씨 OCR 모델 학습 및 추론 파이프라인 구축
- **데이터**: RxHandBD 데이터셋 (Kaggle) - 5,578장 이미지 + 라벨
- **모델**: CRNN (CNN 7블록 + BiLSTM 2층 + Attention Decoder)
- **하드웨어**: 2× RTX PRO 6000 Blackwell (192GB VRAM), 단일 GPU 사용
- **환경**: `/home/dev/doctor_ocr_v2/`, Python venv (torch cu128)

---

## 작업 타임라인

### 2025-07-25: 데이터셋 준비 및 파이프라인 구축

#### 1. 데이터 다운로드 및 구조화
```bash
# Kaggle에서 RxHandBD 다운로드
kaggle datasets download -d fareswaleed/rxhandbd-dataset -p /home/dev/doctor_ocr_v2/
unzip rxhandbd-dataset.zip
```

**데이터 통계**:
| 구분 | 개수 |
|------|------|
| Train 이미지 | 5,000장 (P1116.jpg ~ P5578.jpg) |
| Test 이미지 | 400장 (P0001.jpg ~ P0404.jpg) |
| **총 이미지** | **5,578장** |
| 라벨 행 수 | 5,578행 (헤더 제외) |
| 고유 문자 | 73자 (알파벳, 숫자, 특수문자) |

**폴더 구조** (doctor_ocr 원본과 동일하게 맞춤):
```
/home/dev/doctor_ocr_v2/
├── dataset/
│   ├── img/img/                    # 5,578장 이미지
│   ├── combined_labels.csv         # 전체 라벨 (filename,label)
│   └── doctor_handwriting_labels.csv  # 호환용 복사본
├── working/
│   └── checkpoints/                # 모델 체크포인트
├── local_train.py                  # 학습 스크립트
└── local_infer.py                  # 추론 스크립트
```

#### 2. 학습/추론 스크립트 이식 및 수정
- `doctor_ocr/local_train.py` → `doctor_ocr_v2/local_train.py` (경로만 변경)
- `doctor_ocr/local_infer.py` → `doctor_ocr_v2/local_infer.py` (경로만 변경)
- 모델 아키텍처: CRNN (CNN + BiLSTM + Attention Decoder) 동일 유지

---

### 2025-07-25: 1차 학습 (DataParallel 시도)

#### 설정
| 파라미터 | 값 |
|----------|-----|
| Batch Size | 8 |
| Accum Steps | 2 (유효 배치 16) |
| Num Workers | 2 |
| GPU 모드 | DataParallel (2 GPU) |
| Epochs | 17까지 진행 후 타임아웃 중단 |

#### 결과
- **속도**: ~10 it/s (DataParallel 오버헤드로 단일 GPU보다 느림)
- **Best (Epoch 16)**: Val Loss 2.5587, Val Acc 2.51%
- **문제**: DataParallel이 소규모 모델(12M 파라미터)에 비효율적

---

### 2025-07-25: 2차 학습 (단일 GPU + 대용량 배치)

#### 설정 변경
| 파라미터 | 이전 | 변경 |
|----------|------|------|
| **Batch Size** | 8 | **32** (4× 증가) |
| **Accum Steps** | 2 | **1** (누적 제거) |
| **유효 배치** | 16 | **32** (2× 증가) |
| **Num Workers** | 2 | **4** |
| **GPU 모드** | DataParallel | **단일 GPU** |

#### 학습 진행
- **속도**: ~14.5 it/s (일정 유지)
- **에포크당 시간**: ~10초 (기존 33초 → 3.3× 빠름)
- **총 학습 시간**: 50 에포크 ≈ **8분 30초**

#### 성능 추이
| 에포크 | Val Loss | Val Acc | 비고 |
|--------|----------|---------|------|
| 16 (1차) | 2.5587 | 2.51% | DataParallel 중단 시점 |
| 35 (중간) | 2.1977 | 5.11% | 2차 학습 중 갱신 |
| **50 (최종)** | **2.1499** | **6.18%** | **Best 모델** |

**Val Loss 지속 감소, Acc 2.5% → 6.2%로 2.5배 향상**

---

### 2025-07-25: 추론 테스트 및 분석

#### 검증 세트 정확도 (1,116 샘플)
- **Exact Match**: **6.18%** (69/1116)

#### 랜덤 샘플 테스트
| 테스트 | Exact Match |
|--------|-------------|
| Random 10샘플 #1 | 0% (0/10) |
| Random 10샘플 #2 | 10% (1/10) |
| 순차 50샘플 | 8% (4/50) |

#### 문자 단위 분석 (처음 50개)
```
Dexter → Dextr          (67%)  ✓ 거의 맞음
Clavusef sw → Calbra ll (9%)   ✗ 완전 다름
m-lucas 10 → T-lca 0    (30%)  ✗ 숫자/특수문자 약함
Dophyllin → Deliiiii    (22%)  ✗ 반복 패턴 생성
Dominos → Dexionn       (29%)  ✗
Tablet → Tablet         (100%) ✓ 완벽
```

**패턴**: 짧은 단일 단어("Dexter", "Tablet")는 잘 맞음. 복합/긴 라벨은 실패.

---

## 문제 진단

| 원인 | 증거 |
|------|------|
| **Attention Decoder 불안정** | Teacher forcing 0.5인데도 반복 토큰 생성 (`Deliiiii`, `lll`) |
| **데이터 부족** | 5,578장 중 Train 4,462장 → 클래스당 샘플 적음 |
| **Augmentation 없음** | 회전/노이즈/밝기 변형 없이 원본만 학습 |
| **CTC Loss 미사용** | Attention seq2seq보다 CTC가 OCR에 더 안정적 |
| **Vocab 크기** | 73자 (특수문자 포함) - 디코더 복잡도 높음 |

---

## 체크포인트 현황

```
working/checkpoints/
├── best_model.pth      # Epoch 50 (val_loss=2.1499, acc=6.18%) ← 최종 베스트
├── epoch_10.pth
├── epoch_20.pth
├── epoch_30.pth
└── epoch_50.pth        # 최종 에포크 저장
```

---

## 다음 단계 계획

### 즉시 개선 사항 (우선순위 순)

#### 1. CTC Loss 도입 (가장 중요)
```python
# Attention Decoder → CTC Head 교체
# 장점: alignment 불필요, 반복 토큰 문제 해결, OCR 표준 방식
# 구현: nn.CTCLoss() + Linear projection to vocab
```

#### 2. Data Augmentation 추가
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize((0.5,), (0.5,))
])
```

#### 3. Pretrained Backbone 교체
- 현재: 7-block CNN (scratch)
- 변경: ResNet18/34 (ImageNet pretrained) → feature extractor로 사용
- 기대효과: 초기 feature 품질 향상, 수렴 속도 개선

#### 4. 학습 설정 튜닝
- LR warmup (첫 5 에포크 linear warmup)
- Cosine annealing 유지
- Epoch 100~200으로 연장
- Batch size 64 시도 (VRAM 8GB 여유)

#### 5. 데이터 증강 전략
- 합성 데이터 생성 (폰트 렌더링)
- 백번역/노이즈 주입으로 라벨 보존하며 이미지 변형

---

## 실행 명령어 정리

```bash
# 학습 재개 (기존 best_model.pth에서 이어서)
cd /home/dev/doctor_ocr_v2 && /home/dev/doctor_ocr/venv/bin/python local_train.py

# 추론 테스트
cd /home/dev/doctor_ocr_v2 && /home/dev/doctor_ocr/venv/bin/python local_infer.py

# 상세 분석 (처음 50샘플 문자 단위)
cd /home/dev/doctor_ocr_v2 && /home/dev/doctor_ocr/venv/bin/python -c "
# 분석 스크립트 내용 참조
"

# 백그라운드 학습 (긴 시간 실행 시)
hermes terminal --background --notify-on-complete "cd /home/dev/doctor_ocr_v2 && /home/dev/doctor_ocr/venv/bin/python local_train.py"
```

---

## 파일 위치 요약

| 파일 | 경로 | 용도 |
|------|------|------|
| 학습 스크립트 | `/home/dev/doctor_ocr_v2/local_train.py` | 메인 학습 루프 |
| 추론 스크립트 | `/home/dev/doctor_ocr_v2/local_infer.py` | 모델 로드 + 예측 |
| 베스트 모델 | `/home/dev/doctor_ocr_v2/working/checkpoints/best_model.pth` | Epoch 50, val_loss=2.1499 |
| 문자 사전 | `/home/dev/doctor_ocr_v2/working/char_dict.pkl` | char2idx, idx2char |
| 데이터 라벨 | `/home/dev/doctor_ocr_v2/dataset/doctor_handwriting_labels.csv` | 5,578행 |
| 이미지 폴더 | `/home/dev/doctor_ocr_v2/dataset/img/img/` | 5,578장 JPG |

---

## 메모: 원본 doctor_ocr과의 차이점

| 항목 | doctor_ocr (원본) | doctor_ocr_v2 (현재) |
|------|-------------------|----------------------|
| 데이터 | ~100장 의사 손글씨 | **5,578장 RxHandBD** |
| 라벨 형식 | filename,label | 동일 |
| 이미지 구조 | img/img/ | 동일 |
| 모델 | CRNN (동일) | CRNN (동일) |
| 학습 에포크 | 50까지 기록 있음 | **50 완료 (재학습)** |
| 최고 성능 | 미확인 | **Val Acc 6.18%** |

---

## 참고: 하드웨어 활용도

- **GPU 0**: 학습 전용 (89GB 사용, 8.2GB 여유)
- **GPU 1**: 다른 작업 중 (85GB 사용)
- **VRAM 여유 충분**: Batch 64~128까지 시도 가능
- **단일 GPU 최적**: DataParallel 오버헤드 회피로 3.3× 속도 향상 달성