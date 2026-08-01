# DoctorOcr

의사 손글씨 처방전 OCR — CRNN 계열 모델 학습·추론 파이프라인.
Kaggle RxHandBD 데이터셋(5,578장) 기반 4가지 아키텍처 순차 실험.

## 프로젝트 구조

모노레포: `/home/dev/DoctorOcr/` (git 저장소, 원본 `.git` 제거 후 통합)

```
DoctorOcr/
├── README.md                     # 상위 인덱스 (본 파일)
├── .gitignore
├── doctor_ocr/                   # v1 — 기준 CRNN (Attention Decoder)
│   ├── local_train.py
│   ├── local_infer.py
│   └── doctor_ocr_v1.md          # v1 상세 아키텍처 보고
├── doctor_ocr_v2/                # v2 — 개선 CRNN (대용량 배치 + AMP)
│   ├── local_train.py
│   ├── local_infer.py
│   └── PROJECT_LOG.md            # v2 전체 작업 로그
├── doctor_ocr_v2_1/              # v2.1 — BiLSTM 3층 + Attention
│   ├── local_train_v2_1.py
│   ├── local_infer_v2_1.py
│   ├── model/model_v2_1.py
│   ├── report_v2_1.md            # v2.1 상세 보고
│   └── report_v2.md              # v2 상세 보고
├── doctor_ocr_v2_2/              # v2.2 — CTC 기반 CRNN (최고 성능)
│   ├── local_train_v2_2.py
│   ├── local_infer_v2_2.py
│   ├── evaluate.py               # 평가/추론 통합
│   ├── model/model_v2_2.py
│   └── report_v2_2.md            # v2.2 상세 보고
└── docs/
    ├── doctor_ocr_v2_report.md   # 4개 프로젝트 종합 보고
    └── doctor_ocr_v1_notion.md   # v1 Notion 업로드용 상세
```

> **주의**: 실제 학습·추론 스크립트는 라이브 작업 디렉토리
> (`/home/dev/doctor_ocr`, `doctor_ocr_v2`, `doctor_ocr_v2_1`, `doctor_ocr_v2_2`)에서 실행.
> 그곳에 `dataset/`, `working/checkpoints/`, `char_dict.pkl`, venv, 로그 존재.
> 본 모노레포는 소스·보고서의 버전 관리용.
> 스크립트 내 경로는 `/home/dev/doctor_ocr_*` 하드코딩.

## 버전별 결과 요약

| 항목 | v1 `doctor_ocr` | v2 `_v2` | v2.1 `_v2_1` | v2.2 `_v2_2` |
|---|---|---|---|---|
| 아키텍처 | CRNN (7CNN + BiLSTM2 + Attn) | 개선 CRNN | BiLSTM 3층 + Attention | CTC 기반 CRNN |
| 손실 함수 | CrossEntropy | CrossEntropy | CrossEntropy | **CTC Loss** |
| BATCH | 8 | 96 (AMP) | 40 | 558 |
| BiLSTM 층수 / hidden | 2 / 256 | 2 / 256 | 3 / 384 | 3 / 256 |
| Epoch | 50 | 50 | 54+ | 38+ |
| Best val_loss | 2.7976 | 2.3205 | 2.2795 | **1.0216** |
| 추론 정확도 | 0% (0/10) | 20% (2/10) | 20% (2/10, 수정 후) | **40% (4/10)** |

- **최고 성능**: v2.2 — CTC Loss + 대용량 배치, val_loss 1.0216, 추론 40%.
- **핵심 개선 요인**: Attention seq2seq → CTC 전환 (반복 토큰 해결, alignment 불필요).

## 데이터셋

- **소스**: Kaggle RxHandBD — 의사 처방전 손글씨 5,578장 (Train 5,000 + Test 400)
- **고유 문자**: 73자 (알파벳, 숫자, 특수문자)
- **구조**: `dataset/img/img/` 이미지 + `doctor_handwriting_labels.csv` (filename, label)
- **입력 크기**: 64×256 RGB (v2.2 기준 IMAGE_HEIGHT=64, IMAGE_WIDTH=256)
- 원본 데이터/라벨 CSV는 `.gitignore` 제외 (용량 문제)

## 하드웨어 / 환경

- **GPU**: 2× RTX PRO 6000 Blackwell (192GB VRAM) — **단일 GPU(cuda:0) 사용**
- **중요**: 학습/추론 스크립트 `CUDA_VISIBLE_DEVICES='0'` 강제.
  USB4 4060 Ti 참조 전부 제거. Blackwell sm_120 호환은 PyTorch 2.11+cu128/130 필요.
- **venv**: v1/v2/v2_1 → `/home/dev/doctor_ocr/venv`,
  v2_1 → `/home/dev/vllm-env` (PyTorch 2.11.0+cu130),
  v2_2 → `/home/dev/doctor_ocr_v2_2/venv`

## 실행 방법

> 스크립트 내 경로는 라이브 디렉토리(`/home/dev/doctor_ocr_*`) 기준.
> 모노레포 사본 실행 시 경로 선수정 필요.

```bash
# v2.2 학습 (최고 성능 모델)
cd /home/dev/doctor_ocr_v2_2 && ./venv/bin/python local_train_v2_2.py

# v2.2 추론/평가
cd /home/dev/doctor_ocr_v2_2 && ./venv/bin/python local_infer_v2_2.py
cd /home/dev/doctor_ocr_v2_2 && ./venv/bin/python evaluate.py   # 검증셋 정확도 + 문자별 분석 + eval_results.csv

# v1 학습/추론
cd /home/dev/doctor_ocr && ./venv/bin/python local_train.py
cd /home/dev/doctor_ocr && ./venv/bin/python local_infer.py
```

- 학습: val_loss 최소 시 `working/checkpoints/best_model.pth` 저장 (epoch/optimizer/config 포함)
- 문자 사전: `working/char_dict.pkl` (char2idx, idx2char)

## 알려진 이슈 / 교훈

- **v2.1 체크포인트 config 버그**: 학습 시 `hidden_size=384`인데 config엔 `256` 저장.
  추론 시 `local_infer_v2_1.py:106` `hidden_size=384` 강제로 우회. 근본 원인(저장 로직) 미수정.
- **Attention 반복 토큰**: teacher forcing 0.5에도 `Deliiiii` 같은 반복 생성 → CTC로 해결.
- **DataParallel 비효율**: 소규모 모델(12M)에선 단일 GPU + 대용량 배치가 3.3× 빠름.
- **4개 동시 실행 시 GPU 0 VRAM 92~97GB** 사용 — OOM 위험.

## Git

```bash
# 모노레포 루트에서
git add -A && git commit -m "..." && git push origin main
```

현재 커밋: `Initial commit` (4 variant mono-repo) + `Add docs: v1 Notion details + v2 report` + README 재작성.
