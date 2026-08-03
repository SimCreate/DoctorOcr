# doctor_ocr_v3 데이터셋 가이드

실험군 1/2/3 데이터셋 구성과 용도, 재생성 방법을 설명한다.
원본(v2 dataset)은 절대 수정하지 않는다 — 전부 v3 전용 폴더에 복사/생성.

## 구조

```
data/
├── experiment_1/          실험군 1: 원본 그대로 (baseline)
│   ├── combined_labels.csv   filename,label,source
│   └── img/img/              원본 5,578장
├── experiment_2/          실험군 2: 실사 증강 2배
│   ├── combined_labels.csv   source=real(5,579) + aug(11,156)
│   └── img/img/              16,734장
└── experiment_3/          실험군 3: 실험군 2 + 저빈도 합성 (12.5% 캡)
    ├── combined_labels.csv   real + aug + synth(697)
    └── img/img/              17,431장
```

## source 마킹 규칙 (핵심)

| source | 의미 | 생성 방법 |
|--------|------|-----------|
| `real` | 원본 이미지 | v2 dataset에서 복사 |
| `aug`  | 실사 증강 | 회전/스케일/밝기/노이즈 (원본 1장당 2장) |
| `synth`| 폰트 합성 | 저빈도(1~2회) 라벨, 1회 우선, 라벨당 1장 |

- `synth`는 학습은 섞되, **분석/평가 시 반드시 분리 집계**한다.
- 라벨은 전부 원본과 동일 (fake 라벨 없음 — 7/25 전례 주의).

## 실험군별 행 수

| 실험군 | real | aug | synth | 합계 |
|--------|------|-----|-------|------|
| 1 | 5,579 | - | - | 5,579 |
| 2 | 5,579 | 11,156 | - | 16,735 |
| 3 | 5,579 | 11,156 | 697 | 17,432 |

## 생성 명령

```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3

# 실험군 1 (원본 복사 + source=real 마킹)
#   → 최초 1회 수동: cp 원본 → data/experiment_1, mark_exp1.py 실행

# 실험군 2 (실사 증강 2배)
/home/dev/doctor_ocr_v2_2/venv/bin/python scripts/augment_dataset.py

# 실험군 3 (저빈도 합성, 12.5% 캡)
/home/dev/doctor_ocr_v2_2/venv/bin/python scripts/synthesize_labels.py

# 무결성 검증 (전 실험군)
/home/dev/doctor_ocr_v2_2/venv/bin/python scripts/verify_datasets.py
```

## 재생성 시 주의

- `experiment_3` 생성 전 `rm -rf data/experiment_3`으로 비워야 충돌 파일이 안 남는다.
- 시드 고정 (증강 rng=42, 합성 rng=7) → 재생성 시 결정적(deterministic)으로 동일 결과.

## 설계 결정 기록

- 증강 배수: **2배** (사용자 확정, 2026-08-03) — vLLM 공존 학습 시간 관리
- 합성 비중: **12.5% 캡** (사용자 확정) — 설계 문서의 10~15% 중간값
- 합성 대상: 1~2회 라벨에서 **1회 우선** (가장 드문 어휘 노출 목적)
- 합성 파일명: `synth_<index>.jpg` (라벨 정규화로 인한 파일명 충돌 방지 — 2026-08-03 버그 수정)
