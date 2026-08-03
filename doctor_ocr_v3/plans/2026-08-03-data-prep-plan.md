# 데이터 가공 (실험군 1/2/3) 구현 계획

> **For Hermes:** 서브에이전트 환경 없음 → 에이전트가 직접 절차형 구현.
> 계획 수립 목적이므로, 이 문서는 사용자 확인 후 구현 착수.

**Goal:** v2.2 CTC 모델을 실험군 1/2/3 데이터로 각각 학습하기 위한 온디스크 데이터셋 3종을 구성한다. 원본(v2 dataset)은 절대 수정하지 않고, 가공물만 v3 전용 폴더에 생성한다.

**Architecture:** 온디스크 생성 방식. 각 실험군은 독립 `data/experiment_N/` 폴더에 CSV + 이미지 배치로 구성. 학습 스크립트가 이 폴더를 읽기만 하도록. 합성 샘플은 CSV에 `source` 컬럼(`real` / `aug` / `synth`)으로 마킹 → 학습은 섞되 평가/분석시 분리 집계 가능.

**환경:**
- venv: `/home/dev/doctor_ocr_v2_2/venv/` (Pillow 12.2.0 있음)
- 원본 데이터: `/home/dev/doctor_ocr_v2/dataset/` (읽기 전용)
- 폰트: `/usr/share/fonts/truetype/dejavu/` (DejaVu 시리즈)
- 디스크: /home/dev에 159G 여유

---

## 사전 검증된 사실 (계획 근거)

- 원본: 5,578장 이미지 + CSV(5,579행), v2_2 dataset = v2 dataset 심볼릭 링크 → **같은 데이터 공유**
- 라벨 빈도: 고빈도(10↑)=105라벨/2,582샘플, 중빈도(2~9)=535라벨/1,849샘플, 저빈도(1)=1,148라벨/1,148샘플
- 실험군 3의 합성 대상: 1~2회 출현 라벨 (2회 = 205라벨, 1회 = 1,148라벨 → 총 1,353라벨)
- v2_2 evaluate.py에 실사 증강 로직(회전/스케일/밝기/노이즈) 이미 있음 → 재사용/이식

---

## Task 1: v3 데이터 가공 폴더 + 원본 복사 (실험군 1)

**Objective:** v3 전용 `data/` 폴더 구조를 만들고, 원본 데이터를 실험군 1로 복사한다.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/`
- Copy: 원본 CSV 2종 + img/img → experiment_1/

**Step 1: 폴더 구조**
```bash
mkdir -p /home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/img/img
```

**Step 2: CSV 복사 + 원본 충실도 검증**
```bash
cp /home/dev/doctor_ocr_v2/dataset/combined_labels.csv /home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/
cp /home/dev/doctor_ocr_v2/dataset/doctor_handwriting_labels.csv /home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/
md5sum 비교로 원본과 동일 확인
```

**Step 3: 이미지 복사 (5,578장)**
```bash
cp -r /home/dev/doctor_ocr_v2/dataset/img/img/. /home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/img/img/
```

**Step 4: 검증**
- 파일 수: 5,578 (원본 5,578과 일치)
- CSV 행 수: 5,579 (원본과 일치)
- 샘플 몇 개 md5 대조

**Step 5: 커밋**
- 결과 CSV에 `source` 컬럼 추가: `source=real` (전부 실제)

> 주의: 원본 5,578장 전체를 git에 커밋하지 않는다. 데이터 폴더는 .gitignore에 추가.
> (저장소는 소스/스크립트/보고서만 관리 — 기존 README 구조와 동일)

---

## Task 2: 실사 증강 도구 (실험군 2)

**Objective:** 실사 이미지 변형 증강을 수행하는 스크립트를 v3에 작성한다. (v2_2의 로직 이식 + 증강 샘플 마킹)

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/scripts/augment_dataset.py`

**Step 1: 증강 함수 작성 (v2_2 evaluate.py 로직 재사용)**
```python
# 회전 ±5°, 스케일 0.9~1.1, 밝기 α 0.8~1.2/β±20, 노이즈 σ10
# 각 샘플마다 3~5장 증강 변형 생성
```

**Step 2: 증강 샘플 저장 + CSV 마킹**
- 원본 `real` 에서 증강본은 `aug` 로 마킹
- 증강 샘플별 파일명: `<원본파일명>__aug<i>.jpg`
- CSV에 `source` 컬럼: `real` / `aug`

**Step 3: 증강 비율 결정 (검증 후 사용자 confirm)**
- **증강 샘플은 각 원본의 2배로 생성** (사용자 확정, 2026-08-03) — vLLM 공존 학습 시간 관리 + "증강 효과 확인"이 목적이면 충분
- 총량: 원본 5,579행 → 증강 포함 ~16,700행 수준

**Step 4: 데이터셋 생성 + 검증**
- experiment_2/ 생성, 이미지 저장, CSV 생성
- 파일 수, 행 수, source 분포 출력

**Step 5: 커밋** (스크립트만 — 생성 데이터는 .gitignore)

---

## Task 3: 저빈도 합성 렌더러 (실험군 3)

**Objective:** 1~2회 출현 라벨을 폰트로 렌더링한 합성 이미지를 생성하고, 10~15% 비중 캡으로 실험군 2에 추가한다.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/scripts/synthesize_labels.py`

**Step 1: 폰트 렌더링 (Pillow)**
```python
# 라벨 문자열을 정해진 폰트(DejaVuSans 등), 크기, 회전, 노이즈로 렌더링
# 각 라벨 1~2회만 생성 (light-touch)
# 파일명: <label>__synth<i>.jpg  → 라벨 정보 유지
```

**Step 2: 비중 캡 계산 + 적용 (가드레일)**
- experiment_2 실사 샘플 수 N2일 때, 합성 샘플 수 = min(필요량, N2 × 0.10~0.15)
- 전체 합성 비중이 10~15%를 넘지 않도록 상한 설정
- light-touch 원칙: 1~2회 라벨당 1~2장만

**Step 3: 마킹**
- 합성 샘플은 `source=synth` 컬럼
- 라벨은 반드시 원본과 동일 (fake 라벨 생성 금지 — 7/25 전례)

**Step 4: 데이터셋 생성 + 검증**
- experiment_3/ 생성
- 사전 검증: 합성 비중 10~15% 이내인지 출력 확인
- 파일/행/source 분포 출력

**Step 5: 커밋** (스크립트만)

---

## Task 4: 데이터셋 무결성 검증 (공통)

**Objective:** 실험군 1/2/3 모두가 학습 가능한 형태인지 검증.

**Step 1: 각 실험군에 대해**
- CSV의 모든 이미지 파일이 실제 존재하는지 (ml-dataset-validation 스킬: filename-label 매핑 무결성)
- 라벨-파일 중복/충돌 없는지
- source 컬럼 존재 + 값 유효성 (real/aug/synth만)

**Step 2: 실험군 간 겹침 확인**
- 실험군 2/3의 증강·합성샘플이 원본(실험군1) 이미지와 파일명 충돌 없는지
- (학습/평가 시 데이터 누수 방지)

**Step 3: 최종 요약 출력 + 커밋**

---

## Task 5: README + 사용법 + 커밋

**Objective:** 데이터셋 구조와 생성 방법 문서화.

**Files:**
- Create: `/home/dev/DoctorOcr/doctor_ocr_v3/data/README.md`

**Step 1: 데이터셋 3종 구조, source 마킹 규칙, 생성 명령, 재생성 방법 기록**

**Step 2: 전체 검증 실행 + 커밋**
```bash
cd /home/dev/DoctorOcr/doctor_ocr_v3
# 각 실험군 검증 스크립트 실행 → ALL PASS
git add . && git commit
```

---

## 가드레일 (전 구간)

- **원본(v2 dataset) 절대 수정 없음** — 복사만, v3 폴더로
- 가공 스크립트는 v3/scripts/에만 추가 (v2/v2_2 코드 수정 없음)
- 합성은 저빈도(1~2회)만 + 비중 10~15% 캡 + 마킹(synth)
- 라벨은 원본과 동일 (fake 라벨 없음)
- 데이터 폴더는 .gitignore — 저장소엔 스크립트/문서/보고서만
- 생성 후 ml-dataset-validation 방식으로 무결성 검증

## 성공 기준

- [ ] experiment_1/2/3 각각 CSV+이미지 생성
- [ ] 실험군 2: aug 샘플이 원본 대비 3~4배, source=aug 마킹
- [ ] 실험군 3: 합성 비중 10~15% 이내, source=synth, 저빈도만
- [ ] 무결성 검증 ALL PASS (파일 존재, 라벨 매핑, 누수 없음)
- [ ] 커밋 + 푸시
