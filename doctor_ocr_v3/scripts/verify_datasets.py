"""
실험군 1/2/3 데이터셋 무결성 검증 (ml-dataset-validation 방식).
- 파일-라벨 매핑: 모든 filename이 이미지에 존재
- source 유효성: real/aug/synth만
- 실험군 간 누수: exp2/3의 aug·synth가 exp1 원본과 파일명 충돌 없나
"""
import os
import sys
from pathlib import Path

import pandas as pd

DATA = Path('/home/dev/DoctorOcr/doctor_ocr_v3/data')
fails = []


def check(dataset):
    csv = DATA / dataset / 'combined_labels.csv'
    img_dir = DATA / dataset / 'img' / 'img'
    df = pd.read_csv(csv)
    print(f"\n=== {dataset} ===")
    print(f"행 수: {len(df)}, 컬럼: {list(df.columns)}")

    # 1. 필수 컬럼
    for col in ('filename', 'label', 'source'):
        if col not in df.columns:
            fails.append(f"{dataset}: 컬럼 {col} 없음")

    # 2. 파일 존재
    missing = [f for f in df['filename'] if not (img_dir / f).exists()]
    print(f"파일 없음: {len(missing)} (기대: Images 결함 1건만)")
    if missing and missing != ['Images']:
        fails.append(f"{dataset}: 예상 밖 누락 {len(missing)}")

    # 3. source 유효성
    valid = df['source'].isin(['real', 'aug', 'synth']).all()
    if not valid:
        fails.append(f"{dataset}: source 값 무효")
    print(f"source 분포: {df['source'].value_counts().to_dict()}")

    # 4. filename 중복 (같은 파일이 두 번?)
    dup = df['filename'].duplicated().sum()
    if dup:
        fails.append(f"{dataset}: filename 중복 {dup}개")
    print(f"filename 중복: {dup}")

    # 5. 라벨-파일: 같은 파일에 서로 다른 라벨?
    lab_conflict = df.groupby('filename')['label'].nunique()
    n_conflict = (lab_conflict > 1).sum()
    if n_conflict:
        fails.append(f"{dataset}: 파일-라벨 충돌 {n_conflict}")
    print(f"파일-라벨 충돌: {n_conflict}")

    return df


# 실험군 1/2/3 각각
dfs = {d: check(d) for d in ('experiment_1', 'experiment_2', 'experiment_3')}

# 6. 듀플리케이션/누수: exp2 aug·synth 파일명이 exp1 원본에 겹치나
exp1_files = set(dfs['experiment_1']['filename'])
print("\n=== 실험군 간 누수 (aug/synth가 exp1 원본과 겹치나) ===")
for d in ('experiment_2', 'experiment_3'):
    non_real = dfs[d][dfs[d]['source'] != 'real']['filename']
    overlap = set(non_real) & exp1_files
    print(f"{d}: aug/synth 신규 파일 {len(non_real)}개 중 exp1과 겹침 {len(overlap)}개")
    if overlap:
        fails.append(f"{d}: 누수 — 겹침 {len(overlap)}")

# 7. exp2와 exp3의 real+aug가 동일해야 (실험 설계 순수성)
e2_ra = dfs['experiment_2'][dfs['experiment_2']['source'] != 'synth'][['filename', 'label', 'source']].sort_values('filename').reset_index(drop=True)
e3_ra = dfs['experiment_3'][dfs['experiment_3']['source'] != 'synth'][['filename', 'label', 'source']].sort_values('filename').reset_index(drop=True)
same = e2_ra.equals(e3_ra)
print(f"\nexp2 non-synth == exp3 non-synth (실험 설계 순수성): {same}")
if not same:
    fails.append("exp2/exp3 비합성 부분이 다름 — 설계 깨짐")

print("\n===== 종합 =====")
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
