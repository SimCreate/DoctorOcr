"""실험군 1: 원본 CSV에 source 컬럼 추가 (전부 real)."""
import pandas as pd

SRC = '/home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/combined_labels.csv'

df = pd.read_csv(SRC)
print('원본 행 수:', len(df))
print('컬럼:', list(df.columns))

df['source'] = 'real'
df.to_csv(SRC, index=False)
print('source 추가 후 행 수:', len(df))
print('source 분포:', df['source'].value_counts().to_dict())

# 무결성: 모든 이미지 존재
import os
img_dir = '/home/dev/DoctorOcr/doctor_ocr_v3/data/experiment_1/img/img'
missing = [f for f in df['filename'] if not os.path.exists(os.path.join(img_dir, f))]
print('없는 이미지 수:', len(missing))
