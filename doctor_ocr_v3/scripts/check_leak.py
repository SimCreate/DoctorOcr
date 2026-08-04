import csv

base = '/home/dev/doctor_ocr_v2/dataset'
with open(f'{base}/combined_labels.csv') as f:
    all_rows = [r for r in csv.DictReader(f)]
all_labels = set(r['label'].strip() for r in all_rows)
print(f"원본 전체 이미지: {len(all_rows)} / 고유 라벨: {len(all_labels)}")

rows = [r for r in csv.DictReader(open('evaluate/result_exp2_orig.csv'))]
val_labels = set(r['label'] for r in rows)
print(f"원본 val(평가) 이미지: {len(rows)} / val 고유 라벨: {len(val_labels)}")

overlap = all_labels & val_labels
print(f"val 라벨 중 train에도 존재하는 라벨 수: {len(overlap)} / {len(val_labels)}")
only_val = val_labels - all_labels
print(f"val에만 있는 라벨(모델이 학습 중 한 번도 안 본 약품명): {len(only_val)} -> {sorted(only_val)[:10]}")
