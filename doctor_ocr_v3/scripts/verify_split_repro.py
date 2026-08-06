#!/usr/bin/env python3
"""
클린 split이 이전 run_eval의 val 1,116장과 정확히 일치하는지 검증.
run_eval.py의 실제 로직(HandwritingDataset = 존재하는 이미지만 valid_indices,
seed 42 random_split 80/20)을 그대로 재현해 val을 뽑고,
기존 result_exp2_orig.csv의 path와 대조한다.
"""
import os, sys
from pathlib import Path
import pandas as pd
import torch

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3 / "model"))
sys.path.insert(0, str(V3 / "evaluate"))
from model_v2_2 import CRNN, build_char_dict, MAX_LABEL_LENGTH  # noqa: E402
from handwriting_dataset import HandwritingDataset, ctc_collate_fn  # noqa: E402
from torchvision import transforms

base_csv = "/home/dev/doctor_ocr_v2/dataset/combined_labels.csv"
base_img = "/home/dev/doctor_ocr_v2/dataset/img/img"

# char_dict (원본 기준)
df0 = pd.read_csv(base_csv)
labels = df0['label'].tolist()
char2idx, idx2char = build_char_dict(labels)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full = HandwritingDataset(base_csv, base_img, char2idx, transform, augment=False)
print(f"[REPRO] valid dataset size = {len(full)}")

torch.manual_seed(42)
train_size = int(0.8 * len(full))
train_ds, val_ds = torch.utils.data.random_split(full, [train_size, len(full) - train_size])
print(f"[REPRO] train={len(train_ds)}, val={len(val_ds)}")

# val 파일명 수집
val_ds_df = full.df  # full dataset의 df (valid_indices 기준이 아니라 원본 df)
# random_split은 valid_indices를 섞은 것. val_ds.indices는 full의 valid_indices 중 일부 인덱스(subset).
val_fnames = set()
for i in val_ds.indices:
    real_i = full.valid_indices[i]
    fname = full.df.iloc[real_i]['filename']
    val_fnames.add(fname)
print(f"[REPRO] val unique filenames = {len(val_fnames)}")

# 기존 result_exp2_orig.csv의 val 파일명
old = pd.read_csv("/home/dev/DoctorOcr/doctor_ocr_v3/evaluate/result_exp2_orig.csv")
old_fnames = set(Path(p).name for p in old['path'])
print(f"[OLD] old run_eval val files = {len(old_fnames)}")

# 대조
inter = val_fnames & old_fnames
print(f"[CMP] 재현 val ∩ 기존 val = {len(inter)} / 기존 {len(old_fnames)}")
print(f"[CMP] 재현 전용(기존에 없는 것): {len(val_fnames - old_fnames)}")
print(f"[CMP] 기존 전용(재현에 없는 것): {len(old_fnames - val_fnames)}")
print(f"[CMP] 일치: {val_fnames == old_fnames}")
