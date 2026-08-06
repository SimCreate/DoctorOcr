#!/usr/bin/env python3
"""
클린 스플릿 데이터 재구성 — 원본을 먼저 80/20 split하고, train에만 증강/합성을 붙인다.

배경 (2026-08-04 리키지 발견):
  기존 파이프라인은 "증강 포함 전체 데이터셋"을 seed 42로 random_split 했기 때문에,
  run_eval이 val로 쓰는 원본 1,116장 중 ~80%가 exp2/exp3 train에 그대로 포함됐다.
  → 98.8%는 이미지 단위 리키지로 부풀려진 수치.

해결:
  step 1. 원본 combined_labels.csv (5,579행, 결함 'Images' 제외 시 5,578 유효)를
          HandwritingDataset 로직(존재하는 이미지 기준) 그대로 seed 42로 80/20 split.
          → data/clean_split/train.csv (4,462) / val.csv (1,116)
          ※ 이 val은 이전 run_eval의 val과 100% 동일 (verify_split_repro.py 로 확인)
  step 2. train.csv에만 증강 2배 (exp2_clean) / 증강+저빈도합성 (exp3_clean)
  step 3. val.csv는 건드리지 않음 → 구조적으로 클린

실행:
  venv/bin/python scripts/build_clean_split.py

산출:
  data/clean_split/{train.csv,val.csv}
  data/experiment_1_clean/   (train=real 4,462)     — val은 포함하지 않음
  data/experiment_2_clean/   (train=real+aug)
  data/experiment_3_clean/   (train=real+aug+synth)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3 / "scripts"))
sys.path.insert(0, str(V3 / "model"))
sys.path.insert(0, str(V3 / "evaluate"))

from model_v2_2 import build_char_dict  # noqa: E402
from handwriting_dataset import HandwritingDataset  # noqa: E402
from augment_dataset import augment_image, AUG_PER_SAMPLE, IMG_H, IMG_W  # noqa: E402
from synthesize_labels import synth_image, FONT_PATHS, SYNTH_RATIO  # noqa: E402

ORIG_CSV = "/home/dev/doctor_ocr_v2/dataset/combined_labels.csv"
ORIG_IMG = "/home/dev/doctor_ocr_v2/dataset/img/img"

CLEAN = V3 / "data" / "clean_split"
EXP1 = V3 / "data" / "experiment_1_clean"
EXP2 = V3 / "data" / "experiment_2_clean"
EXP3 = V3 / "data" / "experiment_3_clean"

SEED = 42


def make_split():
    """원본을 HandwritingDataset 로직(유효 이미지) 그대로 80/20 split."""
    import os
    df0 = pd.read_csv(ORIG_CSV)
    # char_dict는 라벨 전체 기준
    char2idx, _ = build_char_dict(df0['label'].tolist())
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    full = HandwritingDataset(ORIG_CSV, ORIG_IMG, char2idx, transform, augment=False)
    print(f"[SPLIT] 유효 이미지 = {len(full)} (원본 행 {len(df0)})")

    torch.manual_seed(SEED)
    train_size = int(0.8 * len(full))
    train_ds, val_ds = torch.utils.data.random_split(
        full, [train_size, len(full) - train_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    def collect(ds):
        rows = []
        for i in ds.indices:
            real_i = full.valid_indices[i]
            r = full.df.iloc[real_i]
            rows.append({'filename': r['filename'], 'label': r['label']})
        return rows

    train_rows = collect(train_ds)
    val_rows = collect(val_ds)
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    CLEAN.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(CLEAN / "train.csv", index=False)
    val_df.to_csv(CLEAN / "val.csv", index=False)
    print(f"[SPLIT] train {len(train_df)} / val {len(val_df)}")
    return train_df, val_df


def copy_img(src_dir, dst_dir, fname):
    dst_dir.mkdir(parents=True, exist_ok=True)
    s = src_dir / fname
    d = dst_dir / fname
    if s.exists() and not d.exists():
        d.write_bytes(s.read_bytes())


def build_exp1(train_df):
    """exp1_clean = 원본 train 4,462장 (source=real). val 미포함."""
    img = EXP1 / "img" / "img"
    img.mkdir(parents=True, exist_ok=True)
    src = Path(ORIG_IMG)
    for f in train_df['filename']:
        copy_img(src, img, f)
    out = train_df.copy()
    out['source'] = 'real'
    out.to_csv(EXP1 / "combined_labels.csv", index=False)
    print(f"[EXP1] {len(out)} rows (source=real, val 제외)")


def build_exp2(train_df):
    """exp2_clean = 원본 train + 증강 2배 (source=real+aug)."""
    img = EXP2 / "img" / "img"
    img.mkdir(parents=True, exist_ok=True)
    src = Path(ORIG_IMG)
    for f in train_df['filename']:
        copy_img(src, img, f)

    rng = np.random.default_rng(SEED)
    aug_rows = []
    for _, row in train_df.iterrows():
        sp = img / row['filename']
        if not sp.exists():
            continue
        orig = cv2.imread(str(sp))
        if orig is None:
            continue
        imRGB = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        imRGB = cv2.resize(imRGB, (IMG_W, IMG_H))
        for i in range(AUG_PER_SAMPLE):
            aug = augment_image(imRGB.copy(), rng)
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            out_name = f"{Path(row['filename']).stem}__aug{i}.jpg"
            cv2.imwrite(str(img / out_name), aug_bgr)
            aug_rows.append({'filename': out_name, 'label': row['label'], 'source': 'aug'})

    real_df = train_df.copy()
    real_df['source'] = 'real'
    combined = pd.concat([real_df, pd.DataFrame(aug_rows)], ignore_index=True)
    combined.to_csv(EXP2 / "combined_labels.csv", index=False)
    print(f"[EXP2] {len(combined)} rows — source 분포: {combined['source'].value_counts().to_dict()}")


def build_exp3(train_df):
    """exp3_clean = exp2_clean + 저빈도(1~2회) 라벨 폰트 합성 (라벨당 1장, 12.5% 캡)."""
    img3 = EXP3 / "img" / "img"
    img3.mkdir(parents=True, exist_ok=True)

    # exp2_clean의 real+aug 전체 복사 (차이 = synth 유무 하나만)
    exp2_img = EXP2 / "img" / "img"
    for f in exp2_img.glob("*.jpg"):
        d = img3 / f.name
        if not d.exists():
            d.write_bytes(f.read_bytes())
    real_df = pd.read_csv(EXP2 / "combined_labels.csv")

    # 저빈도 대상: 원본 train 내 1~2회 라벨 (1회 우선)
    vc = train_df['label'].value_counts()
    one_two = vc[vc <= 2].index.tolist()
    ordered = [l for l in one_two if vc[l] == 1] + [l for l in one_two if vc[l] == 2]

    real_count = (real_df['source'] == 'real').sum()
    synth_cap = int(real_count * SYNTH_RATIO)
    selected = ordered[:synth_cap]
    print(f"[EXP3] 실사 {real_count} × {SYNTH_RATIO:.1%} → 합성 상한 {synth_cap}, 대상 라벨 {len(ordered)}, 선택 {len(selected)}")

    rng = np.random.default_rng(7)
    synth_rows = []
    for k, label in enumerate(selected):
        img = synth_image(str(label), rng)
        fname = f"synth_{k:05d}.jpg"
        img.save(img3 / fname)
        synth_rows.append({'filename': fname, 'label': str(label), 'source': 'synth'})

    combined = pd.concat([real_df, pd.DataFrame(synth_rows)], ignore_index=True)
    combined.to_csv(EXP3 / "combined_labels.csv", index=False)
    n_synth = (combined['source'] == 'synth').sum()
    n_real = (combined['source'] == 'real').sum()
    print(f"[EXP3] {len(combined)} rows — synth/real = {n_synth}/{n_real} = {n_synth/n_real:.1%} (캡 12.5%)")


def main():
    print("=" * 60)
    print("클린 스플릿 데이터 재구성 시작")
    print("=" * 60)
    train_df, val_df = make_split()
    print(f"[CHECK] val 파일이 이전 run_eval val과 동일해야 100% 일치 (verify_split_repro.py 참고)")
    build_exp1(train_df)
    build_exp2(train_df)
    build_exp3(train_df)
    print("\n[DONE] 클린 스플릿 재구성 완료")
    print(f"  train: {CLEAN}/train.csv ({len(train_df)})")
    print(f"  val:   {CLEAN}/val.csv ({len(val_df)})")
    print(f"  exp1_clean: {EXP1}, exp2_clean: {EXP2}, exp3_clean: {EXP3}")


if __name__ == '__main__':
    main()
