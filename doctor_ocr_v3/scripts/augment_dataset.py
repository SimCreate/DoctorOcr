#!/usr/bin/env python3
"""
실험군 2: 실사 이미지 변형 증강.
원본(experiment_1)에서 각 샘플의 2배 증강본을 생성해 experiment_2로 저장.
증강: 회전 ±5°, 스케일 0.9~1.1, 밝기/대비, 노이즈 (v2_2 evaluate.py 로직 이식)
CSV에 source 컬럼: real(원본) / aug(증강본)

실행:
  /home/dev/doctor_ocr_v2_2/venv/bin/python augment_dataset.py
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

EXP1 = Path(__file__).resolve().parent.parent / "data" / "experiment_1"
EXP2 = Path(__file__).resolve().parent.parent / "data" / "experiment_2"
IMG_H, IMG_W = 64, 256
AUG_PER_SAMPLE = 2  # 사용자 확정: 원본 1장당 증강 2장


def augment_image(img, rng):
    """v2_2 evaluate.py의 증강 로직 재사용 (회전/스케일/밝기/노이즈)."""
    # 회전 ±5°
    if rng.random() > 0.5:
        angle = rng.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 스케일 0.9~1.1
    if rng.random() > 0.5:
        scale = rng.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.resize(img, (w, h))
    # 밝기/대비
    if rng.random() > 0.5:
        alpha = rng.uniform(0.8, 1.2)
        beta = int(rng.integers(-20, 20))
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # 노이즈
    if rng.random() > 0.7:
        noise = rng.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    return img


def main():
    # 읽기: experiment_1
    df = pd.read_csv(EXP1 / "combined_labels.csv")
    print(f"[LOAD] experiment_1: {len(df)} rows")

    # exp2 폴더
    img2 = EXP2 / "img" / "img"
    img2.mkdir(parents=True, exist_ok=True)

    # 원본 csv 먼저 복사 (source=real)
    real_df = df.copy()
    real_df['source'] = 'real'
    # exp1 이미지를 exp2로 복사
    src_img = EXP1 / "img" / "img"
    for fname in df['filename']:
        src = src_img / fname
        if src.exists():
            dst = img2 / fname
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    # 증강 샘플 생성 (2배)
    rng = np.random.default_rng(42)
    aug_rows = []
    for _, row in df.iterrows():
        src = img2 / row['filename']
        if not src.exists():
            continue
        orig = cv2.imread(str(src))
        if orig is None:
            continue
        img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H))
        for i in range(AUG_PER_SAMPLE):
            aug = augment_image(img.copy(), rng)
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            out_name = f"{Path(row['filename']).stem}__aug{i}.jpg"
            cv2.imwrite(str(img2 / out_name), aug_bgr)
            aug_rows.append({'filename': out_name, 'label': row['label'], 'source': 'aug'})

    aug_df = pd.DataFrame(aug_rows)
    combined = pd.concat([real_df[['filename', 'label', 'source']], aug_df], ignore_index=True)
    combined.to_csv(EXP2 / "combined_labels.csv", index=False)

    print(f"[DONE] experiment_2: {len(combined)} rows")
    print("source 분포:", combined['source'].value_counts().to_dict())
    print("이미지 파일 수:", len(list(img2.glob('*.jpg'))))

    # 무결성: 모든 이미지 존재
    missing = [f for f in combined['filename'] if not (img2 / f).exists()]
    print("없는 이미지:", len(missing))


if __name__ == '__main__':
    main()
