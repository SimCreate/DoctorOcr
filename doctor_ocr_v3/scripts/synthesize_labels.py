#!/usr/bin/env python3
"""
실험군 3: 저빈도(1~2회) 라벨 폰트 합성 (light-touch).
- 대상: 1회 출현 라벨 우선 (라벨당 1장), 예산 상한 내에서
- 상한: 실사(real) 샘플 대비 12.5% 이하 (사용자 확정, 2026-08-03)
- 마킹: source=synth
- 라벨은 반드시 원본과 동일 (fake 라벨 생성 금지 — 7/25 전례)
- 폰트: /usr/share/fonts/truetype/dejavu/

실행:
  /home/dev/doctor_ocr_v2_2/venv/bin/python synthesize_labels.py
"""
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import pandas as pd

EXP1 = Path(__file__).resolve().parent.parent / "data" / "experiment_1"
EXP3 = Path(__file__).resolve().parent.parent / "data" / "experiment_3"
IMG_W, IMG_H = 256, 64

# 폰트 후보 (손글씨에 가까운 sans/serif 혼합)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Oblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]

SYNTH_RATIO = 0.125  # 실사 대비 12.5% 상한 (사용자 확정)


def synth_image(text, rng):
    """폰트로 라벨을 렌더링해 256x64 이미지를 만든다."""
    img = Image.new('RGB', (IMG_W, IMG_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 폰트 선택 + 크기
    font_path = rng.choice(FONT_PATHS)
    font_size = int(rng.integers(32, 44))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)

    # 텍스트 중앙 배치 (저빈도 단어는 대체로 짧음)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = int(rng.integers(5, max(6, (IMG_W - tw) // 2 - 10)))
    y = max(0, (IMG_H - th) // 2 + int(rng.integers(-8, 8)))
    draw.text((x, y), text, font=font, fill=(30, 30, 30))

    # 손글씨 흉내: 미세 회전 + 약간 번짐 + 노이즈
    angle = rng.uniform(-4, 4)
    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 0.6)))
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, 12, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # 밝기/대비 약간
    alpha = rng.uniform(0.85, 1.1)
    arr = np.clip(arr * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    rng = np.random.default_rng(7)

    # 실사 real 샘플 수 (experiment_2 기준)
    df2 = pd.read_csv(EXP1.parent / "experiment_2" / "combined_labels.csv")
    real_count = (df2['source'] == 'real').sum()
    synth_cap = int(real_count * SYNTH_RATIO)
    print(f"[SYNTH] 실사 {real_count} × {SYNTH_RATIO:.1%} → 합성 상한 {synth_cap}장")

    # 1~2회 라벨 (1회 우선)
    df1 = pd.read_csv(EXP1 / "combined_labels.csv")
    vc = df1['label'].value_counts()
    one_two = vc[vc <= 2].index.tolist()
    # 1회 먼저, 그다음 2회 (라벨 알파벳 순으로 무작위성 안정)
    one_count = [l for l in one_two if vc[l] == 1]
    two_count = [l for l in one_two if vc[l] == 2]
    ordered = one_count + two_count
    print(f"[SYNTH] 대상 라벨: 총 {len(ordered)} (1회 {len(one_count)} + 2회 {len(two_count)})")

    # 상한까지 (라벨당 1장)
    selected = ordered[:synth_cap]
    print(f"[SYNTH] 실제 합성 라벨 수: {len(selected)} (라벨당 1장)")

    # experiment_3 = 실험군 2 전체(real+aug) + 합성(synth)
    # → 실험군 2와의 차이는 synth 유무 하나만 (실험 설계 순수성)
    img3 = EXP3 / "img" / "img"
    img3.mkdir(parents=True, exist_ok=True)

    # exp2 이미지 + CSV 복사 (real + aug 그대로)
    exp2_img = EXP1.parent / "experiment_2" / "img" / "img"
    for f in exp2_img.glob("*.jpg"):
        dst = img3 / f.name
        if not dst.exists():
            dst.write_bytes(f.read_bytes())
    real_df = df2[['filename', 'label', 'source']].copy()

    # 합성 생성 — 파일명은 고유 인덱스 기반 (라벨 정규화 충돌 방지)
    # 서로 다른 라벨이 정규화 후 같은 파일명이 되는 사건 방지
    synth_rows = []
    for k, label in enumerate(selected):
        img = synth_image(str(label), rng)
        fname = f"synth_{k:05d}.jpg"
        img.save(img3 / fname)
        synth_rows.append({'filename': fname, 'label': str(label), 'source': 'synth'})

    synth_df = pd.DataFrame(synth_rows)
    combined = pd.concat([real_df, synth_df], ignore_index=True)
    combined.to_csv(EXP3 / "combined_labels.csv", index=False)

    print(f"[DONE] experiment_3: {len(combined)} rows")
    print("source 분포:", combined['source'].value_counts().to_dict())
    # 비중 확인
    n_synth = (combined['source'] == 'synth').sum()
    n_real = (combined['source'] == 'real').sum()
    print(f"합성 비중 (synth/real): {n_synth}/{n_real} = {n_synth / n_real:.1%} (캡 12.5%)")

    # 무결성
    missing = [f for f in combined['filename'] if not (img3 / f).exists()]
    print("없는 이미지:", len(missing))


if __name__ == '__main__':
    main()
