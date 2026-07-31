#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Local Inference Script (v2_2 CTC)
CTC Greedy Decode 기반 추론
"""

import os
import sys
import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# ============================================================
# CLI ARGS (GPU 오버라이드용)
# ============================================================
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='CUDA_VISIBLE_DEVICES value (default: 0 = RTX PRO 6000 Blackwell)')
    return parser.parse_args()

args = parse_args()

# Apply CLI overrides BEFORE importing model
if args.device is None:
    args.device = '0'
if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# 모델 정의 임포트
sys.path.insert(0, str(Path(__file__).parent / "model"))
from model_v2_2 import (
    CRNN, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CTC_BLANK, SPECIAL_TOKENS
)

# ============================================================
# CONFIG - /home/dev/doctor_ocr_v2_2/ 경로 (Blackwell — cuda0)
# ============================================================
DATA_ROOT = Path("/home/dev/doctor_ocr_v2_2/dataset")
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "doctor_handwriting_labels.csv"

WORK_DIR = Path("/home/dev/doctor_ocr_v2_2/working")
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = WORK_DIR / "checkpoints" / "best_model.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Blackwell enforced: CUDA_VISIBLE_DEVICES='0' → cuda:0 = GPU 0

print(f"[CONFIG] DEVICE: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
print(f"[CONFIG] MODEL: {BEST_MODEL_PATH}")
print(f"[CONFIG] CHAR_DICT: {CHAR_DICT_PATH}")


# ============================================================
# UTILS
# ============================================================
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = (img - 0.5) / 0.5  # normalize
    return img.unsqueeze(0)  # (1, C, H, W)


def load_model_and_dict():
    # 문자 사전 로드
    with open(CHAR_DICT_PATH, 'rb') as f:
        char_dict = pickle.load(f)
    char2idx = char_dict['char2idx']
    idx2char = char_dict['idx2char']
    vocab_size = len(char2idx)
    print(f"[LOAD] Vocab size: {vocab_size}")

    # 모델 로드
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})

    model = CRNN(
        vocab_size,
        hidden_size=config.get('hidden_size', 256),
        dropout=config.get('dropout', 0.3),
        use_gradient_checkpointing=False  # 추론 시에는 gradient checkpointing 비활성화
    ).to(DEVICE)

    state_dict = checkpoint['model_state_dict']

    # Handle DataParallel checkpoint (keys have 'module.' prefix)
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
        print("[LOAD] DataParallel checkpoint detected, stripped 'module.' prefix")

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[LOAD] Model from epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f}, val_acc={checkpoint.get('val_acc', '?'):.4f}")

    return model, char2idx, idx2char


def predict_single(model, img_tensor, idx2char):
    """CTC greedy decode 예측"""
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        decoded = model.predict(img_tensor)  # List of token lists
        pred_str = decode_sequence(decoded[0], idx2char)
        return pred_str


def main():
    # 모델 + 사전 로드
    model, char2idx, idx2char = load_model_and_dict()

    # CSV 로드
    df = pd.read_csv(LABEL_CSV)

    # 랜덤 10개 샘플 선택 (이미지 존재하는 것만)
    valid_rows = []
    for _, row in df.iterrows():
        img_path = IMG_DIR / row['filename']
        if img_path.exists():
            valid_rows.append(row)

    if len(valid_rows) == 0:
        print("[ERROR] No valid images found")
        return

    samples = random.sample(valid_rows, min(10, len(valid_rows)))

    print(f"\n=== INFERENCE ON {len(samples)} RANDOM SAMPLES (CTC Greedy) ===\n")

    correct = 0
    for row in samples:
        img_path = IMG_DIR / row['filename']
        true_label = row['label']

        img_tensor = preprocess_image(img_path)
        if img_tensor is None:
            print(f"  {row['filename']} | label='{true_label}' | pred='ERROR: cannot load image' | X")
            continue

        pred_label = predict_single(model, img_tensor, idx2char)
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "X"
        print(f"  {row['filename']} | label='{true_label}' | pred='{pred_label}' | {status}")

    acc = correct / len(samples)
    print(f"\n=== ACCURACY: {correct}/{len(samples)} = {acc:.2%} ===")


if __name__ == "__main__":
    main()