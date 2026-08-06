#!/usr/bin/env python3
"""
Doctor Handwriting OCR - v3 공용 데이터셋/평가 유틸 (v2_2 evaluate.py에서 추출)

v2_2 라이브 원본: /home/dev/doctor_ocr_v2_2/evaluate.py
  - v2_2 원본은 수정 금지 (메모리 규칙)
  - 본 파일은 v3 자립을 위한 복제본.
  - 추출 대상: HandwritingDataset, ctc_collate_fn, evaluate_model (main() CLI는 v2_2 전용이라 제외)

사용:
  from handwriting_dataset import HandwritingDataset, ctc_collate_fn, evaluate_model
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# v3 자체 모델 정의 사용 (v3/model/)
from model_v2_2 import (
    encode_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CTC_BLANK,
)


# ============================================================
# DATASET (v2_2 학습 시와 일치해야 함)
# ============================================================
class HandwritingDataset(Dataset):
    def __init__(self, csv_path, img_dir, char2idx, transform=None, max_len=MAX_LABEL_LENGTH, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.char2idx = char2idx
        self.transform = transform
        self.max_len = max_len
        self.augment = augment

        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = self.img_dir / row['filename']
            if img_path.exists():
                self.valid_indices.append(idx)
            else:
                print(f"[WARN] Missing image: {img_path}")

        print(f"[DATASET] Total: {len(self.df)}, Valid: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def augment_image(self, img):
        if self.augment:
            if np.random.random() > 0.5:
                angle = np.random.uniform(-5, 5)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            if np.random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))
                img = cv2.resize(img, (w, h))

            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.randint(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            if np.random.random() > 0.7:
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
        return img

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]

        img_path = self.img_dir / row['filename']
        label = row['label']

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.augment:
                img = self.augment_image(img)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        target = encode_label(label, self.char2idx, self.max_len)
        target = torch.tensor(target, dtype=torch.long)

        return img, target, label, str(img_path)


def ctc_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    paths = [item[3] for item in batch]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)
    return imgs, targets_concat, target_lengths, labels, paths


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, loader, device, idx2char):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for imgs, targets, target_lengths, labels, paths in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(imgs)

            # CTC Loss
            B, T, V = logits.shape
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=CTC_BLANK, reduction='mean', zero_infinity=True
            )
            total_loss += loss.item()

            # Decode predictions
            decoded = model.ctc_head.decode(logits)

            target_list = targets.tolist()
            offset = 0
            for i, t_len in enumerate(target_lengths.tolist()):
                true_seq = target_list[offset:offset + t_len]
                pred_seq = decoded[i]
                true_str = decode_sequence(true_seq, idx2char)
                pred_str = decode_sequence(pred_seq, idx2char)

                predictions.append({
                    'true': true_str,
                    'pred': pred_str,
                    'match': pred_str == true_str,
                    'label': labels[i],
                    'path': paths[i]
                })

                if pred_str == true_str:
                    correct += 1
                total += 1
                offset += t_len

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0

    print(f"\n[EVAL RESULT] Loss={avg_loss:.4f}, Acc={acc:.4f} ({correct}/{total})")

    return avg_loss, acc, predictions
