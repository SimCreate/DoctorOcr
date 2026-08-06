#!/usr/bin/env python3
"""
v3_1 — v2.1(Attention 디코더) + v3 증강 데이터셋 재학습
=========================================================
접근: "가장 진보된 아키텍처(v2.1 multi-head attention + beam search)가
      데이터 부족(롱테일, 1,788 고유라벨 중 64% 1회)으로 학습에 실패했다"는
      진단을, v3에서 만든 확장 데이터셋(실사증강 2배, 13,386장)으로 풀어보는 실험.

v2.1 원본: /home/dev/doctor_ocr_v2_1/local_train_v2_1.py (수정 금지)
v3_1 차이점:
  1. 데이터: 온디스크 실사증강(exp2_clean) 사용 → 런타임 증강 OFF (중복 증강 방지)
  2. val: v3 클린 고정 split(val.csv 1,117장) 사용 → v3 exp2_clean과 동일 기준 비교
  3. checkpoint config 버그 수정 (hidden_size=384 저장, v2.1은 256으로 오기록)
"""

import os
import sys
import pickle
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# ============================================================
# CLI ARGS
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='1', help='CUDA_VISIBLE_DEVICES (default 1 = Max-Q)')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--accum-steps', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    return p.parse_args()

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# v3_1 자체 모델 정의
V3_1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_1 / "model"))
from model_v2_1 import (
    CRNN, build_char_dict, encode_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, SPECIAL_TOKENS
)

# ============================================================
# CONFIG — v3_1
# ============================================================
# 데이터: train=증강 exp2_clean(13,386), val=v2 원본(고정 클린 val 1,117)
V2_ORIG_IMG = Path("/home/dev/doctor_ocr_v2/dataset/img/img")   # v2 원본 (val 참조용)
DATA_ROOT = V3_1 / "data" / "exp2_clean"
IMG_DIR = DATA_ROOT / "img" / "img"                              # train: 증강 포함
LABEL_CSV = DATA_ROOT / "combined_labels.csv"
FIXED_VAL_CSV = V3_1 / "data" / "clean_split" / "val.csv"        # v3 클린 val (파일명 기준)

WORK_DIR = V3_1 / "working"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# 하이퍼파라미터 (v2.1 동일, Attention 디코더용)
BATCH_SIZE = 40
ACCUM_STEPS = 4              # effective batch 160
NUM_EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
DROPOUT = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 20     # v2.1(15)보다 여유 — 증강 데이터로 수렴이 늦을 수 있음
HIDDEN_SIZE = 384            # v2.1 실제값 (v2.1 config엔 256 오기록됨)
TEACHER_FORCING_START = 0.5
TEACHER_FORCING_END = 0.0
TEACHER_FORCING_DECAY = 0.95

# CLI 오버라이드
if args.batch_size: BATCH_SIZE = args.batch_size
if args.accum_steps: ACCUM_STEPS = args.accum_steps
if args.num_workers: NUM_WORKERS = args.num_workers
else: NUM_WORKERS = 4
if args.epochs: NUM_EPOCHS = args.epochs

PIN_MEMORY = True
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == "cuda" else None

print(f"[CONFIG] v3_1: LR={LR}, Batch={BATCH_SIZE}, Accum={ACCUM_STEPS}, Epochs={NUM_EPOCHS}")
print(f"[CONFIG] Data: {LABEL_CSV} ({IMG_DIR})")
print(f"[CONFIG] Val: fixed clean {FIXED_VAL_CSV}")
print(f"[CONFIG] Device: {DEVICE}, AMP={USE_AMP}, GC={USE_GRADIENT_CHECKPOINTING}")


# ============================================================
# DATASET (v2.1 동일 정의, augment=False 기본 — 온디스크 증강 사용)
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
        print(f"[DATASET] {csv_path} Total={len(self.df)}, Valid={len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

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
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        target = encode_label(label, self.char2idx, self.max_len)
        target = torch.tensor(target, dtype=torch.long)
        return img, target


# ============================================================
# LR Scheduler (Warmup + Cosine)
# ============================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_epoch += 1
        return lr


def get_teacher_forcing_ratio(epoch, start=TEACHER_FORCING_START, end=TEACHER_FORCING_END, decay=TEACHER_FORCING_DECAY):
    return max(start * (decay ** epoch), end)


def train_epoch(model, loader, criterion, optimizer, device, epoch, accum_steps=ACCUM_STEPS, use_amp=USE_AMP):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    tf_ratio = get_teacher_forcing_ratio(epoch)

    for step, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(imgs, targets, teacher_forcing_ratio=tf_ratio)
                loss = criterion(outputs[:, :-1].reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(imgs, targets, teacher_forcing_ratio=tf_ratio)
            loss = criterion(outputs[:, :-1].reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
            loss = loss / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}', 'tf': f'{tf_ratio:.2f}'})

    if len(loader) % accum_steps != 0:
        if use_amp and device.type == "cuda":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def validate(model, loader, criterion, device, idx2char, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs, targets=None, teacher_forcing_ratio=0.0)
            loss = criterion(outputs[:, :-1].reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
            total_loss += loss.item()
            preds = outputs.argmax(-1)
            for i in range(imgs.size(0)):
                pred_str = decode_sequence(preds[i].cpu().tolist(), idx2char)
                true_str = decode_sequence(targets[i, 1:].cpu().tolist(), idx2char)
                if pred_str == true_str:
                    correct += 1
                total += 1
    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader)
    print(f"[VAL] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f} ({correct}/{total})")
    return avg_loss, acc


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # char_dict — train(증강) 기준 구축 (val 라벨이 모두 포함되어 있음)
    df = pd.read_csv(LABEL_CSV)
    labels = df['label'].tolist()
    char2idx, idx2char = build_char_dict(labels)
    vocab_size = len(char2idx)
    print(f"[VOCAB] Size={vocab_size}, chars={sorted(char2idx.keys())}")

    with open(CHAR_DICT_PATH, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    print(f"[SAVED] char_dict.pkl -> {CHAR_DICT_PATH}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Train: exp2_clean 전체 (온디스크 증강). augment=False (중복 증강 방지)
    train_dataset = HandwritingDataset(LABEL_CSV, IMG_DIR, char2idx, transform, augment=False)
    # Val: v3 클린 val 1,117장 — v2 원본 디렉토리에서 읽음 (증강 안 섞임)
    val_dataset = HandwritingDataset(FIXED_VAL_CSV, V2_ORIG_IMG, char2idx, transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"[DATA] Train={len(train_dataset)}, Val(fixed)={len(val_dataset)}")

    model = CRNN(vocab_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                 use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING).to(DEVICE)
    print(f"[MODEL] Params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.step()
        print(f"\n[Epoch {epoch}] LR={current_lr:.6f}, TF={get_teacher_forcing_ratio(epoch):.2f}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, idx2char, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'vocab_size': vocab_size,
                'config': {
                    'hidden_size': HIDDEN_SIZE,       # ★ v2.1 버그 수정 (256→384)
                    'dropout': DROPOUT,
                    'max_label_length': MAX_LABEL_LENGTH,
                    'image_height': IMAGE_HEIGHT,
                    'image_width': IMAGE_WIDTH,
                    'use_amp': USE_AMP,
                    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
                    'batch_size': BATCH_SIZE,
                    'accum_steps': ACCUM_STEPS,
                    'data': 'exp2_clean (실사증강 2배)',
                    'val_split': 'fixed clean val.csv (v3 exp2_clean과 동일)',
                }
            }, BEST_MODEL_PATH)
            print(f"[SAVED] Best model -> {BEST_MODEL_PATH} (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"[EARLY STOP] Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if epoch % 10 == 0:
            periodic_path = CHECKPOINT_DIR / f"epoch_{epoch}.pth"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc,
                        'vocab_size': vocab_size}, periodic_path)
            print(f"[SAVED] Periodic -> {periodic_path}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
            break

    print(f"\n[DONE] v3_1 training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
