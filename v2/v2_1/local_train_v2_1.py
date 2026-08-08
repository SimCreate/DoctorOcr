#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Local Training Script (v2_1)
개선된 학습: Warmup + Cosine Annealing, Mixed Precision, Gradient Accumulation,
데이터 증강, Early Stopping, Teacher Forcing Scheduling
"""

import os
import sys
import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# ============================================================
# CLI ARGS (GPU/배치/accum 오버라이드용 - 기본값은 기존 CONFIG 유지)
# ============================================================
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='CUDA_VISIBLE_DEVICES value (default: 0 = RTX PRO 6000 Blackwell)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override BATCH_SIZE')
    parser.add_argument('--accum-steps', type=int, default=None, help='Override ACCUM_STEPS')
    parser.add_argument('--num-workers', type=int, default=None, help='Override NUM_WORKERS')
    parser.add_argument('--amp', action='store_true', default=None, help='Enable AMP (default: from CONFIG)')
    parser.add_argument('--no-amp', action='store_true', default=None, help='Disable AMP')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=None, help='Enable gradient checkpointing')
    parser.add_argument('--no-gradient-checkpointing', action='store_true', default=None, help='Disable gradient checkpointing')
    return parser.parse_args()

args = parse_args()

# Apply CLI overrides BEFORE importing model (so CUDA_VISIBLE_DEVICES takes effect)
# Default to GPU 0 (RTX PRO 6000 Blackwell Workstation Edition) — cuda0
if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 모델 정의 임포트
sys.path.insert(0, str(Path(__file__).parent / "model"))
from model_v2_1 import (
    CRNN, build_char_dict, encode_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, SPECIAL_TOKENS
)

# ============================================================
# CONFIG - /home/dev/doctor_ocr_v2_1/ 경로 (RTX PRO 6000 Blackwell Workstation — cuda0, USB4 4060Ti 제거)
# ============================================================
DATA_ROOT = Path("/home/dev/doctor_ocr_v2_1/dataset")
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "doctor_handwriting_labels.csv"

WORK_DIR = Path("/home/dev/doctor_ocr_v2_1/working")
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# 학습 하이퍼파라미터 (Blackwell workstation — cuda0, 192GB VRAM)
BATCH_SIZE = 40             # 16GB VRAM에서 AMP + gradient checkpointing 시 40~48 안전 (목표 VRAM ~4GB)
ACCUM_STEPS = 4             # Effective batch size = 160
NUM_EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
DROPOUT = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
TEACHER_FORCING_START = 0.5
TEACHER_FORCING_END = 0.0
TEACHER_FORCING_DECAY = 0.95

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Blackwell RTX PRO 6000 enforced via CUDA_VISIBLE_DEVICES='0'; cuda:0 → physical GPU 0
NUM_WORKERS = 4              # USB4지만 CPU 전처리 병렬화 위해 복구 (데이터 로딩 병목 해소)
PIN_MEMORY = True            # 페이지드 메모리 사용 (USB4에서 도움)

# AMP (Automatic Mixed Precision) - 필수 (VRAM 절약)
USE_AMP = True
scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == "cuda" else None

# Gradient Checkpointing (메모리 절약) - 배치 키우면서 유지
USE_GRADIENT_CHECKPOINTING = True

# CLI 오버라이드 적용 (기본값은 CONFIG 값 유지)
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
if args.accum_steps is not None:
    ACCUM_STEPS = args.accum_steps
if args.num_workers is not None:
    NUM_WORKERS = args.num_workers
if args.amp is not None:
    USE_AMP = True
if args.no_amp is not None:
    USE_AMP = False
if args.gradient_checkpointing is not None:
    USE_GRADIENT_CHECKPOINTING = True
if args.no_gradient_checkpointing is not None:
    USE_GRADIENT_CHECKPOINTING = False

print(f"[CONFIG] DEVICE: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[CONFIG] GPUs available: {torch.cuda.device_count()}")
    print(f"[CONFIG] AMP: {USE_AMP}")
    print(f"[CONFIG] LR: {LR}, Batch: {BATCH_SIZE}, Accum: {ACCUM_STEPS}, Epochs: {NUM_EPOCHS}")
    print(f"[CONFIG] IMG_DIR: {IMG_DIR}")
    print(f"[CONFIG] LABEL_CSV: {LABEL_CSV}")
    print(f"[CONFIG] CHECKPOINT_DIR: {CHECKPOINT_DIR}")


    # ============================================================
    # DATASET (개선: 데이터 증강 추가)
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
        """데이터 증강: 회전, 확대/축소, 대비 조절, 잡음"""
        # Random rotation (-5 to 5 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Random scale (0.9 to 1.1)
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            # Resize back to original
            img = cv2.resize(img, (w, h))

        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)     # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Random noise
        if random.random() > 0.7:
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

        return img, target


# ============================================================
# LR Scheduler (개선: Warmup + Cosine Annealing)
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
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


# ============================================================
# TRAINING
# ============================================================
def get_teacher_forcing_ratio(epoch, start=TEACHER_FORCING_START, end=TEACHER_FORCING_END, decay=TEACHER_FORCING_DECAY):
    """Teacher forcing ratio를 epoch에 따라 감소시킴"""
    ratio = start * (decay ** epoch)
    return max(ratio, end)


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
                loss = criterion(
                    outputs[:, :-1].reshape(-1, outputs.size(-1)),
                    targets[:, 1:].reshape(-1)
                )
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
            loss = criterion(
                outputs[:, :-1].reshape(-1, outputs.size(-1)),
                targets[:, 1:].reshape(-1)
            )
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
            optimizer.zero_grad()
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

            loss = criterion(
                outputs[:, :-1].reshape(-1, outputs.size(-1)),
                targets[:, 1:].reshape(-1)
            )
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
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    df = pd.read_csv(LABEL_CSV)
    labels = df['label'].tolist()

    char2idx, idx2char = build_char_dict(labels)
    vocab_size = len(char2idx)
    print(f"[VOCAB] Size: {vocab_size}, chars: {list(char2idx.keys())}")

    with open(CHAR_DICT_PATH, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    print(f"[SAVED] char_dict.pkl -> {CHAR_DICT_PATH}")

    # 데이터 증강을 적용한 transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = HandwritingDataset(LABEL_CSV, IMG_DIR, char2idx, transform, augment=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = CRNN(vocab_size, hidden_size=384, dropout=DROPOUT, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING).to(DEVICE)
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.step()
        print(f"\n[Epoch {epoch}] LR: {current_lr:.6f}, TF_Ratio: {get_teacher_forcing_ratio(epoch):.2f}")

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
                    'hidden_size': 256,
                    'dropout': DROPOUT,
                    'max_label_length': MAX_LABEL_LENGTH,
                    'image_height': IMAGE_HEIGHT,
                    'image_width': IMAGE_WIDTH,
                    'use_amp': USE_AMP,
                    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
                    'batch_size': BATCH_SIZE,
                    'accum_steps': ACCUM_STEPS,
                }
            }, BEST_MODEL_PATH)
            print(f"[SAVED] Best model -> {BEST_MODEL_PATH} (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"[EARLY STOP] Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if epoch % 10 == 0:
            periodic_path = CHECKPOINT_DIR / f"epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, periodic_path)
            print(f"[SAVED] Periodic -> {periodic_path}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping training.")
            break

    print(f"\n[DONE] Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()