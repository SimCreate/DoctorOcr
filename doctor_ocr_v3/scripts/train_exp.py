#!/usr/bin/env python3
"""
doctor_ocr_v3 - 실험군별 학습 스크립트 (CTC 기반 CRNN, v2_2 로직 재사용)

v2_2 local_train_v2_2.py와 동일한 로직/하이퍼파라미터를 유지하되,
--exp 인자로 데이터 경로와 체크포인트 디렉토리를 실험군(1/2/3)별로 분리한다.

- exp 1: data/experiment_1 (원본 only)
- exp 2: data/experiment_2 (실사 증강 2배, 온디스크)
- exp 3: data/experiment_3 (실사 증강 + 저빈도 합성 12.5%)

GPU: GPU1 Max-Q (CUDA_VISIBLE_DEVICES=1), vLLM과 공존. 반드시 백그라운드로.

실행:
  /home/dev/doctor_ocr_v2_2/venv/bin/python train_exp.py --exp 1 \
      > logs/train_exp1.log 2>&1 &
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
    p.add_argument('--exp', type=int, required=True, choices=[1, 2, 3],
                   help='실험군 번호 (데이터/체크포인트 경로 결정)')
    p.add_argument('--device', type=str, default='1',
                   help='CUDA_VISIBLE_DEVICES (기본 1 = GPU1 Max-Q)')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accum-steps', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# ============================================================
# PATH - v3 전용 (실험군별 분리)
# ============================================================
V3 = Path("/home/dev/DoctorOcr/doctor_ocr_v3")
EXP = args.exp
DATA_ROOT = V3 / "data" / f"experiment_{EXP}"
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "combined_labels.csv"

WORK_DIR = V3 / "working" / f"exp{EXP}"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# v2_2 모델 정의 재사용
sys.path.insert(0, "/home/dev/doctor_ocr_v2_2/model")
from model_v2_2 import (
    CRNN, build_char_dict, encode_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CTC_BLANK, SPECIAL_TOKENS
)

# ============================================================
# CONFIG - v2_2와 동일
# ============================================================
BATCH_SIZE = args.batch_size
ACCUM_STEPS = args.accum_steps
NUM_EPOCHS = args.epochs
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
DROPOUT = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
NUM_WORKERS = args.num_workers
PIN_MEMORY = True
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == "cuda" else None

# 결과 디렉토리 생성
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[CONFIG] EXP={EXP} | DEVICE={DEVICE}")
if DEVICE.type == "cuda":
    free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) >> 20
    print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}, Free VRAM: {free_mb} MiB")
print(f"[CONFIG] LR={LR}, Batch={BATCH_SIZE}, Accum={ACCUM_STEPS}, Epochs={NUM_EPOCHS}")
print(f"[CONFIG] LABEL_CSV={LABEL_CSV}")
print(f"[CONFIG] CHECKPOINT_DIR={CHECKPOINT_DIR}")


# ============================================================
# DATASET (v2_2와 동일, augment 유지 — 실험군 간 공정 비교)
# ============================================================
class HandwritingDataset(Dataset):
    def __init__(self, csv_path, img_dir, char2idx, transform=None,
                 max_len=MAX_LABEL_LENGTH, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.char2idx = char2idx
        self.transform = transform
        self.max_len = max_len
        self.augment = augment

        self.valid_indices = []
        for idx, row in self.df.iterrows():
            if (self.img_dir / row['filename']).exists():
                self.valid_indices.append(idx)
        print(f"[DATASET] Total: {len(self.df)}, Valid: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def augment_image(self, img):
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            img = cv2.resize(img, (w, h))
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
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


def ctc_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)
    return imgs, targets_concat, target_lengths


# ============================================================
# LR Scheduler (v2_2와 동일)
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
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1
        return lr


# ============================================================
# TRAINING (v2_2와 동일)
# ============================================================
def train_epoch(model, loader, optimizer, device, epoch, accum_steps=ACCUM_STEPS, use_amp=USE_AMP):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for step, (imgs, targets, target_lengths) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = model.ctc_head.get_loss(logits, targets, target_lengths)
                loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(imgs)
            loss = model.ctc_head.get_loss(logits, targets, target_lengths)
            loss = loss / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

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


def validate(model, loader, device, idx2char, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, targets, target_lengths in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(imgs)
            loss = model.ctc_head.get_loss(logits, targets, target_lengths)
            total_loss += loss.item()

            decoded = model.ctc_head.decode(logits)
            target_list = targets.tolist()
            offset = 0
            for i, t_len in enumerate(target_lengths.tolist()):
                true_seq = target_list[offset:offset + t_len]
                pred_seq = decoded[i]
                true_str = decode_sequence(true_seq, idx2char)
                pred_str = decode_sequence(pred_seq, idx2char)
                if pred_str == true_str:
                    correct += 1
                total += 1
                offset += t_len

    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader)
    print(f"[VAL] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f} ({correct}/{total})")
    return avg_loss, acc


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(LABEL_CSV)
    labels = df['label'].tolist()
    char2idx, idx2char = build_char_dict(labels)
    vocab_size = len(char2idx)
    print(f"[VOCAB] Size: {vocab_size}, blank_idx={CTC_BLANK}")

    with open(CHAR_DICT_PATH, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    print(f"[SAVED] char_dict.pkl -> {CHAR_DICT_PATH}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = HandwritingDataset(LABEL_CSV, IMG_DIR, char2idx, transform, augment=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 실험군 간 동일 분할
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=ctc_collate_fn)

    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = CRNN(vocab_size, hidden_size=256, dropout=DROPOUT,
                 use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING).to(DEVICE)
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.step()
        print(f"\n[Epoch {epoch}] LR: {current_lr:.6f}")

        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, DEVICE, idx2char, epoch)

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
                'exp': EXP,
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

    print(f"\n[DONE] Training complete (exp{EXP}). Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
