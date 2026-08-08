#!/usr/bin/env python3
"""
v4 — resnet18(ImageNet pretrained) 백본 + 비율유지 패딩 전처리 재학습
======================================================================
v3_1(SEBlock, 30.4%)의 개선 실험:
  1. encoder: SEBlock CNN 5블록 → resnet18 layer3 (ImageNet pretrained)
     - 특징맵 해상도: 16열 → 64열 (stride 32 → 16)  = 저빈도 미세철자 포착 여지
     - 사전학습 전이: 저수준 획/에지 특징
  2. 전처리: 256x64(4:1 왜곡) → 256x128(비율유지 패딩)
     - 128x128 원본에서 형태 보존 (인코더 "못 읽음"의 주 용의자 제거)

동일 비교 기준:
  - val: v3 클린 고정 split (1,116장) — v3_1/v3 exp2와 동일
  - 지표: exact / CER / 빈도그룹 / beam oracle — eval_v3_2.py로 측정

실행 (GPU1 Max-Q, vLLM과 공존, 여유 2.9GB):
  CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_2.py
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
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    p.add_argument('--freeze-backbone', action='store_true', help='resnet backbone 파라미터 동결 (전이학습 실험)')
    return p.parse_args()

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# v4 모델/전처리 import
V3_2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_2 / "model"))
from model_v3_2 import (
    CRNN, build_char_dict, encode_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, SPECIAL_TOKENS
)
from preprocess_v3_2 import load_resize_pad, preprocess_tensor

# ============================================================
# CONFIG — v4
# ============================================================
V2_ORIG_IMG = Path("/home/dev/doctor_ocr_v2/dataset/img/img")
DATA_ROOT = V3_2 / "data" / "exp2_clean"
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "combined_labels.csv"
FIXED_VAL_CSV = V3_2 / "data" / "clean_split" / "val.csv"

WORK_DIR = V3_2 / "working"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict_v3_2.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model_v3_2.pth"

# 하이퍼파라미터 (v3_1 유지, but backbone pretrained → LR 낮춤)
BATCH_SIZE = 24          # 256x128이라 v3_1(40)보다 작게 — VRAM 실측 후 조정
ACCUM_STEPS = 6          # effective batch 144
NUM_EPOCHS = 60
LR = 1e-4                # pretrained backbone fine-tune엔 3e-4보다 낮게
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
DROPOUT = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
HIDDEN_SIZE = 384
TEACHER_FORCING_START = 0.5
TEACHER_FORCING_END = 0.0
TEACHER_FORCING_DECAY = 0.95
FREEZE_BACKBONE = args.freeze_backbone

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

print(f"[CONFIG] v3_2: LR={LR}, Batch={BATCH_SIZE}, Accum={ACCUM_STEPS}, Epochs={NUM_EPOCHS}")
print(f"[CONFIG] backbone: resnet18 ImageNet pretrained, freeze={FREEZE_BACKBONE}")
print(f"[CONFIG] preprocess: {IMAGE_WIDTH}x{IMAGE_HEIGHT} 비율유지 패딩 (v3_1 256x64 왜곡 → 개선)")
print(f"[CONFIG] Device: {DEVICE}, AMP={USE_AMP}, GC={USE_GRADIENT_CHECKPOINTING}")


# ============================================================
# DATASET — v4 전처리 (비율유지 패딩 + ImageNet normalize)
# ============================================================
class HandwritingDataset(Dataset):
    def __init__(self, csv_path, img_dir, char2idx, max_len=MAX_LABEL_LENGTH):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.char2idx = char2idx
        self.max_len = max_len

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

        img = load_resize_pad(img_path, IMAGE_HEIGHT, IMAGE_WIDTH)   # (H,W,3) float [0,1]
        img = preprocess_tensor(img)                                  # (3,H,W) ImageNet norm
        img = torch.from_numpy(img).float()

        target = encode_label(label, self.char2idx, self.max_len)
        target = torch.tensor(target, dtype=torch.long)
        return img, target


# ============================================================
# LR Scheduler / TF (v3_1 재사용)
# ============================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, group_idx=None):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        # group_idx 지정 시 해당 그룹만 조정 (backbone 그룹0은 고정, head 그룹1만 스케줄)
        if group_idx is not None:
            self.optimizer.param_groups[group_idx]['lr'] = lr
        else:
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

    df = pd.read_csv(LABEL_CSV)
    labels = df['label'].tolist()
    char2idx, idx2char = build_char_dict(labels)
    vocab_size = len(char2idx)
    print(f"[VOCAB] Size={vocab_size}, chars={sorted(char2idx.keys())}")

    with open(CHAR_DICT_PATH, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    print(f"[SAVED] char_dict_v3_2.pkl -> {CHAR_DICT_PATH}")

    train_dataset = HandwritingDataset(LABEL_CSV, IMG_DIR, char2idx)
    val_dataset = HandwritingDataset(FIXED_VAL_CSV, V2_ORIG_IMG, char2idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"[DATA] Train={len(train_dataset)}, Val(fixed)={len(val_dataset)}")

    model = CRNN(vocab_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                 use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                 pretrained=True).to(DEVICE)

    # 백본 동결 옵션
    if FREEZE_BACKBONE:
        for name, p in model.named_parameters():
            if name.startswith('encoder.'):
                p.requires_grad = False
        print("[CONFIG] resnet backbone FROZEN (BiLSTM+decoder만 학습)")
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())

    print(f"[MODEL] Params={sum(p.numel() for p in model.parameters()):,} "
          f"(trainable={sum(p.numel() for p in trainable):,})")

    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<PAD>'])

    # ============================================================
    # 옵티마이저 — 파라미터 그룹별 LR 개별화 (2026-08-07 재설정)
    # ============================================================
    # 문제: 처음엔 backbone/RNN/decoder 모두 1e-4 → backbone은 ImageNet 적응이
    #       너무 느리고, 스크래치 초기화된 RNN+decoder는 더더욱 느림 → epoch10 acc 0.27%.
    # 해결: 새로 초기화된 부분(RNN+decoder)은 높은 LR, 사전학습 backbone은 낮은 LR.
    #       backbone도 결국 미세조정돼야 하지만, 초반엔 RNN/decoder가 먼저
    #       '문자를 찍는 법'을 배워야 하므로 backbone은 상대적으로 천천히.
    backbone_lr = LR                     # 1e-4 — pretrained backbone
    head_lr = LR * 5                     # 5e-4 — 스크래치 BiLSTM+decoder
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('encoder.'):
            backbone_params.append(p)
        else:
            head_params.append(p)
    print(f"[OPT] backbone_lr={backbone_lr}, head_lr={head_lr} "
          f"(backbone {len(backbone_params)} params, head {len(head_params)} params)")

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr},
    ], weight_decay=WEIGHT_DECAY)

    # 스케줄러: head 그룹(그룹1)만 cosine. backbone(그룹0)은 고정 LR 유지 → step(1).
    # base_lr은 head_lr(5e-4) 기준 — head가 warmup/decay됨.
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, head_lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.step(group_idx=1)
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
                    'backbone': 'resnet18',
                    'pretrained': True,
                    'freeze_backbone': FREEZE_BACKBONE,
                    'hidden_size': HIDDEN_SIZE,
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
            periodic_path = CHECKPOINT_DIR / f"epoch_v3_2_{epoch}.pth"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc,
                        'vocab_size': vocab_size}, periodic_path)
            print(f"[SAVED] Periodic -> {periodic_path}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
            break

    print(f"\n[DONE] v3_2 training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
