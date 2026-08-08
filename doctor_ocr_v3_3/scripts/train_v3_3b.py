#!/usr/bin/env python3
"""
v3_3 — Hybrid(resnet18 + Attention + CTC) 학습
================================================
기반 v3_2(attention 34.7%)에 CTC 헤드를 병렬 추가한 하이브리드.
  손실: L = λ·L_ctc + (1-λ)·L_attn   (기본 λ=0.5)
  목표: attention의 고빈도 강점(62.3%) + CTC의 중·저빈도·CER 강점 결합

기반 v3_2 대비 변경:
  - model_v3_3 (이중 헤드)
  - 데이터로더: attention target + CTC target(변수길이) 동시 반환
  - collate: CTC용 (concat + lengths)
  - 손실: 하이브리드 (ctc + cross_entropy)
  - 검증: attention beam exact + CTC greedy exact 둘 다 기록

실행 (GPU1 Max-Q, vLLM 공존):
  CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/train_v3_3b.py

레거시 반영:
  - 클린 val 1,116 (구조 분리, 리키지 방지) — 기반 v3_2와 동일
  - CTCHead/get_loss/decode: v2_2 검증 구현 재사용
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

V3_3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_3 / "model"))

from model_v3_3 import (
    CRNN, build_char_dict, build_ctc_char_dict,
    encode_label, encode_ctc_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, SPECIAL_TOKENS,
)
from preprocess_v3_3 import load_resize_pad, preprocess_tensor

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = V3_3 / "data" / "exp2_clean"
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "combined_labels.csv"
FIXED_VAL_CSV = V3_3 / "data" / "clean_split" / "val.csv"
V2_ORIG_IMG = Path("/home/dev/doctor_ocr_v2/dataset/img/img")

WORK_DIR = V3_3 / "working"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict_v3_3.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model_v3_3b.pth"

BATCH_SIZE = 24
ACCUM_STEPS = 6          # effective 144 (기반 v3_2와 동일)
NUM_EPOCHS = 60
LR = 1e-4                # backbone LR
LR_HEAD_MULT = 5         # head(attention+ctc) LR 배수
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
DROPOUT = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
TEACHER_FORCING_START = 0.4    # v3_3b: attention 자기회귀 적응 위해 낮춤
TEACHER_FORCING_END = 0.05     # v3_3b: 0.05까지 강감쇠 (v3_3=0.1)
TEACHER_FORCING_DECAY = 0.90
CTC_WEIGHT = 0.3               # λ=0.3 — v3_3b: attention에 무게 (v3_3=0.5)
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
NUM_WORKERS = 4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == "cuda" else None


# ============================================================
# DATASET — attention + CTC 듀얼 타겟
# ============================================================
class HybridDataset(Dataset):
    def __init__(self, csv_path, img_dir, attn_char2idx, ctc_char2idx, max_len=MAX_LABEL_LENGTH):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.attn_c2i = attn_char2idx
        self.ctc_c2i = ctc_char2idx
        self.max_len = max_len

        self.valid_indices = []
        for idx, row in self.df.iterrows():
            if (self.img_dir / row['filename']).exists():
                self.valid_indices.append(idx)
        print(f"[DATASET] {csv_path} Total={len(self.df)}, Valid={len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]
        img_path = self.img_dir / row['filename']
        label = str(row['label'])

        img = load_resize_pad(img_path, IMAGE_HEIGHT, IMAGE_WIDTH)
        img = preprocess_tensor(img)
        img = torch.from_numpy(img).float()                    # (3,H,W)

        attn_t = torch.tensor(encode_label(label, self.attn_c2i, self.max_len), dtype=torch.long)
        ctc_t = torch.tensor(encode_ctc_label(label, self.ctc_c2i, self.max_len), dtype=torch.long)
        return img, attn_t, ctc_t


def hybrid_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    attn = torch.stack([b[1] for b in batch])           # [B, L] padded
    ctcs = [b[2] for b in batch]
    ctc_lens = torch.tensor([len(t) for t in ctcs], dtype=torch.long)
    ctc_concat = torch.cat(ctcs)
    return imgs, attn, ctc_concat, ctc_lens


# ============================================================
# LR Scheduler (head 그룹만 cosine — backbone 고정)
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
        if group_idx is not None:
            self.optimizer.param_groups[group_idx]['lr'] = lr
        else:
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        self.current_epoch += 1
        return lr


def get_teacher_forcing_ratio(epoch, start=TEACHER_FORCING_START, end=TEACHER_FORCING_END, decay=TEACHER_FORCING_DECAY):
    return max(start * (decay ** epoch), end)


# ============================================================
# TRAIN EPOCH (하이브리드 손실)
# ============================================================
def train_epoch(model, loader, optimizer, device, epoch, accum_steps=ACCUM_STEPS, use_amp=USE_AMP):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for step, (imgs, attn, ctc_concat, ctc_lens) in enumerate(pbar):
        imgs = imgs.to(device)
        attn = attn.to(device)
        ctc_concat = ctc_concat.to(device)
        ctc_lens = ctc_lens.to(device)

        tf_ratio = get_teacher_forcing_ratio(epoch)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                attn_logits, ctc_logits = model(imgs, attn, ctc_concat, ctc_lens, tf_ratio)
                loss, ctc_l, ce = model.hybrid_loss(attn_logits, attn, ctc_logits, ctc_concat, ctc_lens)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            attn_logits, ctc_logits = model(imgs, attn, ctc_concat, ctc_lens, tf_ratio)
            loss, ctc_l, ce = model.hybrid_loss(attn_logits, attn, ctc_logits, ctc_concat, ctc_lens)
            loss = loss / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{loss.item()*accum_steps:.4f}', 'ctc': f'{ctc_l.item():.3f}', 'ce': f'{ce.item():.3f}'})

    # 트레일링 배치
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


# ============================================================
# VALIDATE — CTC greedy exact (주 기준) + attention beam exact (5에폭 간격)
# ============================================================
def validate(model, loader, device, attn_idx2char, ctc_idx2char, epoch, check_attn=False):
    model.eval()
    total_loss = 0
    ctc_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, attn, ctc_concat, ctc_lens in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            imgs = imgs.to(device)
            attn = attn.to(device)
            ctc_concat = ctc_concat.to(device)
            ctc_lens = ctc_lens.to(device)

            # eval에서 attention 디코더는 beam_search 전용 → CTC loss + ctc head만 사용
            feat = model.encoder(imgs)
            feat = model.rnn(feat)
            ctc_logits = model.ctc_head(feat)
            ctc_loss = model.ctc_head.get_loss(ctc_logits, ctc_concat, ctc_lens)
            total_loss += ctc_loss.item()

            # CTC greedy exact
            ctc_preds = model.ctc_head.decode(ctc_logits)
            offset = 0
            for i, t_len in enumerate(ctc_lens.tolist()):
                true_toks = ctc_concat[offset:offset + t_len].tolist()
                true_str = decode_sequence(true_toks, ctc_idx2char)
                pred_str = decode_sequence(ctc_preds[i], ctc_idx2char)
                if pred_str == true_str:
                    ctc_correct += 1
                total += 1
                offset += t_len

    ctc_acc = ctc_correct / total if total else 0
    avg_loss = total_loss / len(loader)

    # attention beam 정확도 (선택 시 — val 전체 앞 200장만, 빠르게)
    attn_acc = -1.0
    if check_attn:
        attn_corr = 0
        attn_tot = 0
        sub = 200
        for imgs, attn, ctc_concat, ctc_lens in tqdm(loader, desc=f"Epoch {epoch} [Val-Attn]"):
            imgs = imgs.to(device)
            feat = model.encoder(imgs)
            feat = model.rnn(feat)
            for b in range(imgs.size(0)):
                beams = model.decoder.beam_search(feat[b:b+1], beam_width=3)
                pred_str = decode_sequence(beams[0], attn_idx2char)
                # attn 타겟에서 label 복원
                toks = attn[b].tolist()
                lbl = []
                for t in toks:
                    if t == SPECIAL_TOKENS['<EOS>']:
                        break
                    if t in attn_idx2char and attn_idx2char[t] not in ['<SOS>', '<PAD>', '<UNK>']:
                        lbl.append(attn_idx2char[t])
                true_str = ''.join(lbl)
                if pred_str == true_str:
                    attn_corr += 1
                attn_tot += 1
                if attn_tot >= sub:
                    break
            if attn_tot >= sub:
                break
        attn_acc = attn_corr / attn_tot if attn_tot else -1.0

    print(f"[VAL] Epoch {epoch}: Loss={avg_loss:.4f}, CTC Acc={ctc_acc:.4f} ({ctc_correct}/{total})"
          + (f", Attn beam(200)={attn_acc:.4f}" if check_attn else ""))
    return avg_loss, ctc_acc


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    df = pd.read_csv(LABEL_CSV)
    labels = df['label'].astype(str).tolist()

    attn_c2i, attn_i2c = build_char_dict(labels)
    ctc_c2i, ctc_i2c = build_ctc_char_dict(labels)
    print(f"[VOCAB] attn={len(attn_c2i)}, ctc={len(ctc_c2i)}")

    with open(CHAR_DICT_PATH, 'wb') as f:
        pickle.dump({'attn_char2idx': attn_c2i, 'attn_idx2char': attn_i2c,
                     'ctc_char2idx': ctc_c2i, 'ctc_idx2char': ctc_i2c}, f)
    print(f"[SAVED] {CHAR_DICT_PATH}")

    train_ds = HybridDataset(LABEL_CSV, IMG_DIR, attn_c2i, ctc_c2i)
    val_ds = HybridDataset(FIXED_VAL_CSV, V2_ORIG_IMG, attn_c2i, ctc_c2i)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=hybrid_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=hybrid_collate)
    print(f"[DATA] Train={len(train_ds)}, Val={len(val_ds)}")

    model = CRNN(vocab_size=len(attn_c2i), ctc_vocab_size=len(ctc_c2i),
                 hidden_size=384, dropout=DROPOUT,
                 use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                 pretrained=True, ctc_weight=CTC_WEIGHT).to(DEVICE)
    print(f"[MODEL] Params={sum(p.numel() for p in model.parameters()):,}")

    # LR 그룹별 개별화 (backbone 낮게 / head 높게)
    backbone_lr = LR
    head_lr = LR * LR_HEAD_MULT
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if name.startswith('encoder.'):
            backbone_params.append(p)
        else:
            head_params.append(p)
    print(f"[OPT] backbone_lr={backbone_lr}, head_lr={head_lr} ({len(backbone_params)}/{len(head_params)} tensors)")

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, head_lr)

    best_ctc_acc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.step(group_idx=1)
        print(f"\n[Epoch {epoch}] LR={current_lr:.6f}, TF={get_teacher_forcing_ratio(epoch):.2f}, λ_ctc={CTC_WEIGHT}")

        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        check_attn = (epoch % 5 == 0)
        val_loss, val_acc = validate(model, val_loader, DEVICE, attn_i2c, ctc_i2c, epoch, check_attn=check_attn)

        # best 저장: CTC acc 기준 (하이브리드 목표 = 중·저빈도·CER 개선)
        if val_acc > best_ctc_acc:
            best_ctc_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ctc_acc': val_acc,
                'vocab_size': len(attn_c2i),
                'ctc_vocab_size': len(ctc_c2i),
                'config': {
                    'hidden_size': 384,
                    'dropout': DROPOUT,
                    'max_label_length': MAX_LABEL_LENGTH,
                    'image_height': IMAGE_HEIGHT,
                    'image_width': IMAGE_WIDTH,
                    'use_amp': USE_AMP,
                    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
                    'batch_size': BATCH_SIZE,
                    'accum_steps': ACCUM_STEPS,
                    'ctc_weight': CTC_WEIGHT,
                }
            }, BEST_MODEL_PATH)
            print(f"[SAVED] Best model -> {BEST_MODEL_PATH} (val_ctc_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"[EARLY STOP] Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if epoch % 10 == 0:
            periodic_path = CHECKPOINT_DIR / f"epoch_v3_3b_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ctc_acc': val_acc,
            }, periodic_path)
            print(f"[SAVED] Periodic -> {periodic_path}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping training.")
            break

    print(f"\n[DONE] v3_3 training complete. Best val_ctc_acc: {best_ctc_acc:.4f}")


if __name__ == "__main__":
    main()
