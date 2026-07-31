#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Evaluation & Inference Script
Loads best_model.pth and runs evaluation on validation set + sample predictions
"""

import os
import sys
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# 모델 정의 임포트
sys.path.insert(0, str(Path(__file__).parent / "model"))
from model_v2_2 import (
    CRNN, build_char_dict, encode_label, decode_sequence,
    MAX_LABEL_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CTC_BLANK, SPECIAL_TOKENS
)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("/home/dev/doctor_ocr_v2_2/dataset")
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "doctor_handwriting_labels.csv"

WORK_DIR = Path("/home/dev/doctor_ocr_v2_2/working")
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# 학습 하이퍼파라미터 (체크포인트 config와 일치해야 함)
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
DROPOUT = 0.3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Blackwell cuda0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Blackwell RTX PRO 6000 enforced: CUDA_VISIBLE_DEVICES='0' → cuda:0 = GPU 0

# ============================================================
# DATASET (동일 정의 - 학습 시와 일치해야 함)
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


def print_samples(predictions, n=20):
    print(f"\n=== Sample Predictions (first {n}) ===")
    for i, p in enumerate(predictions[:n]):
        status = "✓" if p['match'] else "✗"
        print(f"{status} True: '{p['true']}' | Pred: '{p['pred']}' | File: {Path(p['path']).name}")
    
    # 잘못된 예측만 모아서 보기
    wrong = [p for p in predictions if not p['match']]
    if wrong:
        print(f"\n=== Wrong Predictions ({len(wrong)}/{len(predictions)}) ===")
        for p in wrong[:30]:
            print(f"✗ True: '{p['true']}' | Pred: '{p['pred']}' | File: {Path(p['path']).name}")


def main():
    # 체크포인트 로드
    print(f"[LOAD] Loading checkpoint: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    config = checkpoint.get('config', {})
    vocab_size = checkpoint.get('vocab_size')
    
    print(f"[CONFIG] vocab_size={vocab_size}")
    print(f"[CONFIG] hidden_size={config.get('hidden_size')}, dropout={config.get('dropout')}")
    print(f"[CONFIG] batch_size={config.get('batch_size')}, accum_steps={config.get('accum_steps')}")
    print(f"[CONFIG] use_amp={config.get('use_amp')}, use_gradient_checkpointing={config.get('use_gradient_checkpointing')}")
    
    # char_dict 로드
    with open(CHAR_DICT_PATH, 'rb') as f:
        char_dict = pickle.load(f)
    char2idx = char_dict['char2idx']
    idx2char = char_dict['idx2char']
    print(f"[VOCAB] Loaded char_dict: {len(char2idx)} tokens")
    
    # 모델 생성 및 가중치 로드
    model = CRNN(
        vocab_size=vocab_size,
        hidden_size=config.get('hidden_size', 256),
        dropout=config.get('dropout', DROPOUT),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', USE_GRADIENT_CHECKPOINTING)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[MODEL] Loaded weights from epoch {checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss'):.4f}")
    
    # 데이터 로더 (검증 세트와 동일하게 split)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = HandwritingDataset(LABEL_CSV, IMG_DIR, char2idx, transform, augment=False)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 학습 시와 동일한 split 재현 (seed 고정)
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=ctc_collate_fn)
    
    print(f"[DATA] Val set: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # 평가 실행
    val_loss, val_acc, predictions = evaluate_model(model, val_loader, DEVICE, idx2char)
    
    # 샘플 출력
    print_samples(predictions, n=30)
    
    # 문자별 정확도 분석
    print("\n=== Per-Character Analysis ===")
    char_correct = {}
    char_total = {}
    for p in predictions:
        true_str = p['true']
        pred_str = p['pred']
        for i, ch in enumerate(true_str):
            char_total[ch] = char_total.get(ch, 0) + 1
            if i < len(pred_str) and pred_str[i] == ch:
                char_correct[ch] = char_correct.get(ch, 0) + 1
    
    print("Char | Total | Correct | Acc")
    print("-----|-------|---------|-----")
    for ch in sorted(char_total.keys(), key=lambda x: char_total[x], reverse=True)[:30]:
        c = char_correct.get(ch, 0)
        t = char_total[ch]
        print(f"  {ch}  |  {t:3d}  |   {c:3d}   | {c/t:.3f}")

    # 결과 저장
    output_path = WORK_DIR / "eval_results.csv"
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"\n[SAVED] Evaluation results -> {output_path}")


if __name__ == "__main__":
    main()