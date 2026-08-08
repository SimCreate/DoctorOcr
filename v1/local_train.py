#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Local Training Script
Patched for /home/dev/doctor_ocr/ paths, torch cu128, Blackwell sm_120
Original: Kaggle notebook (sutharamanikanta/notebookd701bd0fcb) - failure quality
"""

import os
import sys
import json
import pickle
import random
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

# Apply CLI overrides BEFORE model creation
if args.device is None:
    args.device = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# ============================================================
# CONFIG - HARDCODED PATHS FOR /home/dev/doctor_ocr/
# ============================================================
DATA_ROOT = Path("/home/dev/doctor_ocr/dataset")
IMG_DIR = DATA_ROOT / "img" / "img"          # 실제 이미지 폴더
LABEL_CSV = DATA_ROOT / "doctor_handwriting_labels.csv"

WORK_DIR = Path("/home/dev/doctor_ocr/working")
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# 학습 하이퍼파라미터
BATCH_SIZE = 8              # GPU 사용: 배치 8
ACCUM_STEPS = 2             # 그래디언트 누적으로 유효 배치 16
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 로딩
NUM_WORKERS = 2
PIN_MEMORY = True

# 모델 설정
MAX_LABEL_LENGTH = 64
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] DEVICE: {DEVICE}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')})")
if DEVICE.type == "cuda":
    print(f"[CONFIG] GPU (mapped): {torch.cuda.get_device_name(0)}, UUID={torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0),'uuid') else 'N/A'}")
print(f"[CONFIG] GPUs available: {torch.cuda.device_count()}")
print(f"[CONFIG] IMG_DIR: {IMG_DIR}")
print(f"[CONFIG] LABEL_CSV: {LABEL_CSV}")
print(f"[CONFIG] CHECKPOINT_DIR: {CHECKPOINT_DIR}")

# CLI 오버라이드 적용 (기본값은 CONFIG 값 유지)
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
if args.accum_steps is not None:
    ACCUM_STEPS = args.accum_steps
if args.num_workers is not None:
    NUM_WORKERS = args.num_workers

# ============================================================
# CHARACTER TOKENIZER
# ============================================================
SPECIAL_TOKENS = {
    '<PAD>': 0,     # Padding 
    '<SOS>': 1,     # 시작 Start of Sequence
    '<EOS>': 2,     # 종료 End of Sequence
    '<UNK>': 3,     # 미확인 Unknown
}

def build_char_dict(labels):
    """라벨에서 문자 집합 구축"""
    chars = set() # python은 집합 구축할 때 set() 사용 (중복 제거) - dict 는 {} (Key-Value)
    for label in labels:
        for ch in str(label):
            chars.add(ch)
    chars = sorted(list(chars))
    
    char2idx = SPECIAL_TOKENS.copy()
    idx = len(SPECIAL_TOKENS)
    for ch in chars:
        char2idx[ch] = idx
        idx += 1
    
    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char

def encode_label(label, char2idx, max_len=MAX_LABEL_LENGTH):
    """라벨을 토큰 시퀀스로 인코딩"""
    tokens = [char2idx['<SOS>']]
    for ch in str(label):
        tokens.append(char2idx.get(ch, char2idx['<UNK>']))
    tokens.append(char2idx['<EOS>'])
    
    # 패딩
    if len(tokens) < max_len:
        tokens += [char2idx['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

def decode_sequence(seq, idx2char):
    """토큰 시퀀스를 문자열로 디코딩"""
    chars = []
    for idx in seq:
        if idx == SPECIAL_TOKENS['<EOS>'] or idx == SPECIAL_TOKENS['<PAD>']:
            break
        if idx in idx2char and idx2char[idx] not in ['<SOS>', '<PAD>', '<EOS>', '<UNK>']:
            chars.append(idx2char[idx])
    return ''.join(chars)

# ============================================================
# DATASET
# ============================================================
class HandwritingDataset(Dataset):
    def __init__(self, csv_path, img_dir, char2idx, transform=None, max_len=MAX_LABEL_LENGTH):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.char2idx = char2idx
        self.transform = transform
        self.max_len = max_len
        
        # 파일 존재 확인
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
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        img_path = self.img_dir / row['filename']
        label = row['label']
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            # 빈 이미지 생성 (fallback)
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 라벨 인코딩
        target = encode_label(label, self.char2idx, self.max_len)
        target = torch.tensor(target, dtype=torch.long)
        
        return img, target

# ============================================================
# MODEL: CRNN (CNN + BiLSTM + Attention Decoder)
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self, img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 32x128
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 16x64
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),  # 16x64
            # Block 4
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1)),  # 8x64
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),  # 8x64
            # Block 6
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1)),  # 4x64
            # Block 7
            nn.Conv2d(512, 512, 2, padding=0), nn.BatchNorm2d(512), nn.ReLU(True),  # 3x63
        )
        self.out_channels = 512
    
    def forward(self, x):
        return self.cnn(x)  # (B, 512, H, W) -> (B, 512, 3, 63)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=0.2
        )
        self.out_features = hidden_size * 2
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, W, C*H) for sequence
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x)
        return out  # (B, W, hidden*2)


class AttentionDecoder(nn.Module):
    def __init__(self, encoder_out_dim, vocab_size, hidden_size=256, max_len=MAX_LABEL_LENGTH):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        # [FIX] encoder_proj를 __init__에서 생성 (forward()마다 새로 만들던 버그 수정)
        self.encoder_proj = nn.Linear(encoder_out_dim, hidden_size)
    
    def forward(self, encoder_out, targets=None, teacher_forcing_ratio=0.5):
        # encoder_out: (B, seq_len, encoder_dim) -> (B, W, 512*3)
        B, seq_len, encoder_dim = encoder_out.shape
        
        # Project encoder output to hidden_size — 재사용
        encoder_proj = self.encoder_proj(encoder_out)
        
        # Start token
        decoder_input = torch.full((B, 1), SPECIAL_TOKENS['<SOS>'], dtype=torch.long, device=encoder_out.device)
        
        outputs = []
        hidden = None
        
        for t in range(self.max_len):
            embedded = self.embedding(decoder_input)  # (B, 1, hidden)
            
            # Attention
            attn_out, _ = self.attention(embedded, encoder_proj, encoder_proj)
            
            # Concat embedded + attention
            lstm_in = torch.cat([embedded, attn_out], dim=-1)
            
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            out = self.fc(lstm_out)  # (B, 1, vocab)
            outputs.append(out)
            
            # Teacher forcing
            if targets is not None and random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t:t+1]
            else:
                decoder_input = out.argmax(-1)
        
        outputs = torch.cat(outputs, dim=1)  # (B, max_len, vocab)
        return outputs


class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        encoder_out_dim = self.encoder.out_channels * 3  # 512 * 3 = 1536
        
        self.rnn = BiLSTM(encoder_out_dim, hidden_size=256)
        rnn_out_dim = self.rnn.out_features  # 512
        
        self.decoder = AttentionDecoder(rnn_out_dim, vocab_size)
    
    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        feat = self.encoder(x)  # (B, 512, 3, 63)
        feat = self.rnn(feat)   # (B, 63, 512)
        out = self.decoder(feat, targets, teacher_forcing_ratio)
        return out


# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, accum_steps=ACCUM_STEPS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for step, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # Forward with teacher forcing
        outputs = model(imgs, targets, teacher_forcing_ratio=0.5)
        
        # Loss: ignore PAD token
        # outputs: (B, max_len, vocab), targets: (B, max_len) with SOS at index 0
        # Compare outputs[:, :-1] (63 steps) to targets[:, 1:] (63 steps, exclude SOS)
        loss = criterion(
            outputs[:, :-1].reshape(-1, outputs.size(-1)),
            targets[:, 1:].reshape(-1)
        )
        
        # Normalize loss by accumulation steps
        loss = loss / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps  # rescale for logging
        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})
    
    # Handle remaining gradients
    if len(loader) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            
            outputs = model(imgs, targets=None, teacher_forcing_ratio=0.0)  # no teacher forcing
            
            loss = criterion(
                outputs[:, :-1].reshape(-1, outputs.size(-1)),
                targets[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
            
            # Accuracy: exact match
            preds = outputs.argmax(-1)
            for i in range(imgs.size(0)):
                pred_str = decode_sequence(preds[i].cpu().tolist(), idx2char)
                true_str = decode_sequence(targets[i, 1:].cpu().tolist(), idx2char)  # skip SOS
                if pred_str == true_str:
                    correct += 1
                total += 1
    
    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader)
    print(f"[VAL] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f} ({correct}/{total})")
    
    return avg_loss, acc


def main():
    # 시드 고정
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 데이터 로드
    df = pd.read_csv(LABEL_CSV)
    labels = df['label'].tolist()
    
    # 문자 사전 구축
    char2idx, idx2char = build_char_dict(labels)
    vocab_size = len(char2idx)
    print(f"[VOCAB] Size: {vocab_size}, chars: {list(char2idx.keys())}")
    
    # 문자 사전 저장
    with open(CHAR_DICT_PATH, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    print(f"[SAVED] char_dict.pkl -> {CHAR_DICT_PATH}")
    
    # Transform
    # 이미지 전처리 파이프라인

    transform = transforms.Compose([
        # ToTenser 역할
        # 1. Height - Weight - Channel to Channel - Height - Width 순서로 (Pytorch는 CHW 합성곱 연산을 지원)
        # 2. Gradient + 역전파 계산을 위해 정수를 소수점을 변환
        transforms.ToTensor(), 

        # 3. Normalize 정규화 (-1 ~ 1)
        # 이번 데이터셋은 필기채라 RGB 모두 동일하게 Normalize 적용
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # __init__함수만
    full_dataset = HandwritingDataset(LABEL_CSV, IMG_DIR, char2idx, transform)
    
    # Train/Val split (80/20)
    # 학습용 80%, 검증용 20%로 나누기
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 데이터셋 다음과 같이 로드
    # 8개 샘플씩 묶ㄱ음
    # 매 epoch마다 데이터 순서 무작위로 섞기
    # 2 way 병렬 처리
    # GPU 전송 최적화 Flag Pin Memory True

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # 검증용 데이터셋은 순서를 섞지 않고 그대로 불러온다
    # 이유 : 검증용 데이터셋은 학습에 영향을 주지 않으므로 순서를 섞을 필요가 없음
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    # 모델의 모든 가중치를 gpu로 전송
    # CRNN = CNN + BiLSTM + Attention Decoder
    model = CRNN(vocab_size).to(DEVICE)
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Single GPU (DataParallel disabled for stability on small dataset)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     print(f"[MODEL] DataParallel on {torch.cuda.device_count()} GPUs")
    print(f"[MODEL] Single GPU mode (DataParallel disabled)")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, idx2char, epoch)
        
        scheduler.step()
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'vocab_size': vocab_size,
            }, BEST_MODEL_PATH)
            print(f"[SAVED] Best model -> {BEST_MODEL_PATH} (val_loss={val_loss:.4f})")
        
        # Periodic save
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
    
    print(f"\n[DONE] Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()