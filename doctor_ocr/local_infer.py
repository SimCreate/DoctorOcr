#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Local Inference Script
Loads checkpoint from local_train.py and runs prediction on samples
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

# Apply CLI overrides BEFORE model creation
if args.device is None:
    args.device = '0'
if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# ============================================================
# CONFIG - HARDCODED PATHS
# ============================================================
DATA_ROOT = Path("/home/dev/doctor_ocr/dataset")
IMG_DIR = DATA_ROOT / "img" / "img"
LABEL_CSV = DATA_ROOT / "doctor_handwriting_labels.csv"

WORK_DIR = Path("/home/dev/doctor_ocr/working")
CHAR_DICT_PATH = WORK_DIR / "char_dict.pkl"
BEST_MODEL_PATH = WORK_DIR / "checkpoints" / "best_model.pth"

# 모델 설정 (학습과 동일해야 함)
MAX_LABEL_LENGTH = 64
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Guard: if CUDA_VISIBLE_DEVICES set to 0, cuda:0 maps to physical GPU 0 (Blackwell)
if torch.cuda.is_available():
    print(f"[CONFIG] DEVICE: {DEVICE} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')} | Mapped GPU: {torch.cuda.get_device_name(0)} (UUID={torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0),'uuid') else 'N/A'})")
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}

print(f"[CONFIG] DEVICE: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
print(f"[CONFIG] MODEL: {BEST_MODEL_PATH}")
print(f"[CONFIG] CHAR_DICT: {CHAR_DICT_PATH}")

# ============================================================
# MODEL DEFINITION (must match local_train.py exactly)
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self, img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.out_channels = 512
    
    def forward(self, x):
        return self.cnn(x)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=0.2
        )
        self.out_features = hidden_size * 2
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x)
        return out


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
        # [FIX] encoder_proj를 __init__에서 생성
        self.encoder_proj = nn.Linear(encoder_out_dim, hidden_size)
    
    def forward(self, encoder_out, targets=None, teacher_forcing_ratio=0.0):
        B, seq_len, encoder_dim = encoder_out.shape
        
        encoder_proj = self.encoder_proj(encoder_out)
        
        decoder_input = torch.full((B, 1), SPECIAL_TOKENS['<SOS>'], dtype=torch.long, device=encoder_out.device)
        
        outputs = []
        hidden = None
        
        for t in range(self.max_len):
            embedded = self.embedding(decoder_input)
            attn_out, _ = self.attention(embedded, encoder_proj, encoder_proj)
            lstm_in = torch.cat([embedded, attn_out], dim=-1)
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            out = self.fc(lstm_out)
            outputs.append(out)
            decoder_input = out.argmax(-1)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs


class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        encoder_out_dim = self.encoder.out_channels * 3
        self.rnn = BiLSTM(encoder_out_dim, hidden_size=256)
        rnn_out_dim = self.rnn.out_features
        self.decoder = AttentionDecoder(rnn_out_dim, vocab_size)
    
    def forward(self, x, targets=None, teacher_forcing_ratio=0.0):
        feat = self.encoder(x)
        feat = self.rnn(feat)
        out = self.decoder(feat, targets, teacher_forcing_ratio)
        return out


# ============================================================
# UTILS
# ============================================================
def decode_sequence(seq, idx2char):
    chars = []
    for idx in seq:
        if idx == SPECIAL_TOKENS['<EOS>'] or idx == SPECIAL_TOKENS['<PAD>']:
            break
        if idx in idx2char and idx2char[idx] not in ['<SOS>', '<PAD>', '<EOS>', '<UNK>']:
            chars.append(idx2char[idx])
    return ''.join(chars)


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
    model = CRNN(vocab_size).to(DEVICE)
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[LOAD] Model from epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f}, val_acc={checkpoint.get('val_acc', '?'):.4f}")
    
    # Single CPU model (no DataParallel)
    return model, char2idx, idx2char


def predict_single(model, img_tensor, idx2char):
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        output = model(img_tensor, teacher_forcing_ratio=0.0)
        pred = output.argmax(-1)[0].cpu().tolist()
        return decode_sequence(pred, idx2char)


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
    
    print(f"\n=== INFERENCE ON {len(samples)} RANDOM SAMPLES ===\n")
    
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
        
        status = "✓" if is_correct else "✗"
        print(f"  {row['filename']} | label='{true_label}' | pred='{pred_label}' | {status}")
    
    acc = correct / len(samples)
    print(f"\n=== ACCURACY: {correct}/{len(samples)} = {acc:.2%} ===")


if __name__ == "__main__":
    main()