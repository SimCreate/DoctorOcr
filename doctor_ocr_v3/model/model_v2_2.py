#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Model Definition (v2_2 → v3 복제본)
CTC 기반 CRNN: BiLSTM 3층 + CTC Head (병렬 연산으로 GPU 100% 활용)

v2_2 라이브 원본: /home/dev/doctor_ocr_v2_2/model/model_v2_2.py
  - v2_2 원본은 수정 금지 (메모리 규칙). 본 파일은 기능적으로 원본과 동일한 복제본.
  - v2_2 원본 수정 시 동기화 필요 (diff로 반영).
"""

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

# ============================================================
# CONFIG
# ============================================================
MAX_LABEL_LENGTH = 64
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

# CTC용: blank token은 0으로 고정 (PyTorch CTC 표준)
CTC_BLANK = 0

SPECIAL_TOKENS = {
    '<PAD>': 1,   # pad는 1로
    '<SOS>': 2,
    '<EOS>': 3,
    '<UNK>': 4,
}


# ============================================================
# CNN ENCODER (SE 블록 + Gradient Checkpointing 지원)
# ============================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 블록"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class CNNEncoder(nn.Module):
    def __init__(self, img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            SEBlock(64),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            SEBlock(128),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            SEBlock(256),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            SEBlock(512),

            # Block 5
            nn.Conv2d(512, 512, 2, padding=0), nn.BatchNorm2d(512), nn.ReLU(True),
            SEBlock(512),
        )
        self.out_channels = 512

    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(lambda inp: self.cnn(inp), x, use_reentrant=False)
        return self.cnn(x)


# ============================================================
# BiLSTM (3층 + Dropout)
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.out_features = hidden_size * 2

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x)
        return out


# ============================================================
# CTC Head (병렬 연산 - Attention Decoder 대체)
# ============================================================
class CTCHead(nn.Module):
    def __init__(self, encoder_out_dim, vocab_size, hidden_size=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size  # blank token 포함 크기
        self.hidden_size = hidden_size

        # Linear projection for CTC
        self.fc = nn.Linear(encoder_out_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out):
        """
        Args:
            encoder_out: [B, T, encoder_out_dim] - BiLSTM output
        Returns:
            logits: [B, T, vocab_size] - log_softmax 적용 전
        """
        # Dropout 적용
        x = self.dropout(encoder_out)
        # Linear projection to vocab (blank 포함)
        logits = self.fc(x)
        return logits  # [B, T, vocab_size]

    def get_loss(self, logits, targets, target_lengths):
        """
        CTC Loss 계산
        Args:
            logits: [B, T, vocab_size] - raw logits
            targets: [B, max_target_len] - padded target sequences
            target_lengths: [B] - 실제 target 길이
        """
        B, T, V = logits.shape
        
        # log_softmax 적용 (시간 차원 T에 대해)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        
        # CTC는 [T, B, V] 형태 필요
        log_probs = log_probs.permute(1, 0, 2)  # [T, B, V]
        
        # input lengths: 모든 샘플이 같은 T
        input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
        
        # CTC loss
        loss = F.ctc_loss(
            log_probs, 
            targets, 
            input_lengths, 
            target_lengths,
            blank=CTC_BLANK,
            reduction='mean',
            zero_infinity=True
        )
        return loss

    def decode(self, logits):
        """
        Greedy decoding for inference
        Args:
            logits: [B, T, vocab_size]
        Returns:
            List of decoded strings
        """
        log_probs = F.log_softmax(logits, dim=-1)
        preds = log_probs.argmax(-1)  # [B, T]
        
        decoded = []
        for b in range(preds.size(0)):
            seq = preds[b].tolist()
            # CTC greedy decode: 연속된 동일 토큰 제거, blank 제거
            prev = -1
            result = []
            for token in seq:
                if token != prev and token != CTC_BLANK:
                    result.append(token)
                prev = token
            decoded.append(result)
        return decoded


# ============================================================
# CRNN Model (CTC 버전)
# ============================================================
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, dropout=0.3, use_gradient_checkpointing=False):
        super().__init__()
        # vocab_size에는 blank token(0) + 실제 문자들 포함
        self.encoder = CNNEncoder(use_checkpointing=use_gradient_checkpointing)
        encoder_out_dim = self.encoder.out_channels * 3  # 512 * 3 = 1536

        self.rnn = BiLSTM(encoder_out_dim, hidden_size=hidden_size, num_layers=3, dropout=dropout)
        rnn_out_dim = self.rnn.out_features  # 512

        self.ctc_head = CTCHead(rnn_out_dim, vocab_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x):
        """
        Training forward - returns logits for CTC loss
        Args:
            x: [B, 3, H, W]
        Returns:
            logits: [B, T, vocab_size]
        """
        feat = self.encoder(x)      # [B, 512, H', W']
        feat = self.rnn(feat)       # [B, T, 512]
        logits = self.ctc_head(feat)  # [B, T, vocab_size]
        return logits

    def predict(self, x):
        """Inference - greedy CTC decode"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            decoded = self.ctc_head.decode(logits)
            return decoded


# ============================================================
# Tokenizer (CTC용: blank=0, 문자=1~)
# ============================================================
def build_char_dict(labels):
    chars = set()
    for label in labels:
        for ch in str(label):
            chars.add(ch)
    chars = sorted(list(chars))

    # CTC: blank token은 0으로 예약
    char2idx = {'<BLANK>': CTC_BLANK}
    idx = len(char2idx)
    
    # 특수 토큰들 (blank 제외)
    for tok, val in SPECIAL_TOKENS.items():
        char2idx[tok] = idx
        idx += 1
    
    # 일반 문자들
    for ch in chars:
        char2idx[ch] = idx
        idx += 1

    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char


def encode_label(label, char2idx, max_len=MAX_LABEL_LENGTH):
    """
    CTC 타겟 인코딩: blank token 제외, 실제 문자만 시퀀스로
    """
    tokens = []
    for ch in str(label):
        tokens.append(char2idx.get(ch, char2idx['<UNK>']))
    # CTC 타겟은 패딩하지 않음 (target_lengths로 길이 전달)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    return tokens


def decode_sequence(seq, idx2char):
    """CTC greedy decode 결과 문자열로 변환"""
    chars = []
    for idx in seq:
        if idx in idx2char and idx2char[idx] not in ['<BLANK>', '<PAD>', '<SOS>', '<EOS>', '<UNK>']:
            chars.append(idx2char[idx])
    return ''.join(chars)