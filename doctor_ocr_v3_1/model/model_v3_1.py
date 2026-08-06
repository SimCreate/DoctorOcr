#!/usr/bin/env python3
"""
Doctor Handwriting OCR - Model Definition (v2_1)
개선된 CRNN 모델: BiLSTM 3층 + 강화된 Attention + Dropout
"""

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

# ============================================================
# CONFIG
# ============================================================
MAX_LABEL_LENGTH = 64
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}


# ============================================================
# CNN ENCODER (개선: 더 깊은 구조 + SE 블록 + Gradient Checkpointing 지원)
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
# BiLSTM (개선: 3층 + Dropout 증가)
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=384, num_layers=3, dropout=0.3):
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
# Attention Decoder (개선: Multi-head Attention + Beam Search 지원)
# ============================================================
class AttentionDecoder(nn.Module):
    def __init__(self, encoder_out_dim, vocab_size, hidden_size=384, max_len=MAX_LABEL_LENGTH, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True, dropout=dropout)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.encoder_proj = nn.Linear(encoder_out_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out, targets=None, teacher_forcing_ratio=0.5):
        B, seq_len, encoder_dim = encoder_out.shape
        encoder_proj = self.encoder_proj(encoder_out)

        decoder_input = torch.full((B, 1), SPECIAL_TOKENS['<SOS>'], dtype=torch.long, device=encoder_out.device)
        outputs = []
        hidden = None

        for t in range(self.max_len):
            embedded = self.dropout(self.embedding(decoder_input))
            attn_out, _ = self.attention(embedded, encoder_proj, encoder_proj)
            lstm_in = torch.cat([embedded, attn_out], dim=-1)
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            out = self.fc(self.dropout(lstm_out))
            outputs.append(out)

            if targets is not None and teacher_forcing_ratio > 0 and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t:t+1]
            else:
                decoder_input = out.argmax(-1)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def beam_search(self, encoder_out, beam_width=5, max_len=None, return_beams=False):
        """Beam search decoding - 단일 샘플만 처리

        return_beams=False (기본): 최고 점수 1개 시퀀스만 반환 (기존 동작, top-1).
        return_beams=True:  top-k 후보 토큰 시퀀스 리스트 반환 (중복 제거,
                            각 원소는 EOS 이전까지의 token list). beam oracle용.
        """
        if max_len is None:
            max_len = self.max_len

        # 단일 샘플만 처리 (batch=1 가정)
        if encoder_out.dim() == 3 and encoder_out.size(0) > 1:
            encoder_out = encoder_out[0:1]  # 첫 번째 샘플만 사용

        B, seq_len, encoder_dim = encoder_out.shape
        encoder_proj = self.encoder_proj(encoder_out)

        # Initialize beam with SOS token
        sos_tensor = torch.tensor([[SPECIAL_TOKENS['<SOS>']]], device=encoder_out.device)
        beam = [(sos_tensor, 0.0, None)]

        for _ in range(max_len):
            new_beam = []
            for seq, score, hidden in beam:
                if seq[0, -1].item() == SPECIAL_TOKENS['<EOS>']:
                    new_beam.append((seq, score, hidden))
                    continue

                # seq shape: [1, current_len], get last token: seq[:, -1:]
                last_token = seq[:, -1:]  # shape [1, 1]
                embedded = self.dropout(self.embedding(last_token))
                attn_out, _ = self.attention(embedded, encoder_proj, encoder_proj)
                lstm_in = torch.cat([embedded, attn_out], dim=-1)
                lstm_out, new_hidden = self.lstm(lstm_in, hidden)
                out = self.fc(self.dropout(lstm_out))
                log_probs = torch.log_softmax(out[0, -1], dim=-1)

                topk_log_probs, topk_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    new_token = topk_indices[i].unsqueeze(0).unsqueeze(0)  # [1, 1]
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score + topk_log_probs[i].item()
                    new_beam.append((new_seq, new_score, new_hidden))

            # Keep top beam_width sequences
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

        if return_beams:
            # top-k 후보 토큰 시퀀스 리스트 (중복 제거, EOS 전까지만)
            seen = set()
            cands = []
            for seq, _score, _h in beam:
                toks = seq[0].tolist()
                # EOS 이후 제거
                cut = []
                for t in toks:
                    if t == SPECIAL_TOKENS['<EOS>']:
                        break
                    cut.append(t)
                key = tuple(cut)
                if key not in seen:
                    seen.add(key)
                    cands.append(cut)
            return cands
        return beam[0][0]


# ============================================================
# CRNN Model
# ============================================================
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=384, dropout=0.3, use_gradient_checkpointing=False):
        super().__init__()
        self.encoder = CNNEncoder(use_checkpointing=use_gradient_checkpointing)
        encoder_out_dim = self.encoder.out_channels * 3  # 1536

        self.rnn = BiLSTM(encoder_out_dim, hidden_size=hidden_size, num_layers=3, dropout=dropout)
        rnn_out_dim = self.rnn.out_features  # 768 (hidden_size * 2)

        # AttentionDecoder expects the BiLSTM output dimension (rnn_out_dim)
        self.decoder = AttentionDecoder(rnn_out_dim, vocab_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        feat = self.encoder(x)
        feat = self.rnn(feat)
        out = self.decoder(feat, targets, teacher_forcing_ratio)
        return out

    def predict(self, x, beam_width=5, return_beams=False):
        """Beam search 기반 예측

        return_beams=False (기본): top-1 token 시퀀스 반환 (기존 동작).
        return_beams=True:        top-k 후보 token 시퀀스 리스트 반환 (oracle용).
        """
        self.eval()
        with torch.no_grad():
            feat = self.encoder(x)
            feat = self.rnn(feat)
            return self.decoder.beam_search(feat, beam_width=beam_width, return_beams=return_beams)


# ============================================================
# Tokenizer
# ============================================================
def build_char_dict(labels):
    chars = set()
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
    tokens = [char2idx['<SOS>']]
    for ch in str(label):
        tokens.append(char2idx.get(ch, char2idx['<UNK>']))
    tokens.append(char2idx['<EOS>'])

    if len(tokens) < max_len:
        tokens += [char2idx['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def decode_sequence(seq, idx2char):
    chars = []
    for idx in seq:
        if idx == SPECIAL_TOKENS['<EOS>'] or idx == SPECIAL_TOKENS['<PAD>']:
            break
        if idx in idx2char and idx2char[idx] not in ['<SOS>', '<PAD>', '<EOS>', '<UNK>']:
            chars.append(idx2char[idx])
    return ''.join(chars)