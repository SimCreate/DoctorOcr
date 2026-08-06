#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 / v2_1 평가 — 클린 val 1,116장 추론 결과 CSV 생성
=========================================================
- v2 (Attention greedy), v2_1 (Attention beam=5) — 각자의 원본 모델 정의 사용
- 데이터: 클린 val.csv (v3_1/data/clean_split/val.csv, 1,116장)
- 이미지: /home/dev/doctor_ocr_v2/dataset/img/img (원본 5,578장)
- 출력: evaluate/result_v2_clean.csv, result_v2_1_clean.csv (true/pred/match/label/path/cer)
"""
import os, sys, pickle, argparse
from pathlib import Path
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '1'   # GPU1 (Max-Q) 타겟
import torch
import pandas as pd
import cv2
import numpy as np

V31 = Path('/home/dev/DoctorOcr/doctor_ocr_v3_1')
V2  = Path('/home/dev/doctor_ocr_v2')
V21 = Path('/home/dev/doctor_ocr_v2_1')
DEST = V31 / 'evaluate'
VAL_CSV = V31 / 'data/clean_split/val.csv'
IMG_DIR = Path('/home/dev/doctor_ocr_v2/dataset/img/img')
V2_CKPT = V2 / 'working/checkpoints/best_model.pth'
V21_CKPT = V21 / 'working/checkpoints/best_model.pth'
V2_CHAR = V2 / 'working/char_dict.pkl'
V21_CHAR = V21 / 'working/char_dict.pkl'

IMG_H, IMG_W = 64, 256


def build_v2_model(vocab_size):
    # v2 원본 모델 정의 (doctor_ocr_v2/local_infer.py와 동일)
    import torch.nn as nn
    class CNNEncoder(nn.Module):
        def __init__(self):
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
        def __init__(self, input_size, hidden_size=256):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, 2,
                                bidirectional=True, batch_first=True, dropout=0.2)
            self.out_features = hidden_size * 2
        def forward(self, x):
            B, C, H, W = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
            out, _ = self.lstm(x)
            return out

    class AttentionDecoder(nn.Module):
        def __init__(self, encoder_out_dim, vocab_size, hidden_size=256, max_len=64):
            super().__init__()
            self.vocab_size = vocab_size
            self.max_len = max_len
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden_size, vocab_size)
            self.encoder_proj = nn.Linear(encoder_out_dim, hidden_size)
        def forward(self, encoder_out, targets=None, teacher_forcing_ratio=0.0):
            B, seq_len, encoder_dim = encoder_out.shape
            encoder_proj = self.encoder_proj(encoder_out)
            decoder_input = torch.full((B, 1), 1, dtype=torch.long, device=encoder_out.device)  # <SOS>
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
            return torch.cat(outputs, dim=1)

    class CRNN(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.encoder = CNNEncoder()
            encoder_out_dim = self.encoder.out_channels * 3
            self.rnn = BiLSTM(encoder_out_dim, hidden_size=256)
            self.decoder = AttentionDecoder(self.rnn.out_features, vocab_size)
        def forward(self, x, targets=None, teacher_forcing_ratio=0.0):
            feat = self.encoder(x)
            feat = self.rnn(feat)
            return self.decoder(feat, targets, teacher_forcing_ratio)

    return CRNN(vocab_size)


def decode_sequence(seq, idx2char, special):
    chars = []
    for idx in seq:
        if idx == special['<EOS>'] or idx == special['<PAD>']:
            break
        if idx in idx2char and idx2char[idx] not in ['<SOS>', '<PAD>', '<EOS>', '<UNK>']:
            chars.append(idx2char[idx])
    return ''.join(chars)


def apply_cer(true, pred):
    # Levenshtein 기반 CER (metrics.cer와 동일)
    if not true:
        return 1.0 if pred else 0.0
    def lev(a, b):
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b)+1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost))
            prev = cur
        return prev[-1]
    return lev(true, pred) / len(true)


def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def run(version):
    print(f"\n===== {version} 평가 시작 =====")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if version == 'v2':
        ckpt = load_ckpt(V2_CKPT)
        char_data = pickle.load(open(V2_CHAR, 'rb'))
        hidden = 256
        beam = 1
        out_csv = DEST / 'result_v2_clean.csv'
    else:  # v2_1
        ckpt = load_ckpt(V21_CKPT)
        char_data = pickle.load(open(V21_CHAR, 'rb'))
        hidden = ckpt['config'].get('hidden_size', 256)
        beam = 5
        out_csv = DEST / 'result_v2_1_clean.csv'

    char2idx, idx2char = char_data['char2idx'], char_data['idx2char']
    special = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    vocab_size = len(char2idx)

    if version == 'v2':
        model = build_v2_model(vocab_size).to(device)
        beam = 1
    else:  # v2_1 — 자체 원본 모델(model_v2_1.py) import
        sys.path.insert(0, str(V21 / 'model'))
        from model_v2_1 import CRNN as CRNN21, decode_sequence as dec21
        # ⚠️ v2_1 체크포인트 config는 hidden=256으로 잘못 저장됐으나
        #    실제 학습은 hidden=384로 진행됨 (원본의 config 버그).
        #    v3_1에서 수정된 값(384)을 사용해야 체크포인트와 크기가 맞음.
        hidden = 384
        model = CRNN21(
            vocab_size,
            hidden_size=hidden,
            dropout=ckpt['config'].get('dropout', 0.3),
            use_gradient_checkpointing=ckpt['config'].get('use_gradient_checkpointing', True),
        ).to(device)
        beam = 5
    sd = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in sd):
        from collections import OrderedDict
        sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in sd.items())
    model.load_state_dict(sd)
    model.eval()
    print(f"[{version}] vocab={vocab_size}, ckpt epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss','?')}, beam={beam}")

    val_df = pd.read_csv(VAL_CSV)
    img_col = 'filename' if 'filename' in val_df.columns else val_df.columns[0]
    rows = []
    with torch.no_grad():
        for i, (_, row) in enumerate(val_df.iterrows()):
            img_path = IMG_DIR / row[img_col]
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(cv2.resize(img, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)
            gt = str(row['label']) if 'label' in val_df.columns else str(row[1])

            if version == 'v2_1' and beam > 1:
                # v2_1 원본: model.predict(x, beam_width) → top-1 token 시퀀스
                pred_tokens = model.predict(img_t, beam_width=beam)[0].cpu().tolist()
                pred_str = dec21(pred_tokens, idx2char)
            else:
                # v2 greedy
                output = model(img_t, teacher_forcing_ratio=0.0)
                pred_tokens = output.argmax(-1)[0].cpu().tolist()
                pred_str = decode_sequence(pred_tokens, idx2char, special)
            rows.append({'true': gt, 'pred': pred_str, 'match': gt == pred_str,
                         'label': gt, 'path': str(img_path)})
            if (i+1) % 300 == 0:
                print(f"  {i+1}/{len(val_df)}")

    for r in rows:
        r['cer'] = apply_cer(r['true'], r['pred'])
    # 실수 float로 저장
    for r in rows:
        r['cer'] = round(r['cer'], 6)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['true', 'pred', 'match', 'label', 'path', 'cer'])
        w.writeheader()
        w.writerows(rows)

    exact = sum(1 for r in rows if r['match'])
    avg_cer = sum(r['cer'] for r in rows) / len(rows)
    print(f"[{version}] SAVED {out_csv}")
    print(f"[{version}] RESULT exact={exact}/{len(rows)} = {exact/len(rows)*100:.1f}%, avg_cer={avg_cer*100:.1f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', nargs='+', default=['v2', 'v2_1'])
    args = ap.parse_args()
    for v in args.versions:
        run(v)
