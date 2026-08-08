#!/usr/bin/env python3
"""
Doctor Handwriting OCR - v4 (resnet18 pretrained backbone)
===========================================================
v3_1(SEBlock CNN 5블록)의 인코더를 ImageNet pretrained resnet18로 교체한 실험 모델.

동기 (교차검증 2026-08-07):
  - 오류 82%가 인코더 미인식(oracle=False) → 인코더 표현력이 근본 병목
  - 사전학습 backbone이 저수준 특징(획/에지) 전이에 가장 효과적 (ChatGPT+로컬LLM 합의)
  - 기존 SEBlock CNN: pretrained 불가, 특징맵 16열(stride 32)로 해상도 부족
  - resnet18: 11.7M(기존과 비슷), layer3 특징맵 64열(stride 16), ImageNet pretrained
    → GPU1 여유 2.9GB로 미세조정 가능 (ConvNeXt/EfficientNet은 VRAM 초과)

아키텍처:
  resnet18 (layer3까지, stride 16, C=256)
    → BiLSTM 3층(hidden 384)         [기존 유지]
    → AttentionDecoder 8-head + beam [기존 유지]

전처리 (핵심 개선):
  - 기존: 128x128 원본 → 256x64 강제 리사이즈 = 4:1 왜곡 (시각 검증으로 글자 눌림 확인)
  - 변경: 128x128 원본 → 256x128 패딩(비율유지) = 형태 보존
    (이 왜곡이 인코더가 "못 읽는" 주 용의자)

resnet18 특징 추출: ImageNet pretrained 가중치 사용, layer3까지 (stride 16).
  - layer1(1/4), layer2(1/8), layer3(1/16) 출력을 모두 BiLSTM 입력으로 쓰지 않고
    layer3만 사용 (기존 CNNEncoder와 동일한 단일 특징맵 설계 유지 → 코드 변경 최소화)
"""
import torch
import torch.nn as nn
import math
import torchvision

# ============================================================
# CONFIG (v3_1과 동일 — 입력만 256x128로 변경)
# ============================================================
MAX_LABEL_LENGTH = 64
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}


# ============================================================
# RESNET-18 ENCODER (ImageNet pretrained, layer3까지)
# ============================================================
class ResNetEncoder(nn.Module):
    """resnet18의 conv1~layer3까지 사용 (stride 16, 채널 256).

    256x128 입력 → 특징맵 (256, 8, 16) → BiLSTM은 H*C=2048을 시간축으로,
    폭 16열을 순서로 사용.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # conv1 ~ layer3 (stride 16)
        self.conv1 = resnet.conv1      # 7x7 stride=2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride 2
        self.layer1 = resnet.layer1    # 64ch
        self.layer2 = resnet.layer2    # 128ch
        self.layer3 = resnet.layer3    # 256ch
        self.out_channels = 256

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # [B, C, H, W] = [B, 256, 8, 16]
        return x


# ============================================================
# BiLSTM (v3_1 재사용)
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=384, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out_features = hidden_size * 2

    def forward(self, x):
        B, C, H, W = x.shape
        # [B, W, C*H] — 폭 방향(256→16열)을 시간축으로
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x)
        return out


# ============================================================
# AttentionDecoder (v3_1 재사용)
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
        if max_len is None:
            max_len = self.max_len
        if encoder_out.dim() == 3 and encoder_out.size(0) > 1:
            encoder_out = encoder_out[0:1]

        B, seq_len, encoder_dim = encoder_out.shape
        encoder_proj = self.encoder_proj(encoder_out)

        sos_tensor = torch.tensor([[SPECIAL_TOKENS['<SOS>']]], device=encoder_out.device)
        beam = [(sos_tensor, 0.0, None)]

        for _ in range(max_len):
            new_beam = []
            for seq, score, hidden in beam:
                if seq[0, -1].item() == SPECIAL_TOKENS['<EOS>']:
                    new_beam.append((seq, score, hidden))
                    continue
                last_token = seq[:, -1:]
                embedded = self.dropout(self.embedding(last_token))
                attn_out, _ = self.attention(embedded, encoder_proj, encoder_proj)
                lstm_in = torch.cat([embedded, attn_out], dim=-1)
                lstm_out, new_hidden = self.lstm(lstm_in, hidden)
                out = self.fc(self.dropout(lstm_out))
                log_probs = torch.log_softmax(out[0, -1], dim=-1)

                topk_log_probs, topk_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    new_token = topk_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score + topk_log_probs[i].item()
                    # deepcopy 대신 hidden 유지 (clone 필요 시 성능 저하 — 여기선 참조 유지)
                    new_beam.append((new_seq, new_score, new_hidden))

            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

        if return_beams:
            seen = set()
            cands = []
            for seq, _score, _h in beam:
                toks = seq[0].tolist()
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
# CRNN v4 — resnet18 backbone
# ============================================================
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=384, dropout=0.3,
                 use_gradient_checkpointing=False, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        encoder_out_dim = self.encoder.out_channels * 8   # 256 * 8(H) = 2048

        self.rnn = BiLSTM(encoder_out_dim, hidden_size=hidden_size, num_layers=3, dropout=dropout)
        rnn_out_dim = self.rnn.out_features  # 768

        self.decoder = AttentionDecoder(rnn_out_dim, vocab_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        feat = self.encoder(x)
        feat = self.rnn(feat)
        out = self.decoder(feat, targets, teacher_forcing_ratio)
        return out

    def predict(self, x, beam_width=5, return_beams=False):
        self.eval()
        with torch.no_grad():
            feat = self.encoder(x)
            feat = self.rnn(feat)
            return self.decoder.beam_search(feat, beam_width=beam_width, return_beams=return_beams)


# ============================================================
# Tokenizer (v3_1 재사용)
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
