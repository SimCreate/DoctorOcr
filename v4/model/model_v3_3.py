#!/usr/bin/env python3
"""
Doctor Handwriting OCR - v3_3 (Hybrid: resnet18 + Attention + CTC)
====================================================================
v3_2(resnet18 backbone + Attention decoder)에 CTC 헤드를 병렬로 추가한 하이브리드.
  - BiLSTM 출력 → [Attention 디코더] + [CTC 헤드] 두 갈래
  - 손실:  L = λ·L_ctc + (1-λ)·L_attn   (기본 λ=0.5)
  - v3_2 강점(고빈도 attention) + v2_2 강점(중·저빈도·CER CTC) 결합 실험

레거시 반영:
  - v2_2의 CTCHead/CTCLoss 구현 재사용 (blank=0, greedy decode)
  - 클린 val 1,116장 구조 분리 (v3의 리키지 교훈) — 스크립트에서 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, ResNet18_Weights

# ============================================================
# CONFIG
# ============================================================
MAX_LABEL_LENGTH = 64
IMAGE_HEIGHT = 128            # v3_2: 비율유지 패딩 (256x128)
IMAGE_WIDTH = 256

# attention 용 특수 토큰 (v3_2 그대로)
SPECIAL_TOKENS = {
    '<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3,
}

# CTC 사용 별도 vocab (blank=0 표준)
CTC_BLANK = 0


# ============================================================
# Encoder: resnet18 (v3_2 그대로)
# ============================================================
class ResNetEncoder(nn.Module):
    """resnet18(ImageNet pretrained) stage1~3 특징. [B,256,H/16,W/16]"""
    def __init__(self, pretrained=True):
        super().__init__()
        assert pretrained, "v3_2는 ImageNet pretrained 전제"
        rn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # stage1(conv1+bn1+relu+maxpool) ~ stage3(layer3) 사용
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.out_channels = 256   # layer3 출력 채널

    def forward(self, x):
        if x.shape[-2] < 32 or x.shape[-1] < 32:
            raise ValueError(f"입력이 너무 작음: {x.shape} (resnet stem 마이너스 와도됨)")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ============================================================
# BiLSTM (v3_2 그대로)
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
        # x: [B, C, H, W] → 시간축을 W(폭)으로, 각 위치 특징 = C*H 벡터
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x)
        return out  # [B, W, hidden*2]


# ============================================================
# CTC Head (v2_2 재사용, blank=0)
# ============================================================
class CTCHead(nn.Module):
    """BiLSTM 출력 → [B,T,ctc_vocab]. ctc_vocab = ctc_char2idx 크기"""
    def __init__(self, encoder_out_dim, ctc_vocab_size, dropout=0.3):
        super().__init__()
        self.ctc_vocab_size = ctc_vocab_size
        self.fc = nn.Linear(encoder_out_dim, ctc_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out):
        return self.fc(self.dropout(encoder_out))   # [B,T,ctc_vocab]

    def get_loss(self, logits, targets, target_lengths):
        """CTCLoss. targets: concat된 정수 시퀀스 [sum(len)]"""
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)          # [B,T,V]
        log_probs = log_probs.permute(1, 0, 2)             # [T,B,V]
        input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
        return F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=CTC_BLANK, reduction='mean', zero_infinity=True
        )

    def decode(self, logits):
        """Greedy CTC decode: 중복·blank 제거 → 토큰 리스트"""
        preds = F.log_softmax(logits, dim=-1).argmax(-1)   # [B,T]
        decoded = []
        for b in range(preds.size(0)):
            prev, result = -1, []
            for tok in preds[b].tolist():
                if tok != prev and tok != CTC_BLANK:
                    result.append(tok)
                prev = tok
            decoded.append(result)
        return decoded


# ============================================================
# Attention Decoder (v3_2 그대로 — 양방향 우선은 LSTM, 출력 vocab은 attn)
# ============================================================
class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim, vocab_size, hidden_size=384, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # encoder_proj: 강제 그래프 분리 없이 간단히 Linear
        self.encoder_proj = nn.Linear(encoder_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True, dropout=dropout)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'fc_out' in name:
                nn.init.xavier_uniform_(p if p.dim() >= 2 else p.unsqueeze(0))

    def forward(self, encoder_out, targets=None, teacher_forcing_ratio=0.5):
        # encoder_out: [B, T, enc_dim] → 프로젝션 (그래프는 분리 안 함)
        proj = self.encoder_proj(encoder_out)          # [B,T,hidden]
        B, T, H = proj.shape

        # mask: valid position만 (pad가 없는 이미지 특성상 전부 valid)
        attn_mask = None

        h0 = torch.zeros(2 * 2, B, self.hidden_size, device=encoder_out.device)
        c0 = torch.zeros(2 * 2, B, self.hidden_size, device=encoder_out.device)

        if self.training:
            # teacher forcing: targets [B, L]
            batch_max_len = targets.size(1)
            inputs = targets[:, :-1]                    # [B, L-1]
            emb = self.embedding(inputs)                # [B, L-1, hidden]
            # 어텐션 질의로 디코더 입력 사용 (전체 step 병렬)
            q = emb
            attn_out, _ = self.attention(q, proj, proj, attn_mask=attn_mask)
            lstm_in = torch.cat([emb, attn_out], dim=-1)
            lstm_out, _ = self.lstm(lstm_in)
            logits = self.fc_out(lstm_out)              # [B, L-1, vocab]
            return logits
        else:
            # inference: 캐시 없는 단순 순차 (beam search용 충분)
            # eval은 beam_search() 사용
            raise NotImplementedError("predict()·beam_search() 사용")

    def beam_search(self, encoder_out, beam_width=5, max_len=None, return_beams=False):
        """teacher forcing 없이 자가회귀 beam search (v3_2 검증 로직, 배치 1 가정)"""
        self.eval()
        if encoder_out.dim() == 3 and encoder_out.size(0) > 1:
            encoder_out = encoder_out[0:1]

        B, seq_len, encoder_dim = encoder_out.shape
        encoder_proj = self.encoder_proj(encoder_out)
        max_len = max_len or 32

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
                out = self.fc_out(self.dropout(lstm_out))
                log_probs = torch.log_softmax(out[0, -1], dim=-1)

                topk_log_probs, topk_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    new_token = topk_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score + topk_log_probs[i].item()
                    new_beam.append((new_seq, new_score, new_hidden))

            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

        if return_beams:
            seen, cands = set(), []
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

        return [seq[0].tolist() for seq, _score, _h in beam]


# ============================================================
# CRNN v3_3 — 하이브리드 (attention + CTC)
# ============================================================
class CRNN(nn.Module):
    """
    vocab_size  : attention vocab (SOS/EOS/PAD/UNK + chars)
    ctc_vocab_size: ctc vocab (blank + chars) — 별도 인덱스
    """
    def __init__(self, vocab_size, ctc_vocab_size, hidden_size=384, dropout=0.3,
                 use_gradient_checkpointing=False, pretrained=True, ctc_weight=0.5,
                 ce_weight=3.0):
        super().__init__()
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.encoder = ResNetEncoder(pretrained=pretrained)
        encoder_out_dim = self.encoder.out_channels * 8   # 256*8(H/16) = 2048
        self.rnn = BiLSTM(encoder_out_dim, hidden_size=hidden_size, num_layers=3, dropout=dropout)
        rnn_out_dim = self.rnn.out_features               # 768
        # 이중 헤드
        self.decoder = AttentionDecoder(rnn_out_dim, vocab_size, hidden_size=hidden_size, dropout=dropout)
        self.ctc_head = CTCHead(rnn_out_dim, ctc_vocab_size, dropout=dropout)

    def forward(self, x, attn_targets=None, ctc_targets=None, ctc_target_lengths=None, teacher_forcing_ratio=0.5):
        feat = self.encoder(x)
        feat = self.rnn(feat)                       # [B,T,768]
        attn_logits = self.decoder(feat, attn_targets, teacher_forcing_ratio) if self.training else None
        ctc_logits = self.ctc_head(feat)            # [B,T,ctc_vocab]
        return attn_logits, ctc_logits

    def hybrid_loss(self, attn_logits, attn_targets, ctc_logits, ctc_targets, ctc_target_lengths):
        """
        L = λ·L_ctc + (1-λ)·β·L_ce   (v3_3c)
        β = ce_weight — CE는 만점(0)으로 수렴하기 쉬워 그래디언트가 죽음 → 독립 스케일로 보존
        """
        λ = self.ctc_weight
        β = self.ce_weight
        ce = F.cross_entropy(
            attn_logits.reshape(-1, attn_logits.size(-1)),
            attn_targets[:, 1:].reshape(-1),
            ignore_index=SPECIAL_TOKENS['<PAD>'],
            label_smoothing=0.1,          # v3_3c: CE=0 만점 붕괴 방지
        )
        ctc_l = self.ctc_head.get_loss(ctc_logits, ctc_targets, ctc_target_lengths)
        loss = λ * ctc_l + (1 - λ) * β * ce
        return loss, ctc_l, ce

    def predict(self, x, beam_width=5, return_beams=False):
        self.eval()
        with torch.no_grad():
            feat = self.encoder(x)
            feat = self.rnn(feat)
            return self.decoder.beam_search(feat, beam_width=beam_width, return_beams=return_beams)

    def predict_ctc(self, x):
        self.eval()
        with torch.no_grad():
            feat = self.encoder(x)
            feat = self.rnn(feat)
            logits = self.ctc_head(feat)
            return self.ctc_head.decode(logits)


# ============================================================
# Tokenizer — attention용 (v3_2 그대로) + CTC용 (blank=0)
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


def build_ctc_char_dict(labels):
    """CTC 별도 사전: blank=0, 이후 문자들 (1~)"""
    chars = set()
    for label in labels:
        for ch in str(label):
            chars.add(ch)
    chars = sorted(list(chars))
    char2idx = {'<BLANK>': CTC_BLANK}   # blank token → idx 0
    idx = 1
    for ch in chars:
        char2idx[ch] = idx
        idx += 1
    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char


def encode_label(label, char2idx, max_len=MAX_LABEL_LENGTH):
    """attention용: SOS + chars + EOS + PAD(padded, fixed length)"""
    tokens = [char2idx['<SOS>']]
    for ch in str(label):
        tokens.append(char2idx.get(ch, char2idx['<UNK>']))
    tokens.append(char2idx['<EOS>'])
    if len(tokens) < max_len + 1:
        tokens += [char2idx['<PAD>']] * (max_len + 1 - len(tokens))
    return tokens[:max_len + 1]


def encode_ctc_label(label, char2idx, max_len=MAX_LABEL_LENGTH):
    """CTC용: blank 없이 실제 문자만, variable length"""
    tokens = []
    for ch in str(label):
        tokens.append(char2idx.get(ch, char2idx['<BLANK>']))  # UNK 없으면 blank로
    return tokens[:max_len]


def decode_sequence(seq, idx2char):
    chars = []
    for idx in seq:
        if idx in idx2char and idx2char[idx] not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '<BLANK>']:
            chars.append(idx2char[idx])
    return ''.join(chars)
