#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_3 평가 — Hybrid(resnet18+Attention+CTC) 모델, 클린 val 1,116장
=====================================================================
best checkpoint는 CTC acc 기준 저장 — 주 평가는 CTC greedy.
(v3_3 학습 중 attention 디코더는 0%라 CTC가 실질 모델)

v3_2(attention)/v3 CTC와 동일 집계 파이프라인으로 공정 비교:
  group_predictions + summarize + oracle + acceptance

실행:
  CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/eval_v3_3.py [--attn]
"""
import os, sys, pickle, argparse
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pandas as pd

V3_3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_3 / "model"))
sys.path.insert(0, str(V3_3 / "evaluate"))

from model_v3_3 import CRNN, decode_sequence
from preprocess_v3_3 import load_resize_pad, preprocess_tensor
from metrics import cer, beam_oracle, has_repetition
from aggregate import group_predictions, summarize, oracle_group_predictions
from acceptance import acceptance_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=str(V3_3 / "working/checkpoints/best_model_v3_3.pth"))
    p.add_argument('--val-csv', type=str, default=str(V3_3 / "data/clean_split/val.csv"))
    p.add_argument('--img-dir', type=str, default="/home/dev/doctor_ocr_v2/dataset/img/img")
    p.add_argument('--char-dict', type=str, default=str(V3_3 / "working/char_dict_v3_3.pkl"))
    p.add_argument('--out', type=str, default=str(V3_3 / "evaluate/result_v3_3_clean.csv"))
    p.add_argument('--attn', action='store_true', help='attention beam도 측정')
    p.add_argument('--beam', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    with open(args.char_dict, 'rb') as f:
        d = pickle.load(f)
    attn_c2i, attn_i2c = d['attn_char2idx'], d['attn_idx2char']
    ctc_c2i, ctc_i2c = d['ctc_char2idx'], d['ctc_idx2char']

    model = CRNN(vocab_size=len(attn_c2i), ctc_vocab_size=len(ctc_c2i),
                 hidden_size=384, dropout=0.3, use_gradient_checkpointing=True,
                 pretrained=True, ctc_weight=0.5).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[MODEL] ckpt epoch={ckpt.get('epoch','?')}, val_ctc_acc={ckpt.get('val_ctc_acc','?')}")

    df = pd.read_csv(args.val_csv)
    print(f"[DATA] val rows: {len(df)}")

    rows = []
    with torch.no_grad():
        for i in range(0, len(df), args.batch):
            batch_df = df.iloc[i:i+args.batch]
            imgs = []
            for _, r in batch_df.iterrows():
                img = load_resize_pad(Path(args.img_dir) / r['filename'], 128, 256)
                img = torch.from_numpy(preprocess_tensor(img)).float()
                imgs.append(img)
            imgs = torch.stack(imgs).to(device)
            feat = model.encoder(imgs)
            feat = model.rnn(feat)
            ctc_logits = model.ctc_head(feat)
            dec = model.ctc_head.decode(ctc_logits)

            for j, (_, r) in enumerate(batch_df.iterrows()):
                gt = str(r['label'])
                pred_str = decode_sequence(dec[j], ctc_i2c)
                rows.append({'true': gt, 'pred': pred_str, 'label': gt,
                             'match': gt == pred_str, 'pred_decoder': 'ctc',
                             'candidates': pred_str,
                             'filename': str(r['filename'])})

    df = pd.DataFrame(rows)
    df['cer'] = df.apply(lambda r: cer(r['true'], r['pred']), axis=1)

    # ---- 집계용 label_counts (attention 블록에서도 사용하므로 먼저 계산) ----
    train_csv = str(V3_3 / "data/exp2_clean/combined_labels.csv")
    full_df = pd.read_csv(train_csv)
    label_counts = full_df['label'].value_counts().to_dict()

    # ---- attention beam (옵션, 느림) ----
    if args.attn:
        print(f"[ATTN] attention beam({args.beam}) 평가 중... (val {len(df)}장)")
        attn_rows = []
        with torch.no_grad():
            for idx, r in df.iterrows():
                img = load_resize_pad(Path(args.img_dir) / r['filename'], 128, 256)
                img = torch.from_numpy(preprocess_tensor(img)).float().unsqueeze(0).to(device)
                pred_candidates = model.predict(img, beam_width=args.beam)  # [[tok,...], ...]
                pred_tokens = pred_candidates[0] if pred_candidates else []
                pred_str = decode_sequence(pred_tokens, attn_i2c)
                gt = str(r['true'])
                attn_rows.append({'true': gt, 'pred': pred_str, 'label': gt,
                                  'match': gt == pred_str, 'pred_decoder': 'attn',
                                  'candidates': pred_str,
                                  'cer': cer(gt, pred_str)})
        attn_df = pd.DataFrame(attn_rows)
        # attention 전용 빈도그룹 집계
        attn_pred = attn_df.to_dict('records')
        for p in attn_pred:
            p['group'] = 'low'
            c = label_counts.get(p['label'])
            if c is not None:
                if c >= 10: p['group'] = 'high'
                elif c >= 2: p['group'] = 'mid'
        acc_a = group_predictions(attn_pred, label_counts)
        summ_a = summarize(acc_a)
        print("\n=== [RESULT v3_3 (hybrid, ATTENTION beam = %d)] ===" % args.beam)
        for g in ('high', 'mid', 'low'):
            s = summ_a[g]
            a = f"{s['acc']:.1%}" if s['acc'] is not None else 'N/A'
            c = f"{s['avg_cer']:.1%}" if s['avg_cer'] is not None else 'N/A'
            print(f"  {g:5s} total={s['total']:4d}  exact_acc={a:>7s}  avg_cer={c}")
        n_rep_a = sum(1 for p in attn_pred if p.get('repetition', False))
        print(f"\n=== [REPETITION] {n_rep_a}/{len(attn_pred)} = {n_rep_a/len(attn_pred):.1%}")
        print(f"=== [전체] attention exact={attn_df['match'].mean()*100:.1f}%  CER={attn_df['cer'].mean()*100:.2f}% ===")
        attn_df['group'] = [p['group'] for p in attn_pred]
        attn_out = str(Path(args.out).with_name(Path(args.out).stem + '_attn.csv'))
        attn_df.to_csv(attn_out, index=False, encoding='utf-8')
        print(f"[SAVED] {attn_out}")

    # ---- 집계 (v3 공통) ----
    predictions = df.to_dict('records')
    for p in predictions:
        p['group'] = 'low'
        c = label_counts.get(p['label'])
        if c is not None:
            if c >= 10: p['group'] = 'high'
            elif c >= 2: p['group'] = 'mid'

    acc = group_predictions(predictions, label_counts)
    summ = summarize(acc)
    print("\n=== [RESULT v3_3 (hybrid, CTC greedy)] ===")
    for g in ('high', 'mid', 'low'):
        s = summ[g]
        total, acc_v = s['total'], s['acc']
        cer_v = s['avg_cer']
        a = f"{acc_v:.1%}" if acc_v is not None else 'N/A'
        c = f"{cer_v:.1%}" if cer_v is not None else 'N/A'
        print(f"  {g:5s} total={total:4d}  exact_acc={a:>7s}  avg_cer={c}")

    n_rep = sum(1 for p in predictions if p.get('repetition', False))
    print(f"\n=== [REPETITION] {n_rep}/{len(predictions)} = {n_rep/len(predictions):.1%}")
    print(f"=== [전체] exact={df['match'].mean()*100:.1f}%  CER={df['cer'].mean()*100:.2f}% ===")

    df['group'] = [p['group'] for p in predictions]
    df.to_csv(args.out, index=False, encoding='utf-8')
    print(f"\n[SAVED] {args.out} ({len(df)} rows)")


if __name__ == '__main__':
    main()
