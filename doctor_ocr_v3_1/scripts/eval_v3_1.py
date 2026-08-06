#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_1 평가 — Attention(beam search) 모델을 클린 val 1,116장으로 평가
=====================================================================
v3 exp2_clean(run_eval.py)과 동일한 지표를 사용해 공정 비교:
  exact / CER / 빈도그룹(고·중·저) / 수용기준 판정

실행 (v3_1 venv):
  CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/eval_v3_1.py
"""
import os, sys, pickle, argparse
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

V3_1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_1 / "model"))
sys.path.insert(0, str(V3_1 / "scripts"))

from model_v2_1 import CRNN, decode_sequence, MAX_LABEL_LENGTH

# ---- v3_1 자체 지표 모듈 (기본 지표는 v3에서 복제, v3_1 전용 확장 지표 포함) ----
sys.path.insert(0, str(V3_1 / "evaluate"))
from metrics import cer, beam_oracle, has_repetition     # noqa: E402
from aggregate import group_predictions, summarize, oracle_group_predictions  # noqa: E402
from acceptance import acceptance_report, overall_cer, acceptance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=str(V3_1 / "working/checkpoints/best_model.pth"))
    p.add_argument('--val-csv', type=str, default=str(V3_1 / "data/clean_split/val.csv"))
    p.add_argument('--img-dir', type=str, default="/home/dev/doctor_ocr_v2/dataset/img/img")
    p.add_argument('--char-dict', type=str, default=str(V3_1 / "working/char_dict.pkl"))
    p.add_argument('--out', type=str, default=str(V3_1 / "evaluate/result_v3_1_clean.csv"))
    p.add_argument('--beam', type=int, default=5, help='beam width')
    p.add_argument('--batch', type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    print(f"[CKPT] epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f} config={config}")

    with open(args.char_dict, 'rb') as f:
        cd = pickle.load(f)
    char2idx, idx2char = cd['char2idx'], cd['idx2char']
    vocab_size = len(char2idx)
    print(f"[VOCAB] size={vocab_size}")

    model = CRNN(
        vocab_size,
        hidden_size=config.get('hidden_size', 384),
        dropout=config.get('dropout', 0.3),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', True),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={n_params:,}  ({n_params/1e6:.1f}M)")

    # val.csv 직접 읽기 (GT + path), 이미지는 동일 전처리로 로드
    import pandas as pd, cv2, numpy as np
    from PIL import Image
    val_df = pd.read_csv(args.val_csv)
    # 컬럼명 검증
    print(f"[VAL CSV] columns: {list(val_df.columns)}")
    img_col = 'filename' if 'filename' in val_df.columns else val_df.columns[0]

    def load_img(path: Path):
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((dataset_h, dataset_w, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (dataset_w, dataset_h))
        return img

    model_h = config.get('image_height', 64)
    dataset_h, dataset_w = model_h, config.get('image_width', 256)
    print(f"[DATA] val rows: {len(val_df)}  (img {dataset_w}x{dataset_h})")

    rows = []
    model = model.to(device)
    with torch.no_grad():
        for i, (_, row) in enumerate(val_df.iterrows()):
            img_path = Path(args.img_dir) / row[img_col]
            img = load_img(img_path)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)   # [1,3,H,W]
            gt = str(row['label']) if 'label' in val_df.columns else str(row[1])

            # top-1 + top-k 후보 (oracle 분석용)
            beam_tokens = model.predict(img_t, beam_width=args.beam, return_beams=True)  # list[list[int]]
            pred_str = decode_sequence(beam_tokens[0], idx2char)
            candidates = [decode_sequence(toks, idx2char) for toks in beam_tokens]
            oracle = beam_oracle(gt, candidates)
            rep = has_repetition(pred_str)

            rows.append({'true': gt, 'pred': pred_str, 'match': gt == pred_str,
                         'label': gt, 'path': str(img_path),
                         'oracle': oracle, 'repetition': rep,
                         'candidates': '|'.join(candidates)})
            if (i + 1) % 200 == 0:
                print(f"  evaluated {i+1}/{len(val_df)}  (e.g. {gt!r} -> {pred_str!r} {'' if oracle else '(oracle miss)'})")

    df = pd.DataFrame(rows)
    df['cer'] = df.apply(lambda r: cer(r['true'], r['pred']), axis=1)
    df.to_csv(args.out, index=False)
    print(f"[SAVED] {args.out}  ({len(df)} rows)")

    # ---- v3 exp2_clean과 동일한 집계 (동일 기준 공정 비교) ----
    # v3 run_eval은 args.csv=combined_labels.csv(13,386)로 label_counts 계산
    train_csv = str(V3_1 / "data/exp2_clean/combined_labels.csv")
    full_df = pd.read_csv(train_csv)
    label_counts = full_df['label'].value_counts().to_dict()

    predictions = df.to_dict('records')
    for p in predictions:
        p['group'] = 'low'
        c = label_counts.get(p['label'])
        if c is not None:
            if c >= 10: p['group'] = 'high'
            elif c >= 2: p['group'] = 'mid'

    acc = group_predictions(predictions, label_counts)
    summ = summarize(acc)
    print("\n=== [RESULT v3_1 (attention, beam={})] ===".format(args.beam))
    for g in ('high', 'mid', 'low'):
        s = summ[g]
        total = s['total']
        acc_str = f"{s['acc']:.1%}" if s['acc'] is not None else 'N/A'
        cer_str = f"{s['avg_cer']:.1%}" if s['avg_cer'] is not None else 'N/A'
        print(f"  {g:5s} total={total:4d}  exact_acc={acc_str:>7s}  avg_cer={cer_str}")

    # ---- v3_1 전용 확장: beam oracle + 반복토큰 통계 ----
    oracle_acc = oracle_group_predictions(predictions, label_counts)
    osumm = summarize(oracle_acc)
    print("\n=== [BEAM ORACLE (top-{} 후보 내 정답 포함)] ===".format(args.beam))
    for g in ('high', 'mid', 'low'):
        s = osumm[g]
        acc_str = f"{s['acc']:.1%}" if s['acc'] is not None else 'N/A'
        print(f"  {g:5s} oracle_acc={acc_str:>7s}  (total={s['total']})")

    n_rep = sum(1 for p in predictions if p.get('repetition', False))
    n_oracle_gain = sum(1 for p in predictions if p.get('oracle', False) and not p.get('match', False))
    print(f"\n=== [REPETITION] 반복토큰(연속동일문자≥3) 비율: {n_rep}/{len(predictions)} = {n_rep/len(predictions):.1%}")
    print(f"=== [ORACLE GAIN] top-1은 틀렸으나 후보에 정답 있음: {n_oracle_gain}장 (전체 {n_oracle_gain/len(predictions):.1%})")

    passed, details = acceptance(summ['high'], predictions)
    print(f"\n{acceptance_report(passed, details)}")

    # CSV에 group 반영
    df['group'] = [p['group'] for p in predictions]
    df.to_csv(args.out, index=False)
    print(f"\n[SAVED] {args.out} (final, {len(df)} rows)")


if __name__ == '__main__':
    main()
