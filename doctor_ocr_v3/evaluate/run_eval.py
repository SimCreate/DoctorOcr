#!/usr/bin/env python3
"""
v3 평가 레이어 — 메인 실행 (baseline 재측정용)
v2_2의 모델/체크포인트/데이터를 재사용해 val 전체를 평가하고,
CER + 빈도그룹 + 수용기준 판정을 출력한다.

실행:
  /home/dev/doctor_ocr_v2_2/venv/bin/python run_eval.py

GPU: CUDA_VISIBLE_DEVICES='1' (GPU1 Max-Q, vLLM과 공존)
인자:
  --ckpt  체크포인트 경로 (기본: v2_2 best_model.pth)
  --csv   라벨 CSV (기본: combined_labels.csv)
  --out   결과 저장 경로 (기본: result_val.csv)
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# ---- v2_2 모델 및 평가 로직 재사용 ----
V2_2 = Path("/home/dev/doctor_ocr_v2_2")
sys.path.insert(0, str(V2_2 / "model"))
sys.path.insert(0, str(V2_2))
from model_v2_2 import CRNN  # noqa: E402
from evaluate import HandwritingDataset, ctc_collate_fn, evaluate_model  # noqa: E402

# ---- v3 지표 모듈 ----
CUR = Path(__file__).parent
sys.path.insert(0, str(CUR))
from metrics import cer  # noqa: E402
from aggregate import group_predictions, summarize  # noqa: E402
from acceptance import acceptance, acceptance_report, overall_cer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=str(V2_2 / "working" / "checkpoints" / "best_model.pth"))
    p.add_argument('--csv', type=str,
                   default=str(V2_2 / "dataset" / "combined_labels.csv"))
    p.add_argument('--out', type=str, default=str(CUR / "result_val.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    ckpt_path = Path(args.ckpt)
    print(f"[LOAD] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    vocab_size = ckpt.get('vocab_size')

    # char_dict
    with open(V2_2 / "working" / "char_dict.pkl", 'rb') as f:
        char_dict = pickle.load(f)
    idx2char = char_dict['idx2char']
    char2idx = char_dict['char2idx']

    # 모델 생성 및 로드
    model = CRNN(
        vocab_size=vocab_size,
        hidden_size=config.get('hidden_size', 256),
        dropout=config.get('dropout', 0.3),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', True),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[MODEL] epoch={ckpt.get('epoch')}, "
          f"val_loss={ckpt.get('val_loss'):.4f}, vocab={vocab_size}")

    # 데이터 로더 — 학습 시와 동일한 val split (seed 42, 80/20)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full = HandwritingDataset(args.csv, V2_2 / "dataset" / "img" / "img",
                              char2idx, transform, augment=False)
    torch.manual_seed(42)
    train_size = int(0.8 * len(full))
    _, val = torch.utils.data.random_split(full, [train_size, len(full) - train_size])
    val_loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=4,
                            pin_memory=True, collate_fn=ctc_collate_fn)
    print(f"[DATA] val={len(val)} samples")

    # 평가 실행 (v2_2의 evaluate_model: CTC loss + decode)
    val_loss, val_acc, predictions = evaluate_model(model, val_loader, device, idx2char)
    print(f"\n[EVAL] (기존 채점) val_loss={val_loss:.4f}, "
          f"exact_match_acc={val_acc:.4f} ({sum(p['match'] for p in predictions)}/{len(predictions)})")

    # 실제 라벨 빈도 계산 → 빈도그룹 분류
    df = pd.read_csv(args.csv)
    label_counts = df['label'].value_counts().to_dict()
    for p in predictions:
        p['group'] = 'low'
        c = label_counts.get(p['label'])
        if c is not None:
            if c >= 10:
                p['group'] = 'high'
            elif c >= 2:
                p['group'] = 'mid'
        p['cer'] = cer(p['true'], p['pred'])

    # 빈도그룹별 집계
    acc = group_predictions(predictions, label_counts)
    summ = summarize(acc)
    print("\n=== 빈도그룹별 결과 ===")
    for g in ('high', 'mid', 'low'):
        s = summ[g]
        total = s['total']
        acc_str = f"{s['acc']:.1%}" if s['acc'] is not None else 'N/A'
        cer_str = f"{s['avg_cer']:.1%}" if s['avg_cer'] is not None else 'N/A'
        print(f"  {g:5s} total={total:4d}  exact_acc={acc_str:>7s}  avg_cer={cer_str}")

    # 수용기준 판정
    passed, details = acceptance(summ['high'], predictions)
    print(f"\n{acceptance_report(passed, details)}")

    # 결과 저장
    out = pd.DataFrame(predictions)
    out.to_csv(args.out, index=False)
    print(f"\n[SAVED] {args.out} ({len(out)} rows)")


if __name__ == '__main__':
    main()
