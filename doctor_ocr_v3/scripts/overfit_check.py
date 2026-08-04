import re

log_path = 'logs/train_exp2.log'
train_epoch = {}
val_epoch = {}

with open(log_path, 'r', errors='ignore') as f:
    for line in f:
        # Epoch 마지막 Train 평균 loss (진행바 마지막 it/s 라인 형태: "Epoch X [Train]: 100%|...| 1674/1674 [00:23<00:00, 72.62it/s, loss=0.0005]")
        m = re.search(r'Epoch (\d+) \[Train\]:\s+100%[^\n]*?loss=([0-9.]+)\]', line)
        if m and int(m.group(1)) not in train_epoch:
            train_epoch[int(m.group(1))] = float(m.group(2))
        # [VAL] 요약 라인
        m2 = re.search(r'\[VAL\] Epoch (\d+): Loss=([0-9.]+), Acc=([0-9.]+)', line)
        if m2:
            val_epoch[int(m2.group(1))] = (float(m2.group(2)), float(m2.group(3)))

print("Epoch | train_L | val_L | val_acc | 갭(val_L - train_L)")
for ep in sorted(set(train_epoch) | set(val_epoch)):
    tl = train_epoch.get(ep)
    vl = val_epoch.get(ep, (None, None))[0]
    va = val_epoch.get(ep, (None, None))[1]
    if tl is not None and vl is not None:
        gap = vl - tl
        flag = "  <-- 갭 급증?" if (ep >= 40 and gap > 0.15) else ""
        print(f"{ep:5d} | {tl:.4f} | {vl:.4f} | {va:.4f} | {gap:+.4f}{flag}")
    else:
        print(f"{ep:5d} | {tl} | {vl} | {va} | -")

print()
print("=== 과적합 판정 근거 ===")
# 최초로 val_loss가 오른 뒤로 계속 올랐는지 (early stop epoch만 보면 됨)
best_ep = min(val_epoch, key=lambda k: val_epoch[k][0])
print(f"best val_loss 에포크: {best_ep} (val_loss={val_epoch[best_ep][0]:.4f})")
last_ep = max(val_epoch)
print(f"마지막 에포크: {last_ep} (val_loss={val_epoch[last_ep][0]:.4f})")
