"""Train CCNet 4-class trên Cityscapes (đã được tách 4-class).

Tham số đúng theo paper §III-B.4:
- Adam, lr=3e-4, betas=(0.9, 0.999), wd=1e-5
- Batch size = 4
- 47 epoch
- Loss = BCE-Dice main + 0.4 × BCE-Dice aux

Chạy:
    conda activate sac
    cd /home/huy/sac_project
    python -m scripts.new_branch.train --epochs 47 --batch-size 4

Output:
- Checkpoint best (theo val mIoU): models/best_ccnet_sac.pth
- Log epoch: outputs/new_branch/logs/train_log.csv
"""
from __future__ import annotations

# Cho phép chạy cả `python -m scripts.new_branch.train` lẫn `python train.py`
if __package__ in (None, ""):
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "scripts.new_branch"

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import CityscapesSAC, collate_default
from .losses import BCEDiceWithAux
from .model import build_segnet
from .paths import LOG_DIR, MODELS_DIR, NUM_CLASSES, SAC_CHECKPOINT


def fast_hist(label_true: np.ndarray, label_pred: np.ndarray, n_class: int) -> np.ndarray:
    mask = (label_true >= 0) & (label_true < n_class)
    return np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    eps = 1e-10
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + eps)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        H, W = lbls.shape[1], lbls.shape[2]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
        pred = logits.argmax(dim=1).cpu().numpy()
        hist += fast_hist(lbls.numpy(), pred, num_classes)
    ious = per_class_iou(hist)
    return float(ious.mean()), ious.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=47)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--ckpt-out", type=str, default=str(SAC_CHECKPOINT))
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--clip-grad", type=float, default=35.0,
                   help="Gradient L2 clip; CCNet mặc định 35. Đặt 0 để tắt.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    criterion = BCEDiceWithAux(num_classes=NUM_CLASSES, aux_weight=0.4)
    model = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=True, criterion=None)
    model = model.to(device)

    if args.resume is not None and Path(args.resume).exists():
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"Resumed from {args.resume}")

    train_ds = CityscapesSAC(split="train", augment=True)
    val_ds = CityscapesSAC(split="val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_default)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_default)

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                     weight_decay=args.weight_decay)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler('cuda', enabled=use_amp)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "train_log.csv"
    write_header = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log_w = csv.writer(log_f)
    if write_header:
        log_w.writerow(["epoch", "train_loss", "val_miou", "iou_ROI", "iou_sky",
                        "iou_construction", "iou_nature", "lr", "sec"])

    best_miou = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for it, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                preds = model(imgs)
                loss = criterion(preds, lbls)

            # NaN guard: bỏ batch nếu loss không hữu hạn (tránh poison BN running stats)
            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss tại iter {it}, bỏ batch.")
                continue

            scaler.scale(loss).backward()
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach().item() * imgs.size(0)
            n_seen += imgs.size(0)
            if (it + 1) % args.log_every == 0:
                print(f"  epoch {epoch+1:02d}/{args.epochs} iter {it+1:04d}/{len(train_loader)} "
                      f"loss={running_loss / n_seen:.4f}")

        epoch_loss = running_loss / max(n_seen, 1)
        val_miou, ious = evaluate(model, val_loader, device)
        sec = time.time() - t0
        print(f"[epoch {epoch+1:02d}] loss={epoch_loss:.4f}  val_mIoU={val_miou*100:.2f}%  "
              f"per-class={['%.2f%%' % (x * 100) for x in ious]}  ({sec:.1f}s)")
        log_w.writerow([epoch + 1, f"{epoch_loss:.6f}", f"{val_miou:.6f}",
                        f"{ious[0]:.6f}", f"{ious[1]:.6f}", f"{ious[2]:.6f}", f"{ious[3]:.6f}",
                        args.lr, f"{sec:.1f}"])
        log_f.flush()

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), args.ckpt_out)
            meta = {"epoch": epoch + 1, "val_miou": val_miou, "per_class_iou": ious}
            Path(args.ckpt_out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
            print(f"  ✔ saved best ckpt → {args.ckpt_out}  (val mIoU={val_miou*100:.2f}%)")

    log_f.close()
    print(f"Done. Best val mIoU = {best_miou*100:.2f}%")


if __name__ == "__main__":
    main()
