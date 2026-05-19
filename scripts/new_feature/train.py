"""
Train CCNet-4Class on Cityscapes 4-class labels.

Matches the training setup described in the paper (Section IV-A & IV-B):
  - Optimiser: Adam, lr=3e-4 (head) / 3e-5 (backbone), betas=(0.9,0.999), weight_decay=1e-5
  - LR schedule: poly decay  lr = base_lr × (1 − epoch/max_epochs)^0.9
  - Loss: BCE-Dice  (Eq. 8-10)
  - Epochs: 60 (47 in paper; extended slightly to compensate for poly decay warmup)
  - Batch size: 4 (paper default; override with --batch-size)
  - Input resolution: 512×1024  (paper: downsized from 2048×1024)
  - 4-class labels: 0=ROI, 1=sky, 2=construction, 3=nature

Usage (from scripts/ directory, conda sac env):
  # Fresh train (recommended after architecture update)
  python new_feature/train.py --reset

  # Continue from existing checkpoint
  python new_feature/train.py

The best checkpoint (highest val mIoU) is saved to:
  <project>/models/best_ccnet_4class.pth

Training metrics are written to:
  <project>/outputs/new_feature/training/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from new_feature.ccnet_4class import CCNet4Class
from new_feature.dataset import Cityscapes4Class

PROJECT_ROOT = _HERE.parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "new_feature" / "training"


# ─────────────────────────────────────────────────────────────────────────────
# Loss: BCE-Dice  (Eq. 8-10 in paper)
# ─────────────────────────────────────────────────────────────────────────────

def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Combined Cross-Entropy + Dice loss (Eq. 10)."""
    with torch.amp.autocast("cuda", enabled=False):
        p = logits.float()
        t = targets

        ce = nn.CrossEntropyLoss()(p, t)

        onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
        soft   = F.softmax(p, dim=1)
        dice = 1.0 - (
            (2.0 * (soft * onehot).sum(dim=(2, 3)) + 1.0)
            / (soft.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3)) + 1.0)
        )
        return ce + dice.mean()


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def poly_lr(base_lr: float, epoch: int, max_epochs: int, power: float = 0.9) -> float:
    """Polynomial LR decay — standard for dense-prediction segmentation."""
    return base_lr * (1.0 - epoch / max_epochs) ** power


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_miou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> float:
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum().item()
        union = ((pred == c) | (gt == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    img_train = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train"
    lbl_train = PROJECT_ROOT / "data" / "gt_4class" / "train"
    img_val   = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
    lbl_val   = PROJECT_ROOT / "data" / "gt_4class" / "val"

    train_ds = Cityscapes4Class(img_train, lbl_train, augment=True)
    val_ds   = Cityscapes4Class(img_val,   lbl_val,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ────────────────────────────────────────────────────────────────
    ckpt_path = MODEL_DIR / "best_ccnet_4class.pth"
    model = CCNet4Class(num_classes=4, pretrained=True).to(device)

    start_epoch = 0
    best_miou   = 0.0

    if ckpt_path.is_file() and not args.reset:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Attempt to load weights; skip mismatched keys gracefully
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[WARNING] Checkpoint loaded with {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys — partial load (architecture changed?).")
        best_miou = ckpt.get("meta", {}).get("best_miou", 0.0)
        print(f"Resumed from {ckpt_path}  (best mIoU so far: {best_miou:.4f})")
    elif args.reset:
        print("--reset: training from scratch with ImageNet-pretrained backbone.")

    # ── Optimiser: separate LR for backbone vs head ──────────────────────────
    # Backbone (pretrained): 10× lower LR so it fine-tunes gently.
    # Head (random init): full LR.
    backbone_params = (
        list(model.stem.parameters())
        + list(model.layer1.parameters())
        + list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )
    head_params = (
        list(model.reduction.parameters())
        + list(model.cc1.parameters())
        + list(model.cc2.parameters())
        + list(model.seg_head.parameters())
    )

    BASE_LR_BACKBONE = args.lr * 0.1
    BASE_LR_HEAD     = args.lr

    optim = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": BASE_LR_BACKBONE, "initial_lr": BASE_LR_BACKBONE},
            {"params": head_params,     "lr": BASE_LR_HEAD,     "initial_lr": BASE_LR_HEAD},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Logging ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "train_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr_head", "train_loss", "val_miou", "best_miou"])

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):

        # Poly LR decay
        for pg in optim.param_groups:
            pg["lr"] = poly_lr(pg["initial_lr"], epoch, args.epochs)
        cur_lr = optim.param_groups[1]["lr"]  # head LR for logging

        # Train
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for imgs, lbls, _ in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss   = bce_dice_loss(logits, lbls, num_classes=4)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validate
        model.eval()
        val_ious: list[float] = []
        with torch.no_grad():
            for imgs, lbls, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(1)
                for p, t in zip(preds.cpu().numpy(), lbls.cpu().numpy()):
                    val_ious.append(batch_miou(p, t, num_classes=4))

        val_miou = float(np.mean(val_ious))
        is_best  = val_miou > best_miou

        print(f"Epoch {epoch+1:3d}  lr={cur_lr:.2e}  loss={avg_loss:.4f}"
              f"  val_mIoU={val_miou:.4f}  {'★ best' if is_best else ''}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, cur_lr, avg_loss, val_miou,
                                    max(best_miou, val_miou)])

        if is_best:
            best_miou = val_miou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "model_name": "ccnet_4class",
                        "num_classes": 4,
                        "best_miou":   float(best_miou),
                        "best_epoch":  epoch + 1,
                        "arch":        "dilated_stride8",
                    },
                },
                ckpt_path,
            )
            json_path = OUTPUT_DIR / "best_result.json"
            json_path.write_text(
                json.dumps({"best_miou": float(best_miou), "best_epoch": epoch + 1}, indent=2)
            )
            print(f"  Saved best model  mIoU={best_miou:.4f}  → {ckpt_path}")

    print(f"\nTraining complete.  Best val mIoU = {best_miou:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Metrics CSV: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CCNet-4Class (SAC paper)")
    p.add_argument("--epochs",     type=int,   default=60,
                   help="Training epochs (paper: 47; 60 recommended with poly LR)")
    p.add_argument("--batch-size", type=int,   default=4,   help="Batch size")
    p.add_argument("--lr",         type=float, default=3e-4,
                   help="Base LR for head (backbone gets lr×0.1)")
    p.add_argument("--workers",    type=int,   default=4,   help="DataLoader num_workers")
    p.add_argument("--reset",      action="store_true",
                   help="Ignore existing checkpoint and retrain from scratch")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
