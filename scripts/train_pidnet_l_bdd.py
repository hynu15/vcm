"""
train_pidnet_l_bdd.py — Train PIDNet-L 4-class trên BDD100K cho SAC pipeline.

4 class:  0=ROI  1=sky  2=construction  3=nature  (255=ignore)

Cách dùng (local, kiểm tra nhanh):
  conda run -n sac python3 scripts/train_pidnet_l_bdd.py --dry-run

Trên RunPod (train thật):
  conda run -n sac python3 /workspace/sac_project/scripts/train_pidnet_l_bdd.py \
      --data-root /workspace/data/gt_4class_bdd \
      --img-root  /workspace/data/archive/bdd100k_seg/seg/images \
      --epochs 100 --batch_size 8 --image_size 360,640

Checkpoint lưu tại: models/best_pidnet_l_4class_bdd.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import PIDNet-L architecture và loss từ script gốc
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from train_pidnet_l import (   # noqa: E402
    build_pidnet_l,
    generate_boundary_gt,
    boundary_loss,
    poly_lr_with_warmup,
)

algc = False


def semantic_loss(logits, target):
    """CE loss với ignore_index=255 cho BDD100K."""
    H, W = target.shape[-2], target.shape[-1]
    logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=algc)
    return F.cross_entropy(logits_up, target, ignore_index=IGNORE_INDEX)


def bas_loss(main_logits, target, boundary_gt, threshold=0.8):
    """BAS loss — bỏ qua pixel ignore=255."""
    H, W = target.shape[-2], target.shape[-1]
    logits_up = F.interpolate(main_logits, size=(H, W), mode="bilinear", align_corners=algc)
    mask = (boundary_gt > threshold) & (target != IGNORE_INDEX)
    _, C, _, _ = logits_up.shape
    logits_flat = logits_up.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.reshape(-1)
    mask_flat   = mask.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=main_logits.device, requires_grad=True)
    return F.cross_entropy(logits_flat[mask_flat], target_flat[mask_flat])

PROJECT_ROOT = os.path.abspath(os.path.join(_scripts_dir, ".."))
NUM_CLASSES  = 4
IGNORE_INDEX = 255

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BDD100KSAC(Dataset):
    """BDD100K 4-class dataset cho SAC pipeline.

    Ghép ảnh gốc (img_root/{split}/*.jpg|png) với nhãn đã convert
    (lbl_root/{split}/*.png từ prepare_bdd_labels.py).
    """

    def __init__(self, img_root: str, lbl_root: str,
                 split: str = "train", size_hw: tuple = (360, 640),
                 augment: bool | None = None):
        self.size_hw = size_hw
        self.augment = augment if augment is not None else (split == "train")

        img_dir = Path(img_root) / split
        lbl_dir = Path(lbl_root) / split

        if not img_dir.is_dir():
            raise RuntimeError(f"Không tìm thấy ảnh tại: {img_dir}")
        if not lbl_dir.is_dir():
            raise RuntimeError(
                f"Không tìm thấy nhãn tại: {lbl_dir}\n"
                "  → Hãy chạy prepare_bdd_labels.py trước."
            )

        self.pairs: list[tuple[Path, Path]] = []
        for ext in ("*.jpg", "*.png"):
            for img_path in sorted(img_dir.glob(ext)):
                # Thử cả hai naming convention: stem.png và stem_train_id.png
                for lbl_name in (img_path.stem + "_train_id.png",
                                 img_path.stem + ".png"):
                    lbl_path = lbl_dir / lbl_name
                    if lbl_path.exists():
                        self.pairs.append((img_path, lbl_path))
                        break

        if not self.pairs:
            raise RuntimeError(
                f"Không tìm thấy cặp (ảnh, nhãn) trong\n"
                f"  img : {img_dir}\n"
                f"  lbl : {lbl_dir}\n"
                "  Kiểm tra lại --img-root và --data-root."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.pairs[idx]
        H, W = self.size_hw

        img = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        lbl = Image.open(lbl_path).resize((W, H), Image.NEAREST)

        img_np = np.array(img, dtype=np.float32)
        lbl_np = np.array(lbl, dtype=np.uint8)

        if self.augment:
            if random.random() < 0.5:
                img_np = img_np[:, ::-1, :].copy()
                lbl_np = lbl_np[:, ::-1].copy()
            if random.random() < 0.5:
                img_np = np.clip(img_np * random.uniform(0.8, 1.2), 0, 255)

        img_np = (img_np / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_t  = torch.from_numpy(img_np.transpose(2, 0, 1).copy()).float()
        lbl_t  = torch.from_numpy(lbl_np.astype(np.int64))

        return img_t, lbl_t


# ── mIoU (4 class, bỏ qua pixel ignore) ──────────────────────────────────────

def compute_miou(pred: torch.Tensor, target: torch.Tensor) -> float:
    ious = []
    for c in range(NUM_CLASSES):
        valid = target != IGNORE_INDEX
        p = (pred == c) & valid
        t = (target == c) & valid
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    valid_ious = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid_ious)) if valid_ious else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device     : {device} — {dev_name}")
    print(f"num_classes: {NUM_CLASSES}")
    print(f"image_size : {args.image_size}")
    print(f"data-root  : {args.data_root}")
    print(f"img-root   : {args.img_root}")

    h_str, w_str = args.image_size.split(",")
    size_hw = (int(h_str), int(w_str))

    train_ds = BDD100KSAC(args.img_root, args.data_root, "train", size_hw, augment=True)
    val_ds   = BDD100KSAC(args.img_root, args.data_root, "val",   size_hw, augment=False)
    print(f"Train: {len(train_ds)} ảnh  |  Val: {len(val_ds)} ảnh")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model    = build_pidnet_l(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PIDNet-L   : {n_params / 1e6:.1f}M params")

    # Kiểm tra forward pass trước khi train
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, size_hw[0], size_hw[1]).to(device)
        outs  = model(dummy)
        print(f"Forward OK : x_p={list(outs[0].shape)}  "
              f"x_={list(outs[1].shape)}  x_d={list(outs[2].shape)}")
    model.train()

    if args.dry_run:
        print("\n[dry-run] Forward pass OK. Thoát — không train.")
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    use_amp  = torch.cuda.is_available()
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_iters  = args.epochs * len(train_loader)
    warmup_iters = 5 * len(train_loader)

    model_dir   = os.path.join(PROJECT_ROOT, "models")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    ckpt_path = os.path.join(model_dir,   "best_pidnet_l_4class_bdd.pth")
    csv_path  = os.path.join(metrics_dir, "train_metrics_pidnet_l_bdd.csv")
    best_json = os.path.join(metrics_dir, "best_result_pidnet_l_bdd.json")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "miou", "best_miou", "is_best",
             "l0", "l1", "l2", "l3", "total_loss"]
        )

    best_miou  = 0.0
    best_epoch = -1
    global_iter = 0

    for epoch in range(args.epochs):
        model.train()
        stats    = {k: 0.0 for k in ("l0", "l1", "l2", "l3", "tot")}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for imgs, lbls in pbar:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Boundary GT: dùng valid pixels (ignore → 0 tạm)
            valid_lbls = lbls.clone()
            valid_lbls[valid_lbls == IGNORE_INDEX] = 0
            bnd_gt = generate_boundary_gt(valid_lbls, kernel_size=7)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                x_p, x_, x_d = model(imgs)
                l0 = semantic_loss(x_p, lbls)
                bnd_ratio = bnd_gt.mean().item()
                pw = min(max(1.0, (1 - bnd_ratio) / (bnd_ratio + 1e-6)), 20.0)
                l1 = boundary_loss(x_d, bnd_gt, pos_weight_factor=pw)
                l2 = semantic_loss(x_, lbls)
                l3 = bas_loss(x_, lbls, bnd_gt, threshold=0.8)
                loss = 0.4 * l0 + 20.0 * l1 + 1.0 * l2 + 1.0 * l3

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                global_iter += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            cur_lr = poly_lr_with_warmup(optimizer, global_iter, total_iters,
                                         args.lr, warmup_iters)
            global_iter += 1
            n_batches  += 1

            for k, v in zip(("l0", "l1", "l2", "l3", "tot"),
                            (l0.item(), l1.item(), l2.item(), l3.item(), loss.item())):
                stats[k] += v

            pbar.set_postfix(
                l2=f"{l2.item():.3f}",
                tot=f"{loss.item():.3f}",
                lr=f"{cur_lr:.5f}",
            )

        if n_batches == 0:
            print(f"  [WARN] Epoch {epoch + 1}: không có batch hợp lệ.")
            continue

        avgs = {k: v / n_batches for k, v in stats.items()}
        print(f"\n  Epoch {epoch + 1} — "
              f"l0={avgs['l0']:.4f}  l1={avgs['l1']:.4f}  "
              f"l2={avgs['l2']:.4f}  l3={avgs['l3']:.4f}  "
              f"total={avgs['tot']:.4f}")

        # ── Validation mIoU ───────────────────────────────────────────────────
        model.eval()
        all_ious: list[float] = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, x_, _ = model(imgs)
                x_up = F.interpolate(x_, size=lbls.shape[-2:],
                                     mode="bilinear", align_corners=False)
                pred = x_up.argmax(dim=1)
                for p, t in zip(pred, lbls):
                    all_ious.append(compute_miou(p, t))

        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        is_best  = mean_iou > best_miou
        print(f"  Val mIoU : {mean_iou:.4f}  │  best={max(best_miou, mean_iou):.4f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, mean_iou, max(best_miou, mean_iou), int(is_best),
                avgs["l0"], avgs["l1"], avgs["l2"], avgs["l3"], avgs["tot"],
            ])

        if is_best:
            best_miou  = mean_iou
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "model_name":  "pidnet_l_4class_bdd",
                        "num_classes": NUM_CLASSES,
                        "best_miou":   float(best_miou),
                        "best_epoch":  int(best_epoch),
                    },
                },
                ckpt_path,
            )
            with open(best_json, "w") as f:
                json.dump({"best_miou": float(best_miou), "best_epoch": int(best_epoch),
                           "checkpoint": ckpt_path}, f, indent=2)
            print(f"  [SAVED] mIoU={best_miou:.4f} → {ckpt_path}")

    print(f"\nXong. Best mIoU={best_miou:.4f} tại epoch {best_epoch}.")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Metrics CSV: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train PIDNet-L 4-class trên BDD100K cho SAC pipeline"
    )
    ap.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "gt_4class_bdd"),
        help="Thư mục nhãn đã convert bởi prepare_bdd_labels.py",
    )
    ap.add_argument(
        "--img-root", default=None,
        help="Thư mục ảnh gốc chứa train/ val/. Nếu không truyền thì tự tìm trong data/archive/",
    )
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch_size", type=int,   default=2)
    ap.add_argument("--lr",         type=float, default=1e-2)
    ap.add_argument(
        "--image_size", type=str, default="360,640",
        help="HxW — BDD100K gốc 720×1280, mặc định downscale 2× thành 360×640",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ kiểm tra dataset + forward pass, không train",
    )
    args = ap.parse_args()

    # Auto-detect img-root nếu không truyền
    if args.img_root is None:
        for candidate in [
            os.path.join(PROJECT_ROOT, "data", "archive", "bdd100k_seg", "seg", "images"),
            os.path.join(PROJECT_ROOT, "data", "archive", "bdd100k_seg", "images", "10k"),
            os.path.join(PROJECT_ROOT, "data", "archive", "bdd100k", "images", "10k"),
        ]:
            if os.path.isdir(os.path.join(candidate, "train")):
                args.img_root = candidate
                print(f"[auto] img-root = {candidate}")
                break
        if args.img_root is None:
            print("[LỖI] Không tìm thấy img-root. Truyền --img-root <đường_dẫn>")
            sys.exit(1)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train(args)
