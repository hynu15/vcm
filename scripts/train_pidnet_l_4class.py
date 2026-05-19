# ------------------------------------------------------------------------------
# PIDNet-L training script for 4-class semantic segmentation on Cityscapes.
#
# Classes: 0=ROI, 1=sky, 2=construction, 3=nature
# Prepare labels first: python prepare_4class_labels.py
#
# Architecture identical to train_pidnet_l.py (Xu et al., CVPR 2023).
# Only num_classes, dataset, checkpoint names, and IoU loop differ.
# ------------------------------------------------------------------------------

import os
import csv
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 4-class dataset + augmentation
from train_segmentation_4class import Cityscapes4Class
from train_segmentation import CompressionArtifactAugmentation

# All model building blocks re-used from train_pidnet_l
from train_pidnet_l import (
    build_pidnet_l,
    generate_boundary_gt,
    semantic_loss,
    boundary_loss,
    bas_loss,
    poly_lr_with_warmup,
)

NUM_CLASSES = 4
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CLASS_NAMES = ['ROI', 'sky', 'construction', 'nature']


# ==============================================================================
# 4-class specific utilities
# ==============================================================================

def compute_class_weights(dataset, num_classes, num_samples=300, device='cpu'):
    """Median-frequency class weights từ subset ngẫu nhiên của dataset.

    Class hiếm (sky, nature) nhận weight cao hơn để chống imbalance.
    Dùng median-frequency balancing: weight_c = median(freq) / freq_c.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    print(f"  Computing class weights from {len(indices)} samples...")
    for i in indices:
        _, label = dataset[i]
        lbl = label.numpy() if hasattr(label, 'numpy') else np.array(label)
        for c in range(num_classes):
            counts[c] += int((lbl == c).sum())
        total_pixels += lbl.size
    freqs = counts / (total_pixels + 1e-8)
    present = freqs > 0
    if not present.any():
        return None
    median_f = float(np.median(freqs[present]))
    weights = np.where(present, median_f / (freqs + 1e-8), 1.0)
    # Normalize: mean weight of present classes = 1
    weights = weights / (weights[present].mean() + 1e-8)
    for c in range(num_classes):
        print(f"    {CLASS_NAMES[c]:15s}: freq={freqs[c]:.3f}  weight={weights[c]:.3f}")
    return torch.FloatTensor(weights).to(device)


def semantic_loss_4class(logits, target, class_weights=None):
    """Cross-entropy với class weights để xử lý imbalance 4-class."""
    target_h, target_w = target.shape[-2], target.shape[-1]
    logits_up = F.interpolate(logits, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
    return F.cross_entropy(logits_up, target, weight=class_weights)


def bas_loss_4class(main_logits, target, boundary_gt, class_weights=None, threshold=0.6):
    """BAS loss với class weights và threshold thấp hơn (0.6 vs 0.8) cho 4-class.

    4-class có nhiều loại boundary hơn nên cần threshold thấp hơn để
    bắt đủ hard pixels.
    """
    target_h, target_w = target.shape[-2], target.shape[-1]
    logits_up = F.interpolate(main_logits, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
    mask = (boundary_gt > threshold)
    _, C, _, _ = logits_up.shape
    logits_flat = logits_up.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.reshape(-1)
    mask_flat = mask.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=main_logits.device, requires_grad=True)
    return F.cross_entropy(logits_flat[mask_flat], target_flat[mask_flat],
                           weight=class_weights)


# ==============================================================================
# Training loop
# ==============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else 'CPU')
    print(f"Device: {device} - {device_name}")
    print(f"Num classes: {NUM_CLASSES}  (0=ROI, 1=sky, 2=construction, 3=nature)")

    image_train_dir = os.path.join(
        PROJECT_ROOT, "data", "gt_4class",
        "leftImg8bit_trainvaltest", "leftImg8bit", "train")
    label_train_dir = os.path.join(PROJECT_ROOT, "data", "gt_4class", "train")
    image_val_dir = os.path.join(
        PROJECT_ROOT, "data", "gt_4class",
        "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    label_val_dir = os.path.join(PROJECT_ROOT, "data", "gt_4class", "val")

    model_dir   = os.path.join(PROJECT_ROOT, "models")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    h_str, w_str = args.image_size.split(',')
    image_size = (int(h_str), int(w_str))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        CompressionArtifactAugmentation(p=0.7, min_quality=35, max_quality=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = Cityscapes4Class(
        image_train_dir, label_train_dir, train_transform, image_size=image_size)
    val_ds = Cityscapes4Class(
        image_val_dir, label_val_dir, val_transform, image_size=image_size)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Train dataset rỗng. Hãy chạy prepare_4class_labels.py để tạo nhãn 4-class.")
    if len(val_ds) == 0:
        raise RuntimeError(
            "Val dataset rỗng. Hãy kiểm tra dữ liệu tại data/gt_4class/val.")

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Compute class weights để chống imbalance (class 0 ROI thường chiếm ~45%)
    if args.class_weights:
        print("Computing class weights...")
        class_weights = compute_class_weights(
            train_ds, NUM_CLASSES, num_samples=300, device=device)
    else:
        class_weights = None
        print("Class weights: disabled (--no-class-weights)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    model = build_pidnet_l(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PIDNet-L parameters: {n_params / 1e6:.2f}M")

    # Quick forward shape check
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, image_size[0], image_size[1]).to(device)
        outs = model(dummy)
        print(f"Forward output shapes: "
              f"x_extra_p={list(outs[0].shape)}, "
              f"x_={list(outs[1].shape)}, "
              f"x_extra_d={list(outs[2].shape)}")
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    algc = False  # align_corners consistent with train_pidnet_l

    warmup_epochs = 5
    total_iters   = args.epochs * len(train_loader)
    warmup_iters  = warmup_epochs * len(train_loader)

    metrics_csv       = os.path.join(metrics_dir, "train_metrics_pidnet_l_4class.csv")
    best_result_json  = os.path.join(metrics_dir, "best_result_pidnet_l_4class.json")
    checkpoint_path   = os.path.join(model_dir,   "best_pidnet_l_4class.pth")

    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'mean_iou', 'best_miou_so_far', 'is_best',
            'loss_l0', 'loss_l1', 'loss_l2', 'loss_l3', 'total_loss',
            'iou_ROI', 'iou_sky', 'iou_construction', 'iou_nature',
        ])

    best_miou  = 0.0
    best_epoch = -1
    global_iter = 0

    for epoch in range(args.epochs):
        model.train()

        epoch_l0 = epoch_l1 = epoch_l2 = epoch_l3 = epoch_total = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            boundary_gt = generate_boundary_gt(labels, kernel_size=7)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                x_extra_p, x_, x_extra_d = outputs

                l0 = semantic_loss_4class(x_extra_p, labels, class_weights)

                boundary_ratio = boundary_gt.mean().item()
                pw = max(1.0, (1.0 - boundary_ratio) / (boundary_ratio + 1e-6))
                pw = min(pw, 20.0)
                l1 = boundary_loss(x_extra_d, boundary_gt, pos_weight_factor=pw)

                l2 = semantic_loss_4class(x_, labels, class_weights)
                l3 = bas_loss_4class(x_, labels, boundary_gt, class_weights, threshold=0.6)

                loss = 0.4 * l0 + 20.0 * l1 + 1.0 * l2 + 1.0 * l3

            if not torch.isfinite(loss):
                print(f"  [WARNING] Non-finite loss at iter {global_iter}, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                global_iter += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            cur_lr = poly_lr_with_warmup(
                optimizer, global_iter, total_iters, args.lr, warmup_iters)
            global_iter += 1

            epoch_l0 += l0.item()
            epoch_l1 += l1.item()
            epoch_l2 += l2.item()
            epoch_l3 += l3.item()
            epoch_total += loss.item()
            n_batches += 1

            pbar.set_postfix(
                l0=f"{l0.item():.3f}",
                l1=f"{l1.item():.3f}",
                l2=f"{l2.item():.3f}",
                l3=f"{l3.item():.3f}",
                tot=f"{loss.item():.3f}",
                lr=f"{cur_lr:.5f}",
                bnd=f"{boundary_ratio:.3f}",
            )

        if n_batches == 0:
            print(f"  [WARNING] No valid batches in epoch {epoch + 1}.")
            continue

        avg_l0  = epoch_l0  / n_batches
        avg_l1  = epoch_l1  / n_batches
        avg_l2  = epoch_l2  / n_batches
        avg_l3  = epoch_l3  / n_batches
        avg_tot = epoch_total / n_batches

        print(f"\n--- Epoch {epoch + 1} train summary ---")
        print(f"  l0 (P-CE  x0.4) : {avg_l0:.4f}")
        print(f"  l1 (D-BCE x20 ) : {avg_l1:.4f}")
        print(f"  l2 (main CE x1) : {avg_l2:.4f}")
        print(f"  l3 (BAS    x1 ) : {avg_l3:.4f}")
        print(f"  total loss      : {avg_tot:.4f}")

        # ----------------------------------------------------------------
        # Validation: mIoU over 4 classes (aggregate inter/union across full val set)
        # ----------------------------------------------------------------
        model.eval()
        total_inter = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_union = np.zeros(NUM_CLASSES, dtype=np.float64)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    x_extra_p, x_, x_extra_d = outputs
                    x_up = F.interpolate(
                        x_, size=(labels.shape[-2], labels.shape[-1]),
                        mode='bilinear', align_corners=algc)
                pred = torch.argmax(x_up, dim=1)
                for p, t in zip(pred, labels):
                    for c in range(NUM_CLASSES):
                        total_inter[c] += ((p == c) & (t == c)).sum().item()
                        total_union[c] += ((p == c) | (t == c)).sum().item()

        # Standard mIoU: average over classes present in dataset (skip union==0)
        per_class_iou = []
        for c in range(NUM_CLASSES):
            if total_union[c] > 0:
                per_class_iou.append(total_inter[c] / total_union[c])
            else:
                per_class_iou.append(None)

        valid_ious = [v for v in per_class_iou if v is not None]
        mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
        is_best  = mean_iou > best_miou

        print(f"  Val mIoU ({NUM_CLASSES}-class): {mean_iou:.4f}  |  Best so far: {max(best_miou, mean_iou):.4f}")
        for c in range(NUM_CLASSES):
            v = per_class_iou[c]
            print(f"    {CLASS_NAMES[c]:15s}: {v:.4f}" if v is not None else f"    {CLASS_NAMES[c]:15s}: N/A")

        with open(metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(mean_iou),
                float(max(best_miou, mean_iou)),
                int(is_best),
                avg_l0, avg_l1, avg_l2, avg_l3, avg_tot,
                *(v if v is not None else '' for v in per_class_iou),
            ])

        if is_best:
            best_miou  = mean_iou
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': {
                    'model_name':  'pidnet_l',
                    'num_classes': NUM_CLASSES,
                    'best_miou':   float(best_miou),
                    'best_epoch':  int(best_epoch),
                },
            }, checkpoint_path)

            with open(best_result_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name':      'pidnet_l',
                    'best_miou':       float(best_miou),
                    'best_epoch':      int(best_epoch),
                    'num_epochs':      int(args.epochs),
                    'num_classes':     NUM_CLASSES,
                    'checkpoint_path': checkpoint_path,
                }, f, ensure_ascii=False, indent=2)

            print(f"  [SAVED] New best: mIoU = {best_miou:.4f} "
                  f"at epoch {best_epoch} -> {checkpoint_path}")

    print(f"\nTraining complete. Best mIoU = {best_miou:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Metrics CSV: {metrics_csv}")
    if best_epoch > 0:
        print(f"Best result JSON: {best_result_json}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train PIDNet-L for 4-class segmentation (ROI/sky/construction/nature)")
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-2)
    parser.add_argument('--image_size', type=str,   default='512,1024',
                        help='HxW as "H,W"')
    parser.add_argument('--no-class-weights', dest='class_weights',
                        action='store_false', default=True,
                        help='Tắt median-frequency class weights (mặc định: bật)')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train(args)
