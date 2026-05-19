"""
Evaluate SAC vs traditional compression, reproducing Tables II and IV from:

  Wang et al., "Semantic-Aware Video Compression for Automotive Cameras",
  IEEE Trans. Intelligent Vehicles, Vol. 8, No. 6, June 2023.

Results
-------
Table II equivalent (Cityscapes) — PSNR, SSIM, SA-PSNR, SA-SSIM:
  H.264 / SA-X264 / H.265 / SA-X265 at multiple CRF operating points.

Table IV equivalent — segmentation metrics (mIoU, iIoU) on decompressed frames.

Usage (from scripts/ dir, conda sac env):

  # Val split (default)
  python new_feature/evaluate.py --num-frames 30

  # Test split — strasbourg + ulm, uses best_pidnet_l_4class.pth by default
  python new_feature/evaluate.py --split test --num-frames 50

  # Test split with explicit model path
  python new_feature/evaluate.py --split test --model-path models/best_pidnet_l_4class.pth

  # Full evaluation (matches paper scale)
  python new_feature/evaluate.py --num-frames 500

Outputs (under outputs/new_feature/evaluation/<timestamp>/):
  table2_cityscapes.csv   — Table II equivalent
  table4_segmentation.csv — Table IV equivalent
  per_frame_metrics.csv   — raw per-frame data
  results_summary.txt     — human-readable comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from new_feature.ccnet_4class import CCNet4Class, load_ccnet_4class
from new_feature.compression import compress_sac, compress_traditional, macroblock_align_filter
from new_feature.dataset import Cityscapes4Class
from new_feature.metrics import (
    compute_all_compression_metrics,
    compute_miou,
    compute_iiou,
)

PROJECT_ROOT = _HERE.parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
# Fallback: use existing best PIDNet / CCNet if new model not yet trained
FALLBACK_PATHS = [
    PROJECT_ROOT / "models" / "best_pidnet_l.pth",
    PROJECT_ROOT / "models" / "best_ccnet.pth",
    PROJECT_ROOT / "models" / "best_pidnet.pth",
]

IMG_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LBL_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "val"

IMG_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LBL_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"
# Cityscapes test split: only these 2 cities have 4-class ground-truth labels
TEST_CITIES = ["strasbourg", "ulm"]


# ─────────────────────────────────────────────────────────────────────────────
# CRF configurations  (Table I / Table II / Table IV in paper)
# ─────────────────────────────────────────────────────────────────────────────

class Config(NamedTuple):
    method:  str   # 'H.264', 'SA-X264', 'H.265', 'SA-X265'
    codec:   str   # 'libx264' or 'libx265'
    crf_roi: int
    crf_non: int   # == crf_roi for traditional methods

    @property
    def is_sac(self) -> bool:
        return self.crf_roi != self.crf_non

    @property
    def label(self) -> str:
        return f"{self.method}({self.crf_roi},{self.crf_non})"


# Table II configurations (Cityscapes)
TABLE2_CONFIGS: List[Config] = [
    Config("H.264",   "libx264", 18, 18),
    Config("SA-X264", "libx264", 18, 23),
    Config("H.264",   "libx264", 23, 23),
    Config("SA-X264", "libx264", 18, 27),
    Config("H.264",   "libx264", 27, 27),
    Config("SA-X264", "libx264", 23, 27),
    Config("H.265",   "libx265", 23, 23),
    Config("SA-X265", "libx265", 23, 28),
    Config("H.265",   "libx265", 28, 28),
    Config("SA-X265", "libx265", 23, 32),
    Config("H.265",   "libx265", 32, 32),
    Config("SA-X265", "libx265", 28, 32),
]

# Table IV configurations (segmentation evaluation)
TABLE4_CONFIGS: List[Config] = [
    Config("H.264",   "libx264", 23, 23),
    Config("SA-X264", "libx264", 18, 27),
    Config("H.265",   "libx265", 28, 28),
    Config("SA-X265", "libx265", 23, 32),
]


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation inference helper
# ─────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SEG_SIZE = (512, 1024)   # (H, W) — paper inference resolution


def _segment(model: torch.nn.Module, device: torch.device, frame_rgb: np.ndarray) -> np.ndarray:
    """
    Run segmentation model on one RGB frame (any resolution).

    Returns
    -------
    pred : uint8 [H, W]  with values in {0,1,2,3}
           at the same spatial resolution as frame_rgb.
    """
    H, W = frame_rgb.shape[:2]
    # Resize to model's training resolution
    img_pil = Image.fromarray(frame_rgb).resize((SEG_SIZE[1], SEG_SIZE[0]), Image.BILINEAR)
    tensor  = _TRANSFORM(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)          # [1, C, H, W]
        pred   = logits.argmax(1)[0]    # [H, W]

    # Resize prediction back to original frame resolution
    pred_full = F.interpolate(
        pred.float().unsqueeze(0).unsqueeze(0),
        size=(H, W), mode="nearest",
    )[0, 0].byte().cpu().numpy()

    return pred_full


def _get_roi_mask(seg_pred: np.ndarray) -> np.ndarray:
    """
    Binary ROI mask from 4-class prediction.
    ROI (class 0) → 1, non-ROI (classes 1,2,3) → 0.
    """
    return macroblock_align_filter((seg_pred == 0).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Load segmentation model
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(device: torch.device,
                model_path: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
    """
    Load segmentation model from checkpoint, auto-detecting architecture.

    Supports:
      - PIDNet-L (keys start with 'conv1.')
      - CCNet4Class (new, keys: stem/layer1/layer2/…)
      - LegacyCCNetResNet101 (keys: backbone.X / cc1 / cc2 / conv)
        with num_classes=2 or num_classes=4
    """
    from train_segmentation import build_segmentation_model

    def _detect_and_load(path: Path) -> Tuple[torch.nn.Module, int]:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        meta    = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        state   = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        num_cls = int(meta.get("num_classes", 4))
        model_name = meta.get("model_name", "")

        keys = list(state.keys())
        is_legacy = any(k.startswith("backbone.") for k in keys)
        is_new    = any(k.startswith("stem.") or k.startswith("layer1.") for k in keys)

        if model_name.startswith("pidnet") or (
            not is_legacy and not is_new and any(k.startswith("conv1.") for k in keys)
        ):
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(path), device, num_classes=num_cls)
        elif is_new:
            from new_feature.ccnet_4class import CCNet4Class
            model = CCNet4Class(num_classes=num_cls, pretrained=False).to(device)
            model.load_state_dict(state)
        elif is_legacy:
            model = build_segmentation_model("ccnet", num_classes=num_cls,
                                             backbone_weights=None).to(device)
            model.load_state_dict(state)
        else:
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(path), device, num_classes=num_cls)

        model.num_classes = num_cls
        model.eval()
        return model, num_cls

    # 1. Explicit path from --model-path
    if model_path is not None:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"Model not found: {p}")
        model, num_cls = _detect_and_load(p)
        print(f"Loaded model ({num_cls}-class): {p}")
        return model, str(p)

    # 2. Default 4-class checkpoint
    if MODEL_PATH.is_file():
        model, num_cls = _detect_and_load(MODEL_PATH)
        print(f"Loaded model ({num_cls}-class): {MODEL_PATH}")
        return model, str(MODEL_PATH)

    # 3. Fallback to other checkpoints
    for p in FALLBACK_PATHS:
        if p.is_file():
            print(f"[WARNING] {MODEL_PATH.name} not found. Using fallback: {p}")
            model, num_cls = _detect_and_load(p)
            return model, str(p)

    raise FileNotFoundError(
        f"No segmentation model found. Please run:\n"
        f"  python new_feature/train.py\n"
        f"or ensure {MODEL_PATH} exists."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = PROJECT_ROOT / "outputs" / "new_feature" / "evaluation" / timestamp
    tmp_dir   = out_dir / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load segmentation model ───────────────────────────────────────────────
    seg_model, model_src = _load_model(device, model_path=args.model_path)
    seg_model.eval()
    num_seg_classes = getattr(seg_model, "num_classes", 4)

    # ── Select dataset split ─────────────────────────────────────────────────
    if args.split == "test":
        img_dir      = IMG_TEST
        lbl_dir      = LBL_TEST
        cities_filter = TEST_CITIES
        split_label  = f"test ({', '.join(TEST_CITIES)})"
    else:
        img_dir      = IMG_VAL
        lbl_dir      = LBL_VAL
        cities_filter = None
        split_label  = "val"

    dataset = Cityscapes4Class(img_dir, lbl_dir, augment=False, cities=cities_filter)
    n_total = min(args.num_frames, len(dataset))
    print(f"Split   : {split_label}")
    print(f"Evaluating on {n_total} frames ({split_label} set has {len(dataset)} total)")

    # ── Which configurations to run ──────────────────────────────────────────
    run_table2 = not args.table4_only
    run_table4 = not args.table2_only

    configs_to_run: Dict[str, List[Config]] = {}
    if run_table2:
        configs_to_run["table2"] = TABLE2_CONFIGS
    if run_table4:
        configs_to_run["table4"] = TABLE4_CONFIGS

    all_configs: List[Config] = []
    seen: set = set()
    for cfgs in configs_to_run.values():
        for c in cfgs:
            if c.label not in seen:
                all_configs.append(c)
                seen.add(c.label)

    # ── Per-frame accumulators ─────────────────────────────────────────────
    # {config_label: {metric: [values]}}
    frame_results: Dict[str, Dict[str, List[float]]] = {
        c.label: {"psnr": [], "ssim": [], "sa_psnr": [], "sa_ssim": [],
                  "bitrate_roi": [], "bitrate_non": [],
                  "miou": [], "iiou": []}
        for c in all_configs
    }

    per_frame_rows: List[dict] = []

    # ── Main loop ────────────────────────────────────────────────────────────
    print("\nRunning evaluation…")
    for frame_idx in tqdm(range(n_total), desc="Frames"):
        img_t, lbl_gt, img_path = dataset[frame_idx]

        # Load original frame as RGB numpy array
        orig_rgb = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = orig_rgb.shape[:2]
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

        # Ground-truth 4-class label (uint8 at full resolution)
        lbl_np = lbl_gt.numpy().astype(np.uint8)
        if lbl_np.shape != (orig_h, orig_w):
            lbl_np_full = cv2.resize(lbl_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            lbl_np_full = lbl_np

        # Segment original frame → 4-class mask
        seg_pred = _segment(seg_model, device, orig_rgb)

        # Binary ROI mask (class 0 = ROI)
        roi_mask = _get_roi_mask(seg_pred)  # uint8 [H, W], values 0 or 1
        roi_bool  = roi_mask.astype(bool)

        # Per-configuration metrics
        frame_work_dir = tmp_dir / f"frame_{frame_idx:04d}"
        frame_work_dir.mkdir(exist_ok=True)

        for cfg in all_configs:
            try:
                if cfg.is_sac:
                    # SAC: encode ROI and non-ROI separately
                    dec_frames, roi_bytes, non_bytes = compress_sac(
                        [orig_bgr], [roi_mask],
                        codec=cfg.codec, crf_roi=cfg.crf_roi, crf_non=cfg.crf_non,
                        work_dir=str(frame_work_dir),
                        tag=f"f{frame_idx:04d}",
                        preset=args.preset,
                    )
                else:
                    # Traditional: encode whole frame at uniform CRF
                    dec_frames, total_bytes = compress_traditional(
                        [orig_bgr],
                        codec=cfg.codec, crf=cfg.crf_roi,
                        work_dir=str(frame_work_dir),
                        tag=f"f{frame_idx:04d}",
                        preset=args.preset,
                    )
                    roi_bytes  = total_bytes // 2
                    non_bytes  = total_bytes - roi_bytes

                if not dec_frames:
                    continue
                dec_bgr = dec_frames[0]
                dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)

                # Image quality metrics
                m = compute_all_compression_metrics(
                    orig_rgb, dec_rgb, roi_bool,
                    cfg.crf_roi, cfg.crf_non,
                )

                # Segmentation metrics (run seg model on decompressed frame)
                miou_val = iiou_val = 0.0
                if run_table4 and cfg in TABLE4_CONFIGS:
                    seg_dec = _segment(seg_model, device, dec_rgb)
                    if num_seg_classes == 4:
                        miou_val = compute_miou(seg_dec, lbl_np_full, num_classes=4)
                        iiou_val = compute_iiou(seg_dec, lbl_np_full)
                    else:
                        # Legacy 2-class model: compute binary IoU
                        roi_pred = (seg_dec == 0).astype(np.uint8)
                        roi_gt   = (lbl_np_full == 0).astype(np.uint8)
                        inter = int((roi_pred & roi_gt).sum())
                        union = int((roi_pred | roi_gt).sum())
                        iiou_val = inter / union if union > 0 else 0.0
                        miou_val = iiou_val

                fr = frame_results[cfg.label]
                fr["psnr"].append(m["psnr"])
                fr["ssim"].append(m["ssim"])
                fr["sa_psnr"].append(m["sa_psnr"])
                fr["sa_ssim"].append(m["sa_ssim"])
                fr["bitrate_roi"].append(roi_bytes)
                fr["bitrate_non"].append(non_bytes)
                fr["miou"].append(miou_val)
                fr["iiou"].append(iiou_val)

                per_frame_rows.append({
                    "frame_idx": frame_idx,
                    "method":   cfg.method,
                    "codec":    cfg.codec,
                    "crf_roi":  cfg.crf_roi,
                    "crf_non":  cfg.crf_non,
                    "psnr":     m["psnr"],
                    "ssim":     m["ssim"],
                    "sa_psnr":  m["sa_psnr"],
                    "sa_ssim":  m["sa_ssim"],
                    "roi_bytes": roi_bytes,
                    "non_bytes": non_bytes,
                    "miou":     miou_val,
                    "iiou":     iiou_val,
                })

            except Exception as exc:
                tqdm.write(f"  [skip] frame={frame_idx} {cfg.label}: {exc}")

        # Clean up per-frame tmp
        shutil.rmtree(str(frame_work_dir), ignore_errors=True)

    # Clean up global tmp
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # ── Aggregate results ────────────────────────────────────────────────────
    summary: List[dict] = []
    for cfg in all_configs:
        fr = frame_results[cfg.label]
        if not fr["psnr"]:
            continue
        row = {
            "method":   cfg.method,
            "codec":    cfg.codec,
            "crf_roi":  cfg.crf_roi,
            "crf_non":  cfg.crf_non,
            "psnr":     np.mean(fr["psnr"]),
            "ssim":     np.mean(fr["ssim"]),
            "sa_psnr":  np.mean(fr["sa_psnr"]),
            "sa_ssim":  np.mean(fr["sa_ssim"]),
            "miou_pct": np.mean(fr["miou"]) * 100 if fr["miou"] else 0.0,
            "iiou_pct": np.mean(fr["iiou"]) * 100 if fr["iiou"] else 0.0,
            "n_frames": len(fr["psnr"]),
        }
        summary.append(row)

    # ── Save per-frame CSV ────────────────────────────────────────────────────
    if per_frame_rows:
        pf_path = out_dir / "per_frame_metrics.csv"
        keys = list(per_frame_rows[0].keys())
        with open(pf_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_frame_rows)
        print(f"\nPer-frame CSV : {pf_path}")

    # ── Save Table II CSV ─────────────────────────────────────────────────────
    if run_table2:
        t2_rows = [r for r in summary if any(
            r["method"] == c.method and r["crf_roi"] == c.crf_roi and r["crf_non"] == c.crf_non
            for c in TABLE2_CONFIGS
        )]
        t2_path = out_dir / "table2_cityscapes.csv"
        _save_table2(t2_rows, t2_path)
        print(f"Table II CSV  : {t2_path}")

    # ── Save Table IV CSV ─────────────────────────────────────────────────────
    if run_table4:
        t4_rows = [r for r in summary if any(
            r["method"] == c.method and r["crf_roi"] == c.crf_roi and r["crf_non"] == c.crf_non
            for c in TABLE4_CONFIGS
        )]
        t4_path = out_dir / "table4_segmentation.csv"
        _save_table4(t4_rows, t4_path)
        print(f"Table IV CSV  : {t4_path}")

    # ── Save JSON + text summary ──────────────────────────────────────────────
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    txt_path = out_dir / "results_summary.txt"
    txt_path.write_text(_format_summary(summary, n_total, model_src))

    print(f"\nSummary text  : {txt_path}")
    print(f"All outputs   : {out_dir}")

    # ── Print tables to stdout ────────────────────────────────────────────────
    print("\n" + _format_summary(summary, n_total, model_src))


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_table2(rows: List[dict], path: Path) -> None:
    fields = ["method", "crf_roi", "crf_non", "psnr", "ssim", "sa_psnr", "sa_ssim", "n_frames"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _save_table4(rows: List[dict], path: Path) -> None:
    fields = ["method", "crf_roi", "crf_non", "sa_psnr", "sa_ssim", "miou_pct", "iiou_pct", "n_frames"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _format_summary(summary: List[dict], n_frames: int, model_src: str) -> str:
    lines = [
        "=" * 80,
        "SAC Paper Reproduction Results",
        f"Frames evaluated : {n_frames}",
        f"Segmentation model: {model_src}",
        "=" * 80,
        "",
        "TABLE II — Image Quality Metrics (Cityscapes)",
        "-" * 80,
        f"{'Method':<12} {'C_roi':>5} {'C_non':>5}  {'PSNR(dB)':>9} {'SSIM':>6} "
        f"{'SA-PSNR(dB)':>12} {'SA-SSIM':>8}",
        "-" * 80,
    ]
    # Group: X264 first, then X265
    for codec_tag in ("264", "265"):
        for r in summary:
            if codec_tag not in r["codec"]:
                continue
            lines.append(
                f"{r['method']:<12} {r['crf_roi']:>5} {r['crf_non']:>5}"
                f"  {r['psnr']:>9.2f} {r['ssim']:>6.3f}"
                f" {r['sa_psnr']:>12.2f} {r['sa_ssim']:>8.3f}"
            )
        lines.append("")

    lines += [
        "TABLE II (paper reference — Cityscapes):",
        "-" * 80,
        f"{'Method':<12} {'C_roi':>5} {'C_non':>5}  {'PSNR(dB)':>9} {'SSIM':>6} "
        f"{'SA-PSNR(dB)':>12} {'SA-SSIM':>8}",
        "-" * 80,
    ]
    paper_t2 = [
        ("H.264",   18, 18, 46.76, 0.98, 46.76, 0.98),
        ("SA-X264", 18, 23, 45.99, 0.98, 48.79, 0.99),
        ("H.264",   23, 23, 45.18, 0.98, 45.18, 0.98),
        ("SA-X264", 18, 27, 43.94, 0.98, 48.05, 0.99),
        ("H.264",   27, 27, 43.69, 0.98, 43.70, 0.98),
        ("SA-X264", 23, 27, 44.33, 0.98, 47.07, 0.99),
        ("H.265",   23, 23, 45.76, 0.98, 45.76, 0.98),
        ("SA-X265", 23, 28, 43.98, 0.99, 47.76, 0.99),
        ("H.265",   28, 28, 43.80, 0.98, 43.80, 0.98),
        ("SA-X265", 23, 32, 43.77, 0.98, 47.01, 0.99),
        ("H.265",   32, 32, 42.01, 0.97, 42.01, 0.97),
        ("SA-X265", 28, 32, 43.97, 0.98, 45.20, 0.98),
    ]
    for m, cr, cn, p, s, sp, ss in paper_t2:
        lines.append(f"{m:<12} {cr:>5} {cn:>5}  {p:>9.2f} {s:>6.3f} {sp:>12.2f} {ss:>8.3f}")
    lines.append("")

    lines += [
        "TABLE IV — Segmentation Quality (mIoU, iIoU) on Decompressed Frames",
        "-" * 80,
        f"{'Method':<12} {'C_roi':>5} {'C_non':>5}  "
        f"{'SA-PSNR':>9} {'SA-SSIM':>8} {'mIoU(%)':>9} {'iIoU(%)':>9}",
        "-" * 80,
    ]
    t4_labels = {(c.method, c.crf_roi, c.crf_non) for c in TABLE4_CONFIGS}
    for r in summary:
        if (r["method"], r["crf_roi"], r["crf_non"]) in t4_labels:
            lines.append(
                f"{r['method']:<12} {r['crf_roi']:>5} {r['crf_non']:>5}"
                f"  {r['sa_psnr']:>9.2f} {r['sa_ssim']:>8.3f}"
                f" {r['miou_pct']:>9.2f} {r['iiou_pct']:>9.2f}"
            )
    lines += [
        "",
        "TABLE IV (paper reference):",
        "-" * 80,
        f"{'Method':<12} {'C_roi':>5} {'C_non':>5}  "
        f"{'SA-PSNR':>9} {'SA-SSIM':>8} {'mIoU(%)':>9} {'iIoU(%)':>9}",
        "-" * 80,
        f"{'H.264':<12} {23:>5} {23:>5}  {46.76:>9.2f} {0.98:>8.3f} {87.86:>9.2f} {92.45:>9.2f}",
        f"{'SA-X264':<12} {18:>5} {27:>5}  {48.05:>9.2f} {0.99:>8.3f} {90.56:>9.2f} {92.45:>9.2f}",
        f"{'H.265':<12} {28:>5} {28:>5}  {43.80:>9.2f} {0.98:>8.3f} {85.71:>9.2f} {91.43:>9.2f}",
        f"{'SA-X265':<12} {23:>5} {32:>5}  {47.01:>9.2f} {0.99:>8.3f} {87.48:>9.2f} {92.04:>9.2f}",
        "",
        "=" * 80,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAC (Wang et al. 2023) — reproduce Tables II & IV")
    p.add_argument("--num-frames",  type=int, default=30,
                   help="Number of frames to evaluate (default: 30, paper: 500)")
    p.add_argument("--preset",      default="medium",
                   choices=["ultrafast", "superfast", "veryfast", "faster",
                             "fast", "medium", "slow", "slower", "veryslow"],
                   help="FFmpeg encoding preset (default: medium)")
    p.add_argument("--table2-only", action="store_true",
                   help="Only run Table II configurations (no segmentation eval)")
    p.add_argument("--table4-only", action="store_true",
                   help="Only run Table IV configurations (with segmentation eval)")
    p.add_argument("--split",       choices=["val", "test"], default="val",
                   help="Dataset split: 'val' (frankfurt/lindau/munster) or "
                        "'test' (strasbourg/ulm, default model: best_pidnet_l_4class.pth)")
    p.add_argument("--model-path",  type=str, default=None,
                   help="Path to segmentation model checkpoint (default: auto-detect)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse())
