"""
Bitrate–Accuracy RD curves for SAC vs traditional codecs.

Implements the full Semantic-Aware Compression pipeline of
Wang et al. (IEEE TIV 2023) and produces an RD plot showing how
SAC (SA-X264 / SA-X265) outperforms traditional H.264 / H.265 baselines
on Cityscapes.

Pipeline (per operating point, per codec):
  Step 1: ROI mask = (seg_pred == 0)            (broad ROI from seg model)
  Step 2: macroblock-align mask to 16x16 grid → split frame into img_roi / img_non
  Step 3: encode each stream with FFmpeg
            - SAC      :  roi @ CRF_roi  +  non-ROI @ CRF_non
            - Traditional: full frame @ single CRF (= label of operating point)
  Step 4: decode each stream → for SAC merge via addition
  Step 5: compute PSNR, SSIM, SA-PSNR, SA-SSIM (Eq. 13/14),
                  mIoU (Eq. 16) and iIoU (Eq. 17) of seg model on decoded frame.

Operating points (label = traditional baseline QP; SAC pair averages to it):
  ("22", 19, 26)    →   H.26x @ QP22   vs   SA-X26x @ {19, 26}
  ("27", 24, 31)    →   H.26x @ QP27   vs   SA-X26x @ {24, 31}
  ("32", 29, 36)    →   H.26x @ QP32   vs   SA-X26x @ {29, 36}
  ("37", 34, 41)    →   H.26x @ QP37   vs   SA-X26x @ {34, 41}

Outputs (outputs/new_feature/bitrate_accuracy/<timestamp>/):
  bitrate_accuracy.csv      — summary per (op, method)
  per_frame.csv             — raw per-frame numbers
  bitrate_vs_miou.png       — RD curve (mIoU as accuracy)
  bitrate_vs_iiou.png       — RD curve (iIoU, ROI-only)
  bitrate_vs_sa_psnr.png    — RD curve (SA-PSNR as quality)
  summary.json              — run config + results

Usage (from scripts/ dir, conda sac env):
  conda activate sac
  cd /home/huy/sac_project/scripts
  python new_feature/bitrate_accuracy_rd.py --num-frames 30 --fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
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

from new_feature.compression import (
    compress_sac,
    compress_traditional,
    macroblock_align_filter,
)
from new_feature.dataset import Cityscapes4Class
from new_feature.metrics import (
    compute_iiou,
    compute_miou,
    compute_psnr,
    compute_sa_psnr,
    compute_sa_ssim,
    compute_ssim,
)

PROJECT_ROOT = _HERE.parent.parent
MODEL_PATH_DEFAULT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
FALLBACK_MODELS = [
    PROJECT_ROOT / "models" / "best_pidnet_l.pth",
    PROJECT_ROOT / "models" / "best_ccnet.pth",
    PROJECT_ROOT / "models" / "best_pidnet.pth",
]
IMG_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LBL_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "val"
IMG_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LBL_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"
# Cityscapes test split: only these 2 cities have 4-class ground-truth labels.
TEST_CITIES = ["strasbourg", "ulm"]


# ─────────────────────────────────────────────────────────────────────────────
# Operating points
# ─────────────────────────────────────────────────────────────────────────────

class OperatingPoint(NamedTuple):
    label: str   # traditional baseline QP (string for plotting)
    crf_roi: int
    crf_non: int

    @property
    def crf_trad(self) -> int:
        return int(self.label)


OPERATING_POINTS: List[OperatingPoint] = [
    OperatingPoint("22", 19, 25),
    OperatingPoint("25", 22, 28),
    OperatingPoint("28", 25, 31),
    OperatingPoint("31", 28, 34),
    OperatingPoint("34", 31, 37),
]

# (display_name, ffmpeg_codec, is_sac)
METHODS: List[Tuple[str, str, bool]] = [
    ("H.264",   "libx264", False),
    ("SA-X264", "libx264", True),
    ("H.265",   "libx265", False),
    ("SA-X265", "libx265", True),
]


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation helper
# ─────────────────────────────────────────────────────────────────────────────

SEG_SIZE: Tuple[int, int] = (512, 1024)   # (H, W)
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _segment(model: torch.nn.Module, device: torch.device, frame_rgb: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    img_pil = Image.fromarray(frame_rgb).resize((SEG_SIZE[1], SEG_SIZE[0]), Image.BILINEAR)
    tensor = _TRANSFORM(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1)[0]
    pred_full = F.interpolate(
        pred.float().unsqueeze(0).unsqueeze(0),
        size=(H, W), mode="nearest",
    )[0, 0].byte().cpu().numpy()
    return pred_full


def _load_segmentation_model(
    device: torch.device, model_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, str, int]:
    """Auto-detect PIDNet / new CCNet4Class / legacy CCNet architectures."""
    candidates: List[Path] = []
    if model_path:
        candidates.append(Path(model_path))
    else:
        candidates = [MODEL_PATH_DEFAULT] + FALLBACK_MODELS

    for p in candidates:
        if not p.is_file():
            continue

        ckpt  = torch.load(str(p), map_location=device, weights_only=False)
        meta  = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        num_cls    = int(meta.get("num_classes", 4))
        model_name = meta.get("model_name", "")

        keys      = list(state.keys())
        is_legacy = any(k.startswith("backbone.") for k in keys)
        is_new    = any(k.startswith("stem.") or k.startswith("layer1.") for k in keys)

        if model_name.startswith("pidnet") or (
            not is_legacy and not is_new and any(k.startswith("conv1.") for k in keys)
        ):
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(p), device, num_classes=num_cls)
        elif is_new:
            from new_feature.ccnet_4class import CCNet4Class
            model = CCNet4Class(num_classes=num_cls, pretrained=False).to(device)
            model.load_state_dict(state)
        elif is_legacy:
            from train_segmentation import build_segmentation_model
            model = build_segmentation_model("ccnet", num_classes=num_cls,
                                             backbone_weights=None).to(device)
            model.load_state_dict(state)
        else:
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(p), device, num_classes=num_cls)

        model.num_classes = num_cls
        model.eval()
        print(f"Loaded segmentation model ({num_cls}-class): {p}")
        return model, str(p), num_cls

    raise FileNotFoundError(
        "No segmentation checkpoint found. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Encoding wrapper (returns decoded frames + bitrate)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_and_decode(
    method:     str,
    codec:      str,
    is_sac:     bool,
    op:         OperatingPoint,
    frames_bgr: List[np.ndarray],
    roi_masks:  List[np.ndarray],
    work_dir:   Path,
    fps:        int,
    preset:     str,
) -> Tuple[List[np.ndarray], int, int, int, int, int]:
    """
    Returns: (decoded_bgr, total_bytes, roi_bytes, non_bytes, crf_roi_used, crf_non_used)
    """
    tag = f"op{op.label}_{method.replace('.', '').replace('-', '')}"
    if is_sac:
        dec_bgr, roi_bytes, non_bytes = compress_sac(
            frames_bgr, roi_masks,
            codec=codec, crf_roi=op.crf_roi, crf_non=op.crf_non,
            work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
        )
        total_bytes = roi_bytes + non_bytes
        return dec_bgr, total_bytes, roi_bytes, non_bytes, op.crf_roi, op.crf_non

    dec_bgr, total_bytes = compress_traditional(
        frames_bgr,
        codec=codec, crf=op.crf_trad,
        work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
    )
    roi_bytes = total_bytes // 2
    non_bytes = total_bytes - roi_bytes
    return dec_bgr, total_bytes, roi_bytes, non_bytes, op.crf_trad, op.crf_trad


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_ROOT / "outputs" / "new_feature" / "bitrate_accuracy" / timestamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "tmp"
    work_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {out_dir}")

    # 1) segmentation model
    seg_model, model_src, num_cls = _load_segmentation_model(device, model_path=args.model_path)

    # 2) load frames + GT labels, precompute ROI masks
    if args.split == "test":
        img_dir, lbl_dir = IMG_TEST, LBL_TEST
        cities_filter    = TEST_CITIES
        split_desc       = f"test ({', '.join(TEST_CITIES)})"
    else:
        img_dir, lbl_dir = IMG_VAL, LBL_VAL
        cities_filter    = None
        split_desc       = "val"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image root not found: {img_dir}")

    dataset = Cityscapes4Class(img_dir, lbl_dir, augment=False, cities=cities_filter)
    n_total = min(args.num_frames, len(dataset))
    print(f"Split  : {split_desc}")
    print(f"Frames : {n_total} / {len(dataset)}")

    frames_bgr: List[np.ndarray] = []
    frames_rgb: List[np.ndarray] = []
    gt_labels:  List[np.ndarray] = []
    roi_masks:  List[np.ndarray] = []

    print("\nStep 1-2: load frames, segment, build ROI masks (macroblock-aligned)...")
    for i in tqdm(range(n_total)):
        _, lbl_gt, img_path = dataset[i]
        orig_rgb = np.array(Image.open(img_path).convert("RGB"))
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        H, W = orig_rgb.shape[:2]

        lbl_np = lbl_gt.numpy().astype(np.uint8)
        if lbl_np.shape != (H, W):
            lbl_np = cv2.resize(lbl_np, (W, H), interpolation=cv2.INTER_NEAREST)

        seg_pred = _segment(seg_model, device, orig_rgb)
        roi_mask = macroblock_align_filter((seg_pred == 0).astype(np.uint8))

        frames_bgr.append(orig_bgr)
        frames_rgb.append(orig_rgb)
        gt_labels.append(lbl_np)
        roi_masks.append(roi_mask)

    roi_ratios = [m.mean() * 100 for m in roi_masks]
    print(f"  ROI ratio : mean={np.mean(roi_ratios):.1f}%  "
          f"min={np.min(roi_ratios):.1f}%  max={np.max(roi_ratios):.1f}%")

    duration_sec = n_total / float(args.fps)

    # 3) for each (op, method) → encode video, decode, metrics
    summary_rows:   List[dict] = []
    per_frame_rows: List[dict] = []

    print("\nStep 3-5: encode + decode + metrics")
    total_jobs = len(OPERATING_POINTS) * len(METHODS)
    job_idx = 0
    for op in OPERATING_POINTS:
        for method, codec, is_sac in METHODS:
            job_idx += 1
            cfg_desc = (
                f"crf_roi={op.crf_roi}, crf_non={op.crf_non}"
                if is_sac else f"crf={op.crf_trad}"
            )
            print(f"\n[{job_idx}/{total_jobs}] op={op.label}  {method:<8} ({cfg_desc})")

            dec_bgr, total_bytes, roi_bytes, non_bytes, crf_roi_used, crf_non_used = \
                _encode_and_decode(
                    method, codec, is_sac, op,
                    frames_bgr, roi_masks,
                    work_dir=work_dir, fps=args.fps, preset=args.preset,
                )

            bitrate_kbps = (total_bytes * 8.0 / duration_sec) / 1000.0

            psnrs:   List[float] = []
            ssims:   List[float] = []
            sapsnrs: List[float] = []
            sassims: List[float] = []
            mious:   List[float] = []
            iious:   List[float] = []

            n_dec = min(len(dec_bgr), n_total)
            if n_dec < n_total:
                print(f"  WARNING: decoder returned {n_dec} frames (expected {n_total})")

            for i in range(n_dec):
                orig_rgb = frames_rgb[i]
                dec_rgb  = cv2.cvtColor(dec_bgr[i], cv2.COLOR_BGR2RGB)
                roi_bool = roi_masks[i].astype(bool)

                psnrs.append(compute_psnr(orig_rgb, dec_rgb))
                ssims.append(compute_ssim(orig_rgb, dec_rgb))
                sapsnrs.append(compute_sa_psnr(orig_rgb, dec_rgb, roi_bool,
                                                crf_roi_used, crf_non_used))
                sassims.append(compute_sa_ssim(orig_rgb, dec_rgb, roi_bool,
                                                crf_roi_used, crf_non_used))

                seg_dec = _segment(seg_model, device, dec_rgb)
                if num_cls == 4:
                    miou_val = compute_miou(seg_dec, gt_labels[i], num_classes=4)
                    iiou_val = compute_iiou(seg_dec, gt_labels[i])
                else:
                    roi_pred = (seg_dec == 0).astype(np.uint8)
                    roi_gt   = (gt_labels[i] == 0).astype(np.uint8)
                    inter = int((roi_pred & roi_gt).sum())
                    union = int((roi_pred | roi_gt).sum())
                    iiou_val = inter / union if union > 0 else 0.0
                    miou_val = iiou_val
                mious.append(miou_val)
                iious.append(iiou_val)

                per_frame_rows.append({
                    "op_label": op.label, "method": method, "codec": codec, "is_sac": is_sac,
                    "frame_idx": i,
                    "crf_roi": crf_roi_used, "crf_non": crf_non_used,
                    "psnr":    psnrs[-1],    "ssim":    ssims[-1],
                    "sa_psnr": sapsnrs[-1],  "sa_ssim": sassims[-1],
                    "miou":    miou_val,     "iiou":    iiou_val,
                })

            row = {
                "op_label": op.label, "method": method, "codec": codec, "is_sac": is_sac,
                "crf_roi": crf_roi_used, "crf_non": crf_non_used,
                "total_bytes":  total_bytes,
                "roi_bytes":    roi_bytes,
                "non_bytes":    non_bytes,
                "bitrate_kbps": bitrate_kbps,
                "psnr":     float(np.mean(psnrs))    if psnrs    else 0.0,
                "ssim":     float(np.mean(ssims))    if ssims    else 0.0,
                "sa_psnr":  float(np.mean(sapsnrs))  if sapsnrs  else 0.0,
                "sa_ssim":  float(np.mean(sassims))  if sassims  else 0.0,
                "miou":     float(np.mean(mious))    if mious    else 0.0,
                "iiou":     float(np.mean(iious))    if iious    else 0.0,
                "miou_pct": float(np.mean(mious)) * 100 if mious else 0.0,
                "iiou_pct": float(np.mean(iious)) * 100 if iious else 0.0,
                "n_frames": n_dec,
            }
            summary_rows.append(row)

            print(f"  bitrate={bitrate_kbps:7.1f} kbps  "
                  f"PSNR={row['psnr']:5.2f}  SA-PSNR={row['sa_psnr']:5.2f}  "
                  f"mIoU={row['miou_pct']:5.2f}%  iIoU={row['iiou_pct']:5.2f}%")

    # 4) save CSVs
    summary_csv = out_dir / "bitrate_accuracy.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary CSV : {summary_csv}")

    if per_frame_rows:
        pf_csv = out_dir / "per_frame.csv"
        with open(pf_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
            w.writeheader()
            w.writerows(per_frame_rows)
        print(f"Per-frame CSV : {pf_csv}")

    # 5) plots
    _plot_rd(summary_rows, out_dir / "bitrate_vs_miou.png",
             metric_key="miou_pct", ylabel="mIoU (%)",
             title="Bitrate vs mIoU — SAC vs traditional codecs")
    _plot_rd(summary_rows, out_dir / "bitrate_vs_iiou.png",
             metric_key="iiou_pct", ylabel="iIoU (%) — ROI only",
             title="Bitrate vs iIoU — SAC vs traditional codecs")
    _plot_rd(summary_rows, out_dir / "bitrate_vs_sa_psnr.png",
             metric_key="sa_psnr", ylabel="SA-PSNR (dB)",
             title="Bitrate vs SA-PSNR — SAC vs traditional codecs")

    # 6) json summary
    (out_dir / "summary.json").write_text(json.dumps({
        "split":             args.split,
        "split_desc":        split_desc,
        "num_frames":        n_total,
        "fps":               args.fps,
        "preset":            args.preset,
        "model":             model_src,
        "num_classes":       num_cls,
        "operating_points":  [op._asdict() for op in OPERATING_POINTS],
        "results":           summary_rows,
    }, indent=2))

    shutil.rmtree(str(work_dir), ignore_errors=True)
    print(f"\nAll outputs : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_rd(rows: List[dict], out_png: Path, metric_key: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    style: Dict[str, Dict] = {
        "H.264":   {"color": "#1f77b4", "marker": "o", "ls": "-",  "name": "H.264"},
        "SA-X264": {"color": "#ff7f0e", "marker": "s", "ls": "--", "name": "SA-X264"},
        "H.265":   {"color": "#2ca02c", "marker": "D", "ls": "-",  "name": "H.265"},
        "SA-X265": {"color": "#d62728", "marker": "^", "ls": "--", "name": "SA-X265"},
    }
    order = ["H.264", "SA-X264", "H.265", "SA-X265"]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for method in order:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        method_rows.sort(key=lambda r: r["bitrate_kbps"])
        xs = [r["bitrate_kbps"] for r in method_rows]
        ys = [r[metric_key]    for r in method_rows]
        s = style[method]
        ax.plot(xs, ys, s["ls"],
                color=s["color"], marker=s["marker"], markersize=9,
                linewidth=2.3, label=s["name"])
        for r in method_rows:
            ax.annotate(
                f"QP{r['op_label']}",
                (r["bitrate_kbps"], r[metric_key]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=s["color"],
            )

    ax.set_xlabel("Bitrate (kbps)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Plot        : {out_png}")


# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bitrate-Accuracy RD curves: SAC vs traditional H.264/H.265",
    )
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="Cityscapes split: 'val' (frankfurt/lindau/munster) "
                        "or 'test' (strasbourg/ulm — only those have 4-class GT).")
    p.add_argument("--num-frames", type=int, default=20,
                   help="Number of frames to encode (default: 20).")
    p.add_argument("--fps", type=int, default=30,
                   help="Frame rate for bitrate computation (default: 30).")
    p.add_argument("--preset", default="medium",
                   choices=["ultrafast", "superfast", "veryfast", "faster",
                            "fast", "medium", "slow", "slower", "veryslow"],
                   help="FFmpeg preset (default: medium).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: outputs/new_feature/bitrate_accuracy/<timestamp>).")
    p.add_argument("--model-path", type=str, default=None,
                   help="Path to segmentation checkpoint (auto-detect when omitted).")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse())
