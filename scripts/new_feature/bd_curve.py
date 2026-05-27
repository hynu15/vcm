"""
BD-Rate / BD-iIoU curve cho SAC vs Traditional với các mức QP 16, 19, 22, 25, 28.

Björgaard Delta (BD-Rate):
  Tính diện tích dưới đường cong RD trong không gian (log-bitrate, quality),
  so sánh cặp (SAC vs Traditional) cho cả H.264 và H.265.

Operating points:
  Trad QP  →  SAC (crf_roi, crf_non)  [pattern: roi = QP-3, non = QP+4]
  22       →  (19, 26)
  25       →  (22, 29)
  28       →  (25, 32)
  31       →  (28, 35)
  34       →  (31, 38)

Outputs (outputs/new_feature/bd_curve/<timestamp>/):
  rd_iiou.png          — RD curve (iIoU)
  rd_miou.png          — RD curve (mIoU)
  rd_sa_psnr.png       — RD curve (SA-PSNR)
  bd_results.csv       — BD-Rate và BD-iIoU/mIoU per codec
  bitrate_accuracy.csv — raw operating point data
  summary.json

Chạy từ scripts/:
  conda run -n sac python3 new_feature/bd_curve.py --num-frames 30 --split test
  conda run -n sac python3 new_feature/bd_curve.py --num-frames 10 --split val --preset fast
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

PROJECT_ROOT       = _HERE.parent.parent
MODEL_PATH_DEFAULT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
FALLBACK_MODELS    = [
    PROJECT_ROOT / "models" / "best_pidnet_l.pth",
    PROJECT_ROOT / "models" / "best_ccnet.pth",
]
IMG_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LBL_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "val"
IMG_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LBL_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"
TEST_CITIES = ["strasbourg", "ulm"]


# ─────────────────────────────────────────────────────────────────────────────
# Operating points: traditional QP 16/19/22/25/28
# SAC pattern: crf_roi = QP - 3, crf_non = QP + 4
# ─────────────────────────────────────────────────────────────────────────────

class OP(NamedTuple):
    label: str
    crf_roi: int
    crf_non: int

    @property
    def crf_trad(self) -> int:
        return int(self.label)


OPERATING_POINTS: List[OP] = [
    OP("22", 19, 26),
    OP("25", 22, 29),
    OP("28", 25, 32),
    OP("31", 28, 35),
    OP("34", 31, 38),
]

METHODS: List[Tuple[str, str, bool]] = [
    ("H.264",   "libx264", False),
    ("SA-X264", "libx264", True),
    ("H.265",   "libx265", False),
    ("SA-X265", "libx265", True),
]

SEG_SIZE = (512, 1024)
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation helpers (reuse logic from bitrate_accuracy_rd.py)
# ─────────────────────────────────────────────────────────────────────────────

def _segment(model, device, frame_rgb: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    pil = Image.fromarray(frame_rgb).resize((SEG_SIZE[1], SEG_SIZE[0]), Image.BILINEAR)
    t = _TRANSFORM(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        if isinstance(out, (list, tuple)):
            out = out[1]
        pred = out.argmax(1)[0]
    pred_full = F.interpolate(
        pred.float().unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest",
    )[0, 0].byte().cpu().numpy()
    return pred_full


def _load_seg_model(device, model_path=None):
    candidates = [Path(model_path)] if model_path else [MODEL_PATH_DEFAULT] + FALLBACK_MODELS
    for p in candidates:
        if not p.is_file():
            continue
        ckpt  = torch.load(str(p), map_location=device, weights_only=False)
        meta  = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        num_cls    = int(meta.get("num_classes", 4))
        model_name = meta.get("model_name", "")
        keys = list(state.keys())
        is_new = any(k.startswith("stem.") or k.startswith("layer1.") for k in keys)

        if model_name.startswith("pidnet") or (
            not is_new and any(k.startswith("conv1.") for k in keys)
        ):
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(p), device, num_classes=num_cls)
        elif is_new:
            from new_feature.ccnet_4class import CCNet4Class
            model = CCNet4Class(num_classes=num_cls, pretrained=False).to(device)
            model.load_state_dict(state)
        else:
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(p), device, num_classes=num_cls)

        model.num_classes = num_cls
        model.eval()
        print(f"  Segmentation model ({num_cls}-class): {p.name}")
        return model, str(p), num_cls
    raise FileNotFoundError("No checkpoint found: " + ", ".join(str(c) for c in candidates))


# ─────────────────────────────────────────────────────────────────────────────
# Bjontegaard Delta  (BD-Rate, BD-metric)
# ─────────────────────────────────────────────────────────────────────────────

def _bd_rate(bitrates_ref: List[float], quality_ref: List[float],
             bitrates_test: List[float], quality_test: List[float]) -> float:
    """
    Bjontegaard Delta Rate (%).
    Negative = test saves bitrate vs ref at the same quality.
    Uses cubic polynomial fit in (log-bitrate, quality) space.
    """
    log_br_ref  = np.log(bitrates_ref)
    log_br_test = np.log(bitrates_test)

    # Overlapping quality range
    min_q = max(min(quality_ref), min(quality_test))
    max_q = min(max(quality_ref), max(quality_test))
    if max_q <= min_q:
        return float("nan")

    p_ref  = np.polyfit(quality_ref,  log_br_ref,  3)
    p_test = np.polyfit(quality_test, log_br_test, 3)

    int_ref  = np.polyval(np.polyint(p_ref),  [min_q, max_q])
    int_test = np.polyval(np.polyint(p_test), [min_q, max_q])

    avg_ref  = (int_ref[1]  - int_ref[0])  / (max_q - min_q)
    avg_test = (int_test[1] - int_test[0]) / (max_q - min_q)

    return (np.exp(avg_test - avg_ref) - 1.0) * 100.0


def _bd_quality(bitrates_ref: List[float], quality_ref: List[float],
                bitrates_test: List[float], quality_test: List[float]) -> float:
    """
    Bjontegaard Delta quality (absolute units, e.g. dB or % iIoU).
    Positive = test is better quality at the same bitrate.
    Uses cubic polynomial fit in (log-bitrate, quality) space.
    """
    log_br_ref  = np.log(bitrates_ref)
    log_br_test = np.log(bitrates_test)

    # Overlapping bitrate range
    min_log = max(min(log_br_ref), min(log_br_test))
    max_log = min(max(log_br_ref), max(log_br_test))
    if max_log <= min_log:
        return float("nan")

    p_ref  = np.polyfit(log_br_ref,  quality_ref,  3)
    p_test = np.polyfit(log_br_test, quality_test, 3)

    int_ref  = np.polyval(np.polyint(p_ref),  [min_log, max_log])
    int_test = np.polyval(np.polyint(p_test), [min_log, max_log])

    avg_ref  = (int_ref[1]  - int_ref[0])  / (max_log - min_log)
    avg_test = (int_test[1] - int_test[0]) / (max_log - min_log)

    return avg_test - avg_ref


# ─────────────────────────────────────────────────────────────────────────────
# Encode / decode
# ─────────────────────────────────────────────────────────────────────────────

def _encode_decode(method, codec, is_sac, op: OP,
                   frames_bgr, roi_masks, work_dir, fps, preset):
    tag = f"op{op.label}_{method.replace('.','').replace('-','')}"
    if is_sac:
        dec, roi_b, non_b = compress_sac(
            frames_bgr, roi_masks,
            codec=codec, crf_roi=op.crf_roi, crf_non=op.crf_non,
            work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
        )
        return dec, roi_b + non_b, roi_b, non_b, op.crf_roi, op.crf_non
    dec, total_b = compress_traditional(
        frames_bgr,
        codec=codec, crf=op.crf_trad,
        work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
    )
    return dec, total_b, total_b // 2, total_b - total_b // 2, op.crf_trad, op.crf_trad


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "H.264":   {"color": "#1f77b4", "marker": "o",  "ls": "-",  "label": "H.264"},
    "SA-X264": {"color": "#ff7f0e", "marker": "s",  "ls": "--", "label": "SA-X264"},
    "H.265":   {"color": "#2ca02c", "marker": "D",  "ls": "-",  "label": "H.265"},
    "SA-X265": {"color": "#d62728", "marker": "^",  "ls": "--", "label": "SA-X265"},
}
ORDER = ["H.264", "SA-X264", "H.265", "SA-X265"]


def _plot_rd_with_bd(rows: List[dict], out_png: Path,
                     metric_key: str, ylabel: str, title: str,
                     bd_264: float, bd_265: float, bd_label: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

    for method in ORDER:
        mrs = sorted([r for r in rows if r["method"] == method],
                     key=lambda r: r["bitrate_kbps"])
        if not mrs:
            continue
        xs = [r["bitrate_kbps"] for r in mrs]
        ys = [r[metric_key]     for r in mrs]
        s  = STYLE[method]
        ax.plot(xs, ys, s["ls"], color=s["color"], marker=s["marker"],
                markersize=8, linewidth=2.2, label=s["label"])
        for r in mrs:
            ax.annotate(
                f"QP{r['op_label']}",
                (r["bitrate_kbps"], r[metric_key]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=7.5, color=s["color"],
            )

    # BD annotation box
    bd_text = (
        f"BD-Rate (SA-X264 vs H.264) = {bd_264:+.2f}%\n"
        f"BD-Rate (SA-X265 vs H.265) = {bd_265:+.2f}%\n"
        f"  (BD-{bd_label}: "
        f"{_bd_quality_str(rows, metric_key, 'SA-X264', 'H.264'):+.3f} / "
        f"{_bd_quality_str(rows, metric_key, 'SA-X265', 'H.265'):+.3f})"
    )
    ax.text(0.02, 0.03, bd_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85))

    ax.set_xlabel("Bitrate (kbps)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_png}")


def _bd_quality_str(rows, metric_key, test_method, ref_method):
    ref  = sorted([r for r in rows if r["method"] == ref_method],  key=lambda r: r["bitrate_kbps"])
    test = sorted([r for r in rows if r["method"] == test_method], key=lambda r: r["bitrate_kbps"])
    if len(ref) < 4 or len(test) < 4:
        return float("nan")
    return _bd_quality(
        [r["bitrate_kbps"] for r in ref],  [r[metric_key] for r in ref],
        [r["bitrate_kbps"] for r in test], [r[metric_key] for r in test],
    )


def _bd_rate_pair(rows, metric_key, test_method, ref_method):
    ref  = sorted([r for r in rows if r["method"] == ref_method],  key=lambda r: r[metric_key])
    test = sorted([r for r in rows if r["method"] == test_method], key=lambda r: r[metric_key])
    if len(ref) < 4 or len(test) < 4:
        return float("nan")
    return _bd_rate(
        [r["bitrate_kbps"] for r in ref],  [r[metric_key] for r in ref],
        [r["bitrate_kbps"] for r in test], [r[metric_key] for r in test],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_ROOT / "outputs" / "new_feature" / "bd_curve" / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "tmp"
    work_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {out_dir}")

    seg_model, model_src, num_cls = _load_seg_model(device, args.model_path)

    # Dataset
    if args.split == "test":
        img_dir, lbl_dir = IMG_TEST, LBL_TEST
        cities, split_desc = TEST_CITIES, "test (strasbourg, ulm)"
    else:
        img_dir, lbl_dir = IMG_VAL, LBL_VAL
        cities, split_desc = None, "val"

    dataset = Cityscapes4Class(img_dir, lbl_dir, augment=False, cities=cities)
    n = min(args.num_frames, len(dataset))
    print(f"Split  : {split_desc}")
    print(f"Frames : {n} / {len(dataset)}")
    print(f"OPs    : QP {[op.label for op in OPERATING_POINTS]}\n")

    frames_bgr, frames_rgb, gt_labels, roi_masks = [], [], [], []

    print("Step 1-2: load frames + segment + build ROI masks ...")
    for i in tqdm(range(n)):
        _, lbl_gt, img_path = dataset[i]
        rgb = np.array(Image.open(img_path).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = rgb.shape[:2]
        lbl = lbl_gt.numpy().astype(np.uint8)
        if lbl.shape != (H, W):
            lbl = cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)
        pred = _segment(seg_model, device, rgb)
        mask = macroblock_align_filter((pred == 0).astype(np.uint8))
        frames_bgr.append(bgr);  frames_rgb.append(rgb)
        gt_labels.append(lbl);   roi_masks.append(mask)

    roi_ratios = [m.mean() * 100 for m in roi_masks]
    print(f"  ROI ratio: mean={np.mean(roi_ratios):.1f}%  "
          f"min={np.min(roi_ratios):.1f}%  max={np.max(roi_ratios):.1f}%")

    duration_sec = n / float(args.fps)
    summary_rows: List[dict] = []

    print("\nStep 3-5: encode + decode + metrics ...")
    total_jobs = len(OPERATING_POINTS) * len(METHODS)
    job = 0
    for op in OPERATING_POINTS:
        for method, codec, is_sac in METHODS:
            job += 1
            cfg = (f"crf_roi={op.crf_roi}, crf_non={op.crf_non}"
                   if is_sac else f"crf={op.crf_trad}")
            print(f"\n[{job}/{total_jobs}] QP{op.label}  {method:<8}  ({cfg})")

            dec_bgr, total_b, roi_b, non_b, crf_r, crf_n = _encode_decode(
                method, codec, is_sac, op,
                frames_bgr, roi_masks, work_dir, args.fps, args.preset,
            )

            bitrate = (total_b * 8.0 / duration_sec) / 1000.0
            psnrs, ssims, sapsnrs, sassims, mious, iious = [], [], [], [], [], []

            n_dec = min(len(dec_bgr), n)
            for i in range(n_dec):
                orig = frames_rgb[i]
                dec  = cv2.cvtColor(dec_bgr[i], cv2.COLOR_BGR2RGB)
                roi  = roi_masks[i].astype(bool)
                psnrs.append(compute_psnr(orig, dec))
                ssims.append(compute_ssim(orig, dec))
                sapsnrs.append(compute_sa_psnr(orig, dec, roi, crf_r, crf_n))
                sassims.append(compute_sa_ssim(orig, dec, roi, crf_r, crf_n))
                seg  = _segment(seg_model, device, dec)
                if num_cls == 4:
                    mious.append(compute_miou(seg, gt_labels[i], num_classes=4))
                    iious.append(compute_iiou(seg, gt_labels[i]))
                else:
                    rp = (seg == 0).astype(np.uint8)
                    rg = (gt_labels[i] == 0).astype(np.uint8)
                    u  = int((rp | rg).sum())
                    v  = int((rp & rg).sum()) / u if u > 0 else 0.0
                    mious.append(v); iious.append(v)

            row = dict(
                op_label=op.label, method=method, codec=codec, is_sac=is_sac,
                crf_roi=crf_r, crf_non=crf_n,
                total_bytes=total_b, roi_bytes=roi_b, non_bytes=non_b,
                bitrate_kbps=bitrate,
                psnr=float(np.mean(psnrs)),      ssim=float(np.mean(ssims)),
                sa_psnr=float(np.mean(sapsnrs)), sa_ssim=float(np.mean(sassims)),
                miou=float(np.mean(mious)),       iiou=float(np.mean(iious)),
                miou_pct=float(np.mean(mious)) * 100,
                iiou_pct=float(np.mean(iious)) * 100,
                n_frames=n_dec,
            )
            summary_rows.append(row)
            print(f"  bitrate={bitrate:7.1f} kbps  PSNR={row['psnr']:5.2f}  "
                  f"SA-PSNR={row['sa_psnr']:5.2f}  "
                  f"mIoU={row['miou_pct']:5.2f}%  iIoU={row['iiou_pct']:5.2f}%")

    # Save raw CSV
    raw_csv = out_dir / "bitrate_accuracy.csv"
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nRaw CSV: {raw_csv}")

    # ── Compute BD metrics ────────────────────────────────────────────────────
    metrics = [
        ("iiou_pct",  "iIoU (%)",    "iIoU"),
        ("miou_pct",  "mIoU (%)",    "mIoU"),
        ("sa_psnr",   "SA-PSNR (dB)", "SA-PSNR"),
        ("psnr",      "PSNR (dB)",   "PSNR"),
    ]

    bd_rows = []
    print("\n" + "=" * 70)
    print(f"{'BD-Rate / BD-Quality TABLE':^70}")
    print("=" * 70)

    for metric_key, ylabel, short in metrics:
        bd_264 = _bd_rate_pair(summary_rows, metric_key, "SA-X264", "H.264")
        bd_265 = _bd_rate_pair(summary_rows, metric_key, "SA-X265", "H.265")
        bdq_264 = _bd_quality_str(summary_rows, metric_key, "SA-X264", "H.264")
        bdq_265 = _bd_quality_str(summary_rows, metric_key, "SA-X265", "H.265")
        bd_rows.append(dict(
            metric=short,
            bd_rate_264=bd_264, bd_rate_265=bd_265,
            bd_quality_264=bdq_264, bd_quality_265=bdq_265,
        ))
        print(f"\n  Metric : {ylabel}")
        print(f"    SA-X264 vs H.264 : BD-Rate = {bd_264:+.2f}%   BD-{short} = {bdq_264:+.4f}")
        print(f"    SA-X265 vs H.265 : BD-Rate = {bd_265:+.2f}%   BD-{short} = {bdq_265:+.4f}")

    print("=" * 70)

    # Save BD CSV
    bd_csv = out_dir / "bd_results.csv"
    with open(bd_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "bd_rate_264", "bd_rate_265",
                                          "bd_quality_264", "bd_quality_265"])
        w.writeheader(); w.writerows(bd_rows)
    print(f"\nBD CSV : {bd_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating RD plots ...")
    for metric_key, ylabel, short in metrics:
        bd264 = next(r["bd_rate_264"] for r in bd_rows if r["metric"] == short)
        bd265 = next(r["bd_rate_265"] for r in bd_rows if r["metric"] == short)
        fname = f"rd_{short.lower().replace('-','_').replace(' ','_')}.png"
        _plot_rd_with_bd(
            summary_rows, out_dir / fname,
            metric_key=metric_key, ylabel=ylabel,
            title=f"Bitrate vs {ylabel}  —  QP {{{','.join(op.label for op in OPERATING_POINTS)}}}",
            bd_264=bd264, bd_265=bd265, bd_label=short,
        )

    # ── JSON summary ─────────────────────────────────────────────────────────
    (out_dir / "summary.json").write_text(json.dumps({
        "split": args.split, "split_desc": split_desc,
        "num_frames": n, "fps": args.fps, "preset": args.preset,
        "model": model_src, "num_classes": num_cls,
        "operating_points": [op._asdict() for op in OPERATING_POINTS],
        "bd_results": bd_rows,
        "results": summary_rows,
    }, indent=2))

    shutil.rmtree(str(work_dir), ignore_errors=True)
    print(f"\nAll outputs: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="BD-Rate curve: SAC vs Traditional (QP 16-28)")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preset", default="medium",
                   choices=["ultrafast","superfast","veryfast","faster",
                            "fast","medium","slow","slower","veryslow"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
