"""
RD curves – H.264/AVC group.

Methods compared:
  1. Baseline H.264   – full frame, single CRF
  2. SA-CCNet-x264    – SAC with CCNet ROI mask, encoded with x264
  3. SA-PIDNet-x264   – SAC with PIDNet-L ROI mask, encoded with x264

mIoU / iIoU for ALL methods are evaluated with the same PIDNet-L model.

Operating points (base_crf → roi_crf/non_crf):
  22 → 19/25 | 25 → 22/28 | 28 → 25/31 | 31 → 28/34 | 34 → 31/37

Usage:
  cd /home/huy/sac_project/scripts
  conda run -n sac python3 new_feature/rd_x264.py --num-frames 1
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

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
PIDNET_PATH  = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
CCNET_PATH   = PROJECT_ROOT / "models" / "best_ccnet_4class.pth"
IMG_TEST     = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LBL_TEST     = PROJECT_ROOT / "data" / "gt_4class" / "test"
TEST_CITIES  = ["strasbourg", "ulm"]
CODEC        = "libx264"


class OP(NamedTuple):
    label: str
    crf_roi: int
    crf_non: int

    @property
    def crf_trad(self) -> int:
        return int(self.label)


OPERATING_POINTS: List[OP] = [
    OP("22", 19, 25),
    OP("25", 22, 28),
    OP("28", 25, 31),
    OP("31", 28, 34),
    OP("34", 31, 37),
]

SEG_SIZE = (512, 1024)
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── model loading ─────────────────────────────────────────────────────────────

def _load_pidnet(device: torch.device) -> torch.nn.Module:
    ckpt  = torch.load(str(PIDNET_PATH), map_location=device, weights_only=False)
    meta  = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    num_cls = int(meta.get("num_classes", 4))
    from train_segmentation import load_segmentation_model as _ll
    model, _ = _ll(str(PIDNET_PATH), device, num_classes=num_cls)
    model.eval()
    print(f"PIDNet-L loaded ({num_cls}-class): {PIDNET_PATH}")
    return model


def _load_ccnet(device: torch.device) -> torch.nn.Module:
    ckpt  = torch.load(str(CCNET_PATH), map_location=device, weights_only=False)
    meta  = ckpt.get("meta", {})
    state = ckpt.get("model_state_dict", ckpt)
    num_cls = int(meta.get("num_classes", 4))
    from new_feature.ccnet_4class import CCNet4Class
    model = CCNet4Class(num_classes=num_cls, pretrained=False).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"CCNet-4class loaded ({num_cls}-class): {CCNET_PATH}")
    return model


def _segment(model: torch.nn.Module, device: torch.device, frame_rgb: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    img_pil = Image.fromarray(frame_rgb).resize((SEG_SIZE[1], SEG_SIZE[0]), Image.BILINEAR)
    tensor = _TRANSFORM(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred   = logits.argmax(1)[0]
    return F.interpolate(
        pred.float().unsqueeze(0).unsqueeze(0),
        size=(H, W), mode="nearest",
    )[0, 0].byte().cpu().numpy()


# ── encode/decode ──────────────────────────────────────────────────────────────

def _encode_decode_baseline(
    frames_bgr: List[np.ndarray], crf: int,
    work_dir: Path, tag: str, fps: int, preset: str,
) -> Tuple[List[np.ndarray], int]:
    dec, total = compress_traditional(
        frames_bgr, codec=CODEC, crf=crf,
        work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
    )
    return dec, total


def _encode_decode_sac(
    frames_bgr: List[np.ndarray], roi_masks: List[np.ndarray],
    crf_roi: int, crf_non: int,
    work_dir: Path, tag: str, fps: int, preset: str,
) -> Tuple[List[np.ndarray], int, int, int]:
    dec, roi_b, non_b = compress_sac(
        frames_bgr, roi_masks, codec=CODEC,
        crf_roi=crf_roi, crf_non=crf_non,
        work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
    )
    return dec, roi_b + non_b, roi_b, non_b


# ── main ───────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = PROJECT_ROOT / "outputs" / "new_feature" / "rd_x264" / ts
    work_dir = out_dir / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {out_dir}")

    pidnet = _load_pidnet(device)
    ccnet  = _load_ccnet(device)

    dataset   = Cityscapes4Class(IMG_TEST, LBL_TEST, augment=False, cities=TEST_CITIES)
    n_total   = min(args.num_frames, len(dataset))
    print(f"Frames : {n_total} / {len(dataset)}")

    frames_bgr: List[np.ndarray] = []
    frames_rgb: List[np.ndarray] = []
    gt_labels:  List[np.ndarray] = []
    masks_pid:  List[np.ndarray] = []   # PIDNet ROI masks
    masks_cc:   List[np.ndarray] = []   # CCNet  ROI masks

    print("\nPre-computing ROI masks (PIDNet-L + CCNet)...")
    for i in tqdm(range(n_total)):
        _, lbl_gt, img_path = dataset[i]
        orig_rgb = np.array(Image.open(img_path).convert("RGB"))
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        H, W     = orig_rgb.shape[:2]
        lbl_np   = lbl_gt.numpy().astype(np.uint8)
        if lbl_np.shape != (H, W):
            lbl_np = cv2.resize(lbl_np, (W, H), interpolation=cv2.INTER_NEAREST)

        seg_pid = _segment(pidnet, device, orig_rgb)
        seg_cc  = _segment(ccnet,  device, orig_rgb)

        frames_bgr.append(orig_bgr)
        frames_rgb.append(orig_rgb)
        gt_labels.append(lbl_np)
        masks_pid.append(macroblock_align_filter((seg_pid == 0).astype(np.uint8)))
        masks_cc.append(macroblock_align_filter((seg_cc  == 0).astype(np.uint8)))

    roi_pid = np.mean([m.mean() for m in masks_pid]) * 100
    roi_cc  = np.mean([m.mean() for m in masks_cc])  * 100
    print(f"  ROI ratio – PIDNet: {roi_pid:.1f}%  CCNet: {roi_cc:.1f}%")

    duration_sec = n_total / float(args.fps)
    summary_rows: List[dict] = []

    print("\nEncoding / decoding / metrics...")
    total_jobs = len(OPERATING_POINTS) * 3
    job = 0

    for op in OPERATING_POINTS:
        # ── 1. Baseline H.264 ──────────────────────────────────────────────
        job += 1
        print(f"\n[{job}/{total_jobs}] op={op.label}  Baseline H.264 (crf={op.crf_trad})")
        dec_bgr, total_b = _encode_decode_baseline(
            frames_bgr, crf=op.crf_trad,
            work_dir=work_dir, tag=f"op{op.label}_base", fps=args.fps, preset=args.preset,
        )
        bitrate = (total_b * 8.0 / duration_sec) / 1000.0
        row = _compute_metrics(
            dec_bgr, frames_rgb, gt_labels, masks_pid,
            pidnet, device, op.crf_trad, op.crf_trad,
            method="H.264", op_label=op.label,
            total_b=total_b, roi_b=total_b//2, non_b=total_b-total_b//2,
            bitrate=bitrate, is_sac=False,
        )
        summary_rows.append(row)
        _print_row(row)

        # ── 2. SA-CCNet x264 ───────────────────────────────────────────────
        job += 1
        print(f"\n[{job}/{total_jobs}] op={op.label}  SA-CCNet-x264 (crf_roi={op.crf_roi}, crf_non={op.crf_non})")
        dec_bgr, total_b, roi_b, non_b = _encode_decode_sac(
            frames_bgr, masks_cc, crf_roi=op.crf_roi, crf_non=op.crf_non,
            work_dir=work_dir, tag=f"op{op.label}_ccnet", fps=args.fps, preset=args.preset,
        )
        bitrate = (total_b * 8.0 / duration_sec) / 1000.0
        row = _compute_metrics(
            dec_bgr, frames_rgb, gt_labels, masks_cc,
            pidnet, device, op.crf_roi, op.crf_non,
            method="SA-CCNet", op_label=op.label,
            total_b=total_b, roi_b=roi_b, non_b=non_b,
            bitrate=bitrate, is_sac=True,
        )
        summary_rows.append(row)
        _print_row(row)

        # ── 3. SA-PIDNet x264 ─────────────────────────────────────────────
        job += 1
        print(f"\n[{job}/{total_jobs}] op={op.label}  SA-PIDNet-x264 (crf_roi={op.crf_roi}, crf_non={op.crf_non})")
        dec_bgr, total_b, roi_b, non_b = _encode_decode_sac(
            frames_bgr, masks_pid, crf_roi=op.crf_roi, crf_non=op.crf_non,
            work_dir=work_dir, tag=f"op{op.label}_pidnet", fps=args.fps, preset=args.preset,
        )
        bitrate = (total_b * 8.0 / duration_sec) / 1000.0
        row = _compute_metrics(
            dec_bgr, frames_rgb, gt_labels, masks_pid,
            pidnet, device, op.crf_roi, op.crf_non,
            method="SA-PIDNet", op_label=op.label,
            total_b=total_b, roi_b=roi_b, non_b=non_b,
            bitrate=bitrate, is_sac=True,
        )
        summary_rows.append(row)
        _print_row(row)

    _save_outputs(summary_rows, out_dir, args)
    shutil.rmtree(str(work_dir), ignore_errors=True)
    print(f"\nAll outputs: {out_dir}")


def _compute_metrics(
    dec_bgr: List[np.ndarray],
    frames_rgb: List[np.ndarray],
    gt_labels: List[np.ndarray],
    roi_masks: List[np.ndarray],
    eval_model: torch.nn.Module,
    device: torch.device,
    crf_roi: int, crf_non: int,
    method: str, op_label: str,
    total_b: int, roi_b: int, non_b: int,
    bitrate: float, is_sac: bool,
) -> dict:
    psnrs, ssims, sapsnrs, sassims, mious, iious = [], [], [], [], [], []
    n = min(len(dec_bgr), len(frames_rgb))
    for i in range(n):
        orig_rgb = frames_rgb[i]
        dec_rgb  = cv2.cvtColor(dec_bgr[i], cv2.COLOR_BGR2RGB)
        roi_bool = roi_masks[i].astype(bool)

        psnrs.append(compute_psnr(orig_rgb, dec_rgb))
        ssims.append(compute_ssim(orig_rgb, dec_rgb))
        sapsnrs.append(compute_sa_psnr(orig_rgb, dec_rgb, roi_bool, crf_roi, crf_non))
        sassims.append(compute_sa_ssim(orig_rgb, dec_rgb, roi_bool, crf_roi, crf_non))

        seg_dec = _segment(eval_model, device, dec_rgb)
        mious.append(compute_miou(seg_dec, gt_labels[i], num_classes=4))
        iious.append(compute_iiou(seg_dec, gt_labels[i]))

    return {
        "op_label":    op_label,
        "method":      method,
        "codec":       CODEC,
        "is_sac":      is_sac,
        "crf_roi":     crf_roi,
        "crf_non":     crf_non,
        "total_bytes": total_b,
        "roi_bytes":   roi_b,
        "non_bytes":   non_b,
        "bitrate_kbps": bitrate,
        "psnr":     float(np.mean(psnrs)),
        "ssim":     float(np.mean(ssims)),
        "sa_psnr":  float(np.mean(sapsnrs)),
        "sa_ssim":  float(np.mean(sassims)),
        "miou":     float(np.mean(mious)),
        "iiou":     float(np.mean(iious)),
        "miou_pct": float(np.mean(mious)) * 100,
        "iiou_pct": float(np.mean(iious)) * 100,
        "n_frames": n,
    }


def _print_row(r: dict) -> None:
    print(f"  bitrate={r['bitrate_kbps']:8.1f} kbps  "
          f"PSNR={r['psnr']:5.2f}  SA-PSNR={r['sa_psnr']:5.2f}  "
          f"mIoU={r['miou_pct']:5.2f}%  iIoU={r['iiou_pct']:5.2f}%")


def _save_outputs(rows: List[dict], out_dir: Path, args: argparse.Namespace) -> None:
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"CSV: {csv_path}")

    (out_dir / "summary.json").write_text(json.dumps({
        "codec": CODEC,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "preset": args.preset,
        "operating_points": [op._asdict() for op in OPERATING_POINTS],
        "results": rows,
    }, indent=2))

    _plot(rows, out_dir / "rd_miou.png",
          metric="miou_pct", ylabel="mIoU (%)",
          title="Rate–Accuracy: H.264/AVC group")
    _plot(rows, out_dir / "rd_sa_psnr.png",
          metric="sa_psnr", ylabel="SA-PSNR (dB)",
          title="Rate–Quality (SA-PSNR): H.264/AVC group")


def _plot(rows: List[dict], out_png: Path, metric: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "H.264":     {"color": "#1f77b4", "marker": "o", "ls": "-"},
        "SA-CCNet":  {"color": "#ff7f0e", "marker": "s", "ls": "--"},
        "SA-PIDNet": {"color": "#2ca02c", "marker": "^", "ls": "--"},
    }
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    for method, s in styles.items():
        pts = sorted([r for r in rows if r["method"] == method],
                     key=lambda r: r["bitrate_kbps"])
        if not pts:
            continue
        xs = [r["bitrate_kbps"] for r in pts]
        ys = [r[metric]         for r in pts]
        ax.plot(xs, ys, s["ls"], color=s["color"], marker=s["marker"],
                markersize=8, linewidth=2, label=method)
        for r in pts:
            ax.annotate(f"CRF{r['op_label']}",
                        (r["bitrate_kbps"], r[metric]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=s["color"])
    ax.set_xlabel("Bitrate (kbps)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    print(f"Plot: {out_png}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-frames", type=int, default=50)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--preset",     default="medium")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse())
