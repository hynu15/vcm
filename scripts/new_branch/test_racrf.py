"""test_racrf.py — so sánh RA-CRF (Adaptive CRF) trên N frame.

Phương pháp:
  H264             : libx264 CRF=23, không SAC (baseline)
  CCNet-SA-H264    : SAC + CCNet + RA-CRF (base_crf=22)
  PIDNet-SA-H264   : SAC + PIDNet-L + RA-CRF (base_crf=22)

Mặc định test 2 frame (--num-frames 2). Tăng lên để test nhiều hơn.
Mỗi GOP 30 frame tính roi_ratio mới và cập nhật CRF.

Output:
  outputs/new_branch/eval/racrf_<timestamp>/
    per_frame.csv   — metric từng frame × method
    summary.csv     — tổng hợp per-method
    summary.json    — cùng nội dung, dễ parse
    config.json     — tham số chạy

Chạy:
  conda run -n sac python3 scripts/new_branch/test_racrf.py
  conda run -n sac python3 scripts/new_branch/test_racrf.py --num-frames 30
  conda run -n sac python3 scripts/new_branch/test_racrf.py --start-idx 50 --num-frames 10
"""
from __future__ import annotations

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

from . import metrics as M
from .compression import compress_sac, compress_traditional
from .dataset import CityscapesSAC, _normalize, _rgb_to_gray3
from .model import build_pidnet_l, build_segnet
from .paths import (
    EVAL_DIR, FRAME_HW, NUM_CLASSES, PIDNET_L_CHECKPOINT, ROI_CLASS_ID,
    SAC_CHECKPOINT, SEG_INPUT_HW,
)
from .stream_separation import build_roi_mask, gop_roi_ratio, select_delta_crf

GOP_SIZE = 30
BASE_CRF = 22
H264_CRF = 23


@dataclass
class FrameRow:
    method: str
    frame_idx: int
    frame_name: str
    gop_idx: int
    crf_roi: int
    crf_non: int
    roi_ratio: float
    delta_crf: int
    bytes_gop: int
    kb_per_frame: float
    miou_before: float
    miou_after: float
    miou_drop: float
    iiou_before: float
    iiou_after: float
    iiou_drop: float
    psnr: float
    sa_psnr: float


# ── model helpers ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_lowres(model, frame_rgb_full: np.ndarray, device, use_gray: bool) -> np.ndarray:
    H, W = SEG_INPUT_HW
    lowres = np.array(Image.fromarray(frame_rgb_full).resize((W, H), Image.BILINEAR))
    if use_gray:
        lowres = _rgb_to_gray3(lowres)
    t = _normalize(lowres).unsqueeze(0).to(device)
    use_amp = device.type == "cuda"
    with autocast(device_type="cuda", enabled=use_amp):
        out = model(t)
        logits = out[0] if isinstance(out, (list, tuple)) else out
    logits = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=True)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _seg_to_full(seg_lowres: np.ndarray) -> np.ndarray:
    return np.array(
        Image.fromarray(seg_lowres).resize((FRAME_HW[1], FRAME_HW[0]), Image.NEAREST)
    )


def _load_model(kind: str, ckpt_path: str, device):
    if kind == "pidnet_l":
        m = build_pidnet_l(num_classes=NUM_CLASSES)
        target = m.backbone
    else:
        m = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=False)
        target = m
    p = Path(ckpt_path)
    if p.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = (ck.get("model_state_dict") or ck.get("state_dict") or ck
              if isinstance(ck, dict) else ck)
        missing, _ = target.load_state_dict(sd, strict=False)
        if missing:
            print(f"    ⚠ {kind}: {len(missing)} missing keys")
        print(f"    Loaded: {p.name}")
    else:
        print(f"    ⚠ {ckpt_path} not found — random weights")
    return m.to(device).eval()


# ── per-method runner ──────────────────────────────────────────────────────────

def _run_method(
    method: str,
    frames_full: list[np.ndarray],
    gt_list: list[np.ndarray],
    names: list[str],
    seg_model,
    seg_use_gray: bool,
    seg_lowres_list: list[np.ndarray],
    roi_masks: list[np.ndarray],
    device,
    is_sac: bool,
    h264_crf: int,
    base_crf: int,
) -> list[FrameRow]:
    """Nén toàn bộ frames theo GOP, tính metric, trả về list FrameRow."""
    N = len(frames_full)
    rows: list[FrameRow] = []

    for gop_i, start in enumerate(range(0, N, GOP_SIZE)):
        chunk_idx = list(range(start, min(start + GOP_SIZE, N)))
        chunk_frames = [frames_full[i] for i in chunk_idx]
        chunk_gt     = [gt_list[i]     for i in chunk_idx]
        chunk_names  = [names[i]       for i in chunk_idx]
        chunk_segs   = [seg_lowres_list[i] for i in chunk_idx]

        if is_sac:
            chunk_masks = [roi_masks[i] for i in chunk_idx]
            ratio  = gop_roi_ratio(chunk_masks)
            delta  = select_delta_crf(ratio)
            crf_r  = base_crf - delta
            crf_n  = base_crf + delta
            print(f"  GOP {gop_i} [{chunk_idx[0]:4d}-{chunk_idx[-1]:4d}]: "
                  f"roi_ratio={ratio:.3f}  Δ={delta}  CRF=({crf_r},{crf_n})")
            recon_chunk, b_roi, b_non = compress_sac(
                chunk_frames, chunk_masks, "libx264", crf_r, crf_n)
            gop_bytes = b_roi + b_non
        else:
            ratio, delta, crf_r, crf_n = 0.0, 0, h264_crf, h264_crf
            chunk_masks = [roi_masks[i] for i in chunk_idx]  # dùng để tính SA-PSNR
            recon_chunk, gop_bytes = compress_traditional(chunk_frames, "libx264", h264_crf)
            print(f"  GOP {gop_i} [{chunk_idx[0]:4d}-{chunk_idx[-1]:4d}]: "
                  f"CRF={h264_crf}  bytes={gop_bytes:,}")

        kb_per_frame = gop_bytes / 1024 / len(chunk_idx)

        for local_i, (global_i, orig, recon, gt, mask, seg_b_low, fname) in enumerate(
            zip(chunk_idx, chunk_frames, recon_chunk, chunk_gt, chunk_masks, chunk_segs, chunk_names)
        ):
            # Seg-before
            sb_full = _seg_to_full(seg_b_low)
            mb, _ = M.per_class_iou(sb_full, gt, NUM_CLASSES)
            ib = M.iiou(sb_full, gt, ROI_CLASS_ID)

            # Seg-after trên frame tái tạo
            sa_low  = _predict_lowres(seg_model, recon, device, use_gray=seg_use_gray)
            sa_full = _seg_to_full(sa_low)
            ma, _ = M.per_class_iou(sa_full, gt, NUM_CLASSES)
            ia = M.iiou(sa_full, gt, ROI_CLASS_ID)

            # PSNR + SA-PSNR
            p_full = M.psnr(orig, recon)
            P_i, S_i = M.regional_psnr_ssim(orig, recon, mask)
            P_n, S_n = M.regional_psnr_ssim(orig, recon, 1 - mask)
            sa_p, _ = M.sa_psnr_ssim(crf_r, crf_n, P_i, P_n, S_i, S_n)

            rows.append(FrameRow(
                method=method,
                frame_idx=global_i,
                frame_name=fname,
                gop_idx=gop_i,
                crf_roi=crf_r,
                crf_non=crf_n,
                roi_ratio=ratio,
                delta_crf=delta,
                bytes_gop=gop_bytes,
                kb_per_frame=kb_per_frame,
                miou_before=mb,
                miou_after=ma,
                miou_drop=mb - ma,
                iiou_before=ib,
                iiou_after=ia,
                iiou_drop=ib - ia,
                psnr=p_full,
                sa_psnr=sa_p,
            ))

    return rows


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RA-CRF quick test: H264 vs CCNet/PIDNet SA-H264")
    ap.add_argument("--num-frames", type=int, default=2,
                    help="Số frame test (mặc định 2; tăng lên để test nhiều hơn)")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Frame bắt đầu trong split (0-based)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--base-crf", type=int, default=BASE_CRF,
                    help=f"C_base cho RA-CRF (mặc định {BASE_CRF})")
    ap.add_argument("--h264-crf", type=int, default=H264_CRF,
                    help=f"CRF cho H264 baseline (mặc định {H264_CRF})")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Thư mục lưu kết quả (mặc định: outputs/new_branch/eval/racrf_<timestamp>)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else EVAL_DIR / f"racrf_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Split      : {args.split}")
    print(f"Frames     : [{args.start_idx}, {args.start_idx + args.num_frames}) "
          f"(N={args.num_frames})")
    print(f"Base CRF   : {args.base_crf}  (SA: C_roi=base-Δ, C_non=base+Δ)")
    print(f"H264 CRF   : {args.h264_crf}")
    print(f"Output dir : {out_dir}")
    print()

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    ds = CityscapesSAC(split=args.split, augment=False, return_original=True)
    total = len(ds)
    idx0 = max(0, min(args.start_idx, total - args.num_frames))
    N = min(args.num_frames, total - idx0)
    indices = list(range(idx0, idx0 + N))

    frames_full, gt_list, names = [], [], []
    for i in indices:
        img_t, _lbl, _orig_low, img_path, lbl_path = ds[i]
        names.append(Path(img_path).name)
        f = np.array(Image.open(img_path).convert("RGB"))
        if f.shape[:2] != FRAME_HW:
            f = np.array(Image.fromarray(f).resize((FRAME_HW[1], FRAME_HW[0]), Image.BILINEAR))
        frames_full.append(f)
        g = np.array(Image.open(lbl_path))
        if g.shape[:2] != FRAME_HW:
            g = np.array(Image.fromarray(g).resize((FRAME_HW[1], FRAME_HW[0]), Image.NEAREST))
        gt_list.append(g)

    print(f"Loaded {N} frame(s): {names[:3]}{'...' if N > 3 else ''}")

    # ── 2. Load models ─────────────────────────────────────────────────────────
    print("\n[Load models]")
    ccnet  = _load_model("ccnet",    str(SAC_CHECKPOINT),      device)
    pidnet = _load_model("pidnet_l", str(PIDNET_L_CHECKPOINT), device)

    # ── 3. Predict ROI masks trên frame gốc ───────────────────────────────────
    print(f"\n[Predict ROI masks — {N} frame]")
    cc_segs,  cc_masks  = [], []
    pid_segs, pid_masks = [], []
    for j, f in enumerate(frames_full):
        seg_cc  = _predict_lowres(ccnet,  f, device, use_gray=True)
        seg_pid = _predict_lowres(pidnet, f, device, use_gray=False)
        cc_segs.append(seg_cc);   cc_masks.append(build_roi_mask(seg_cc,  FRAME_HW))
        pid_segs.append(seg_pid); pid_masks.append(build_roi_mask(seg_pid, FRAME_HW))
        if (j + 1) % 10 == 0 or j == N - 1:
            print(f"  {j+1}/{N}")

    # ── 4. Compress + metrics ──────────────────────────────────────────────────
    all_rows: list[FrameRow] = []

    print(f"\n[H264  CRF={args.h264_crf}]")
    rows_h264 = _run_method(
        "H264", frames_full, gt_list, names,
        seg_model=ccnet, seg_use_gray=True, seg_lowres_list=cc_segs,
        roi_masks=cc_masks, device=device,
        is_sac=False, h264_crf=args.h264_crf, base_crf=args.base_crf,
    )
    all_rows.extend(rows_h264)

    print(f"\n[CCNet-SA-H264  base_crf={args.base_crf}]")
    rows_cc = _run_method(
        "CCNet-SA-H264", frames_full, gt_list, names,
        seg_model=ccnet, seg_use_gray=True, seg_lowres_list=cc_segs,
        roi_masks=cc_masks, device=device,
        is_sac=True, h264_crf=args.h264_crf, base_crf=args.base_crf,
    )
    all_rows.extend(rows_cc)

    print(f"\n[PIDNet-SA-H264  base_crf={args.base_crf}]")
    rows_pid = _run_method(
        "PIDNet-SA-H264", frames_full, gt_list, names,
        seg_model=pidnet, seg_use_gray=False, seg_lowres_list=pid_segs,
        roi_masks=pid_masks, device=device,
        is_sac=True, h264_crf=args.h264_crf, base_crf=args.base_crf,
    )
    all_rows.extend(rows_pid)

    # ── 5. Save CSV per-frame ──────────────────────────────────────────────────
    per_frame_path = out_dir / "per_frame.csv"
    fields = list(FrameRow.__dataclass_fields__.keys())
    with open(per_frame_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for row in all_rows:
            w.writerow([getattr(row, k) for k in fields])

    # ── 6. Aggregate summary ───────────────────────────────────────────────────
    def _agg(rows: list[FrameRow]) -> dict:
        def _m(attr): return float(np.mean([getattr(r, attr) for r in rows]))
        r0 = rows[0]
        return {
            "method":        r0.method,
            "num_frames":    len(rows),
            "avg_crf_roi":   _m("crf_roi"),
            "avg_crf_non":   _m("crf_non"),
            "avg_roi_ratio": _m("roi_ratio"),
            "avg_delta_crf": _m("delta_crf"),
            "kb_per_frame":  _m("kb_per_frame"),
            "miou_before":   _m("miou_before"),
            "miou_after":    _m("miou_after"),
            "miou_drop":     _m("miou_drop"),
            "iiou_before":   _m("iiou_before"),
            "iiou_after":    _m("iiou_after"),
            "iiou_drop":     _m("iiou_drop"),
            "psnr":          _m("psnr"),
            "sa_psnr":       _m("sa_psnr"),
        }

    summaries = [_agg(rows_h264), _agg(rows_cc), _agg(rows_pid)]

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(summaries[0].keys()))
        for s in summaries:
            w.writerow(list(s.values()))

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summaries, indent=2))

    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps({
        "split": args.split, "start_idx": idx0, "num_frames": N,
        "base_crf": args.base_crf, "h264_crf": args.h264_crf,
        "gop_size": GOP_SIZE, "timestamp": ts,
    }, indent=2))

    # ── 7. Print summary table ─────────────────────────────────────────────────
    hdr = (f"\n{'Method':<22} {'CRF(r,n)':<12} {'kB/frame':>9} {'roi_ratio':>10}"
           f" {'Δ':>3} {'mIoU-B':>8} {'mIoU-A':>8} {'Drop':>8}"
           f" {'iIoU-B':>8} {'iIoU-A':>8} {'PSNR':>7} {'SA-PSNR':>8}")
    sep = "─" * len(hdr.strip())
    print(); print(sep); print(hdr); print(sep)

    for s in summaries:
        is_sac = s["method"] != "H264"
        roi_r_str = f"{s['avg_roi_ratio']:.3f}" if is_sac else "  —  "
        delt_str  = f"{s['avg_delta_crf']:.1f}"  if is_sac else "—"
        print(
            f"{s['method']:<22} ({s['avg_crf_roi']:.0f},{s['avg_crf_non']:.0f})      "
            f"{s['kb_per_frame']:>9.1f} {roi_r_str:>10} {delt_str:>3}"
            f" {s['miou_before']*100:>7.2f}%"
            f" {s['miou_after']*100:>7.2f}%"
            f" {s['miou_drop']*100:>+7.2f}%"
            f" {s['iiou_before']*100:>7.2f}%"
            f" {s['iiou_after']*100:>7.2f}%"
            f" {s['psnr']:>7.2f}"
            f" {s['sa_psnr']:>8.3f}"
        )

    print(sep)
    print()
    print(f"Saved → {out_dir}/")
    print(f"  per_frame.csv  ({len(all_rows)} rows)")
    print(f"  summary.csv    ({len(summaries)} methods)")
    print(f"  summary.json")
    print(f"  config.json")
    print()
    print("Ghi chú RA-CRF rule: roi_ratio < 0.25 → Δ=5 | 0.25–0.60 → Δ=3 | >0.60 → Δ=2")
    print()


if __name__ == "__main__":
    main()
