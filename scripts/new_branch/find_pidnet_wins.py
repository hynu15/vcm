"""Quét N frame, tìm frame thoả:
  H.264 trad < SA-X264 CCNet backbone < SA-X264 PIDNet backbone  (đo bằng PIDNet)
  H.265 trad < SA-X265 CCNet backbone < SA-X265 PIDNet backbone  (đo bằng PIDNet)

Sinh ra 2 figure riêng biệt:
  family_h264_rank<K>.png  — best frame cho H.264 family
  family_h265_rank<K>.png  — best frame cho H.265 family
  (có thể là 2 frame khác nhau)

Chạy:
  cd /home/huy/sac_project
  conda run -n sac python3 -m scripts.new_branch.find_pidnet_wins \\
      --n-frames 60 --top-k 3
"""
from __future__ import annotations

if __package__ in (None, ""):
    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "scripts.new_branch"

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .paths import EVAL_DIR, FRAME_HW, SEG_INPUT_HW
from .stream_separation import build_roi_mask
from .visualize_seg_comparison import (
    _CRF_SA, _CRF_TRAD,
    _load_model, compress_and_decode, compute_iiou,
    load_orig_and_gt, make_figure_codec_family, predict_seg,
)

# Mapping codec family
_FAMILIES = {
    "h264": {"trad": "H264",   "sa": "SA-X264",
             "codec_label": "H.264", "sa_label": "SA-X264"},
    "h265": {"trad": "H265",   "sa": "SA-X265",
             "codec_label": "H.265", "sa_label": "SA-X265"},
}


def process_frame(orig, gt, cc_model, cc_gray, pid_model, pid_gray, device):
    """Chạy pipeline 4 codec, trả về dicts chứa PIDNet-eval segs và iIoU.

    Quy ước:
      segs_trad[key]   = PIDNet seg trên frame nén truyền thống
      iious_trad[key]  = PIDNet iIoU trên frame nén truyền thống
      segs_cc[key]     = PIDNet seg trên frame SAC với CCNet ROI
      iious_cc[key]    = PIDNet iIoU trên frame SAC với CCNet ROI
      segs_pid[key]    = PIDNet seg trên frame SAC với PIDNet ROI
      iious_pid[key]   = PIDNet iIoU trên frame SAC với PIDNet ROI
    """
    seg_cc_orig  = predict_seg(cc_model,  orig, device, gray_input=cc_gray)
    seg_pid_orig = predict_seg(pid_model, orig, device, gray_input=pid_gray)
    roi_cc  = build_roi_mask(
        np.array(Image.fromarray(seg_cc_orig).resize(
            (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.NEAREST)),
        target_hw=FRAME_HW)
    roi_pid = build_roi_mask(
        np.array(Image.fromarray(seg_pid_orig).resize(
            (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.NEAREST)),
        target_hw=FRAME_HW)

    segs_trad:  dict[str, np.ndarray] = {}
    iious_trad: dict[str, float]      = {}
    segs_cc:    dict[str, np.ndarray] = {}
    iious_cc:   dict[str, float]      = {}
    segs_pid:   dict[str, np.ndarray] = {}
    iious_pid:  dict[str, float]      = {}

    # Traditional — dùng PIDNet để seg + eval
    for key, cfg in _CRF_TRAD.items():
        codec, crf_r, crf_n = cfg["codec"], cfg["crf_roi"], cfg["crf_non"]
        recon = compress_and_decode(orig, roi_cc, codec, crf_r, crf_n)
        seg = predict_seg(pid_model, recon, device, gray_input=pid_gray)
        segs_trad[key]  = seg
        iious_trad[key] = compute_iiou(seg, gt)

    # SAC — PIDNet eval trên CCNet-ROI frame và PIDNet-ROI frame
    for key, cfg in _CRF_SA.items():
        codec, crf_r, crf_n = cfg["codec"], cfg["crf_roi"], cfg["crf_non"]
        recon_cc  = compress_and_decode(orig, roi_cc,  codec, crf_r, crf_n)
        recon_pid = compress_and_decode(orig, roi_pid, codec, crf_r, crf_n)
        seg_c = predict_seg(pid_model, recon_cc,  device, gray_input=pid_gray)
        seg_p = predict_seg(pid_model, recon_pid, device, gray_input=pid_gray)
        segs_cc[key]  = seg_c;  iious_cc[key]  = compute_iiou(seg_c, gt)
        segs_pid[key] = seg_p;  iious_pid[key] = compute_iiou(seg_p, gt)

    return segs_trad, iious_trad, segs_cc, iious_cc, segs_pid, iious_pid


def scan_frames(cc_model, cc_gray, pid_model, pid_gray,
                device, split: str, n_frames: int):
    from .dataset import _list_split
    pairs = _list_split(split)
    total = min(n_frames, len(pairs))
    rows  = []

    for idx in range(total):
        img_name = pairs[idx][0].name
        print(f"  [{idx+1:>3}/{total}] {img_name}", end=" → ", flush=True)

        orig, gt = load_orig_and_gt(img_name, split)
        strad, itrad, scc, icc, spid, ipid = process_frame(
            orig, gt, cc_model, cc_gray, pid_model, pid_gray, device)

        row = {"idx": idx, "name": img_name}
        parts = []
        for fam, info in _FAMILIES.items():
            tk, sk = info["trad"], info["sa"]
            row[f"iiou_trad_{fam}"] = itrad[tk]
            row[f"iiou_cc_{fam}"]   = icc[sk]
            row[f"iiou_pid_{fam}"]  = ipid[sk]
            # Điều kiện hierarchy: trad < cc < pid
            row[f"ok_{fam}"] = (itrad[tk] < icc[sk] < ipid[sk])
            # Margin = PIDNet SAC - Traditional
            row[f"margin_{fam}"] = ipid[sk] - itrad[tk]
            parts.append(
                f"{fam.upper()}: "
                f"trad={itrad[tk]:.4f} cc={icc[sk]:.4f} pid={ipid[sk]:.4f} "
                f"ok={'✓' if row[f'ok_{fam}'] else '✗'}")
        print("  |  ".join(parts))
        rows.append(row)

    return rows


def save_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {out_path}")


def visualize_top(rows, fam: str, top_k: int,
                  cc_model, cc_gray, pid_model, pid_gray,
                  device, split: str, out_dir: Path) -> None:
    """Tạo figure cho top-K frame thỏa hierarchy của codec family."""
    info = _FAMILIES[fam]
    tk, sk = info["trad"], info["sa"]

    # Lọc frame thỏa hierarchy, xếp theo margin giảm dần
    qualifying = [r for r in rows if r[f"ok_{fam}"]]
    top = sorted(qualifying, key=lambda r: r[f"margin_{fam}"], reverse=True)[:top_k]

    if not top:
        print(f"  [{fam.upper()}] Không có frame nào thỏa trad < CCNet SAC < PIDNet SAC.")
        return

    print(f"\n  [{fam.upper()}] {len(qualifying)}/{len(rows)} frame thỏa điều kiện "
          f"→ visualize top-{len(top)}")

    from .dataset import _list_split
    pairs = _list_split(split)

    for rank, row in enumerate(top, 1):
        idx      = row["idx"]
        img_name = row["name"]
        print(f"    rank={rank} idx={idx} "
              f"trad={row[f'iiou_trad_{fam}']:.4f} "
              f"cc={row[f'iiou_cc_{fam}']:.4f} "
              f"pid={row[f'iiou_pid_{fam}']:.4f}  {img_name}")

        orig, gt = load_orig_and_gt(img_name, split)
        strad, itrad, scc, icc, spid, ipid = process_frame(
            orig, gt, cc_model, cc_gray, pid_model, pid_gray, device)

        out_path = out_dir / f"family_{fam}_rank{rank:02d}_idx{idx}.png"
        make_figure_codec_family(
            orig, gt,
            seg_trad    = strad[tk],
            seg_cc_sac  = scc[sk],
            seg_pid_sac = spid[sk],
            iiou_trad   = itrad[tk],
            iiou_cc_sac = icc[sk],
            iiou_pid_sac= ipid[sk],
            codec_label = info["codec_label"],
            sa_label    = info["sa_label"],
            out_path    = out_path,
            fig_title   = (f"{info['codec_label']} Family — "
                           f"trad={itrad[tk]:.4f} < "
                           f"CCNet={icc[sk]:.4f} < "
                           f"PIDNet={ipid[sk]:.4f}  |  {img_name}"),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",    default="test")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--top-k",    type=int, default=3,
                    help="Số figure sinh ra cho mỗi codec family.")
    ap.add_argument("--ckpt-ccnet",  default=None)
    ap.add_argument("--ckpt-pidnet", default=None)
    ap.add_argument("--out-dir",  default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | frames: {args.n_frames} | top-k: {args.top_k}")

    cc_model,  cc_gray  = _load_model("ccnet",  args.ckpt_ccnet,  device)
    pid_model, pid_gray = _load_model("pidnet", args.ckpt_pidnet, device)

    out_dir = Path(args.out_dir) if args.out_dir else EVAL_DIR / "pidnet_wins"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning {args.n_frames} frames...")
    rows = scan_frames(cc_model, cc_gray, pid_model, pid_gray,
                       device, args.split, args.n_frames)

    csv_path = out_dir / f"backbone_compare_{args.n_frames}frames.csv"
    save_csv(rows, csv_path)

    print(f"\n{'='*60}")
    for fam in ("h264", "h265"):
        ok = sum(1 for r in rows if r[f"ok_{fam}"])
        avg_margin = np.mean([r[f"margin_{fam}"] for r in rows])
        print(f"  {fam.upper()} hierarchy (trad<CC<PID): {ok}/{len(rows)} frame  "
              f"avg_margin={avg_margin:+.4f}")
    print(f"{'='*60}")

    for fam in ("h264", "h265"):
        visualize_top(rows, fam, args.top_k,
                      cc_model, cc_gray, pid_model, pid_gray,
                      device, args.split, out_dir)

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
