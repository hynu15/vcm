#!/usr/bin/env python3
"""Plot mIoU vs Bitrate RD curve with iso-bitrate annotations.

Mục tiêu: minh hoạ luận điểm của paper Wang et al. 2023 — cùng một mức
bitrate (cùng "lượng dữ liệu video sau giải mã"), SAC duy trì mIoU cao hơn
H.265 truyền thống nhờ ưu tiên chất lượng vùng ROI.

Chạy từ /home/huy/sac_project/scripts/:
  conda activate sac
  python plot_rd_isobitrate.py
  python plot_rd_isobitrate.py --csv <path/combined_rd_detail.csv> --out <png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import PchipInterpolator

# DejaVu Sans (bundled w/ matplotlib) + FreeSans đều render đủ Vietnamese diacritics
# trong cả Regular và Bold. Liberation Sans v1 (system) bị thiếu một số glyph
# nên không dùng.
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "FreeSans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN  = PROJECT_ROOT / "outputs" / "sac_qpmap_comparison" / "run_20260516_151710"


def load_rd(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("h265_bitrate_mbps").reset_index(drop=True)
    return df


def pchip_curve(x: np.ndarray, y: np.ndarray, n: int = 200):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    fn = PchipInterpolator(xs, ys, extrapolate=False)
    xx = np.linspace(xs.min(), xs.max(), n)
    return xx, fn(xx), fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_RUN / "combined_rd_detail.csv")
    ap.add_argument("--summary", type=Path, default=DEFAULT_RUN / "summary.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUN / "rd_curve_isobitrate.png")
    args = ap.parse_args()

    df = load_rd(args.csv)
    with open(args.summary) as f:
        meta = json.load(f)

    # ─── data ────────────────────────────────────────────────────────────────
    br_h265   = df["h265_bitrate_mbps"].to_numpy()
    miou_h265 = df["h265_miou"].to_numpy() * 100.0
    roi_h265  = df["h265_roi_iou"].to_numpy() * 100.0

    br_sac    = df["sac_ccnet_bitrate_mbps"].to_numpy()
    miou_sac  = df["sac_ccnet_miou"].to_numpy() * 100.0
    roi_sac   = df["sac_ccnet_roi_iou"].to_numpy() * 100.0

    # PCHIP curves (mIoU vs bitrate)
    x_h265,  y_h265,  fn_h265_miou  = pchip_curve(br_h265, miou_h265)
    x_sac,   y_sac,   fn_sac_miou   = pchip_curve(br_sac,  miou_sac)
    _,       _,       fn_h265_roi   = pchip_curve(br_h265, roi_h265)
    _,       _,       fn_sac_roi    = pchip_curve(br_sac,  roi_sac)

    # ─── iso-bitrate comparison ─────────────────────────────────────────────
    # tại mỗi bitrate của SAC, đọc mIoU H.265 trên đường PCHIP để so sánh
    iso_pts = []
    for br_pt in br_sac:
        if br_pt < br_h265.min() or br_pt > br_h265.max():
            continue
        m_sac_pt  = float(fn_sac_miou(br_pt))
        m_h265_pt = float(fn_h265_miou(br_pt))
        r_sac_pt  = float(fn_sac_roi(br_pt))
        r_h265_pt = float(fn_h265_roi(br_pt))
        iso_pts.append({
            "bitrate_mbps": br_pt,
            "miou_h265":    m_h265_pt,
            "miou_sac":     m_sac_pt,
            "delta_miou":   m_sac_pt - m_h265_pt,
            "roi_h265":     r_h265_pt,
            "roi_sac":      r_sac_pt,
            "delta_roi":    r_sac_pt - r_h265_pt,
        })
    iso_df = pd.DataFrame(iso_pts)

    # ─── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ===== panel 1: mIoU vs bitrate =====
    ax = axes[0]
    H265_COLOR = "#d62728"   # red
    SAC_COLOR  = "#1f77b4"   # blue

    ax.plot(x_h265, y_h265, "-",  color=H265_COLOR, lw=2.4, alpha=0.9,
            label="H.265 (Traditional, single-stream)")
    ax.plot(x_sac,  y_sac,  "-",  color=SAC_COLOR,  lw=2.4, alpha=0.9,
            label="SAC-CCNet (Proposed, ROI-aware)")
    ax.plot(br_h265, miou_h265, "o", color=H265_COLOR, ms=10, mec="white", mew=1.5, zorder=5)
    ax.plot(br_sac,  miou_sac,  "s", color=SAC_COLOR,  ms=10, mec="white", mew=1.5, zorder=5)

    # iso-bitrate arrows: tại mỗi bitrate, vẽ mũi tên từ H.265 lên SAC
    for _, row in iso_df.iterrows():
        if row["delta_miou"] <= 0:
            continue
        ax.annotate(
            "",
            xy=(row["bitrate_mbps"], row["miou_sac"]),
            xytext=(row["bitrate_mbps"], row["miou_h265"]),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.0),
        )
        ax.text(
            row["bitrate_mbps"], (row["miou_sac"] + row["miou_h265"]) / 2,
            f"+{row['delta_miou']:.2f}%",
            color="#1a701a", fontsize=10, fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#e8f5e8", ec="#2ca02c", lw=1.0),
            zorder=6,
        )

    # CRF labels next to scatter points
    for i, row in df.iterrows():
        ax.annotate(
            f"CRF {int(row['baseline_crf'])}",
            xy=(row["h265_bitrate_mbps"], row["h265_miou"] * 100),
            xytext=(8, -16), textcoords="offset points",
            fontsize=8, color="#7f1b1b", alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (Mbps, log scale)",  fontsize=12)
    ax.set_ylabel("mIoU on decompressed video (%)", fontsize=12)
    ax.set_title("RD curve: mIoU vs Bitrate — cùng bitrate, SAC ≥ H.265",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, ls="--", alpha=0.4, which="both")
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    # ===== panel 2: ROI-IoU vs bitrate (vùng quan trọng cho lái xe) =====
    ax2 = axes[1]
    x_h_r, y_h_r, _ = pchip_curve(br_h265, roi_h265)
    x_s_r, y_s_r, _ = pchip_curve(br_sac,  roi_sac)

    ax2.plot(x_h_r, y_h_r, "-",  color=H265_COLOR, lw=2.4, alpha=0.9,
             label="H.265 (Traditional)")
    ax2.plot(x_s_r, y_s_r, "-",  color=SAC_COLOR,  lw=2.4, alpha=0.9,
             label="SAC-CCNet (Proposed)")
    ax2.plot(br_h265, roi_h265, "o", color=H265_COLOR, ms=10, mec="white", mew=1.5, zorder=5)
    ax2.plot(br_sac,  roi_sac,  "s", color=SAC_COLOR,  ms=10, mec="white", mew=1.5, zorder=5)

    for _, row in iso_df.iterrows():
        if row["delta_roi"] <= 0:
            continue
        ax2.annotate(
            "",
            xy=(row["bitrate_mbps"], row["roi_sac"]),
            xytext=(row["bitrate_mbps"], row["roi_h265"]),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.0),
        )
        ax2.text(
            row["bitrate_mbps"], (row["roi_sac"] + row["roi_h265"]) / 2,
            f"+{row['delta_roi']:.2f}%",
            color="#1a701a", fontsize=10, fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#e8f5e8", ec="#2ca02c", lw=1.0),
            zorder=6,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Bitrate (Mbps, log scale)",  fontsize=12)
    ax2.set_ylabel("ROI-IoU on decompressed video (%)", fontsize=12)
    ax2.set_title("ROI-IoU vs Bitrate — chất lượng vùng quan trọng (xe/người/đường)",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.grid(True, ls="--", alpha=0.4, which="both")
    ax2.legend(loc="lower right", fontsize=11, frameon=True)

    # ─── header / footer info ───────────────────────────────────────────────
    bd_rate = meta.get("bd_rate_ccnet_vs_h265_pct")
    bd_acc  = meta.get("bd_acc_ccnet_vs_h265")
    n_fr    = meta.get("num_frames")
    method  = meta.get("method", "n/a")
    fig.suptitle(
        f"Semantic-Aware Compression vs Traditional H.265 on Cityscapes test split "
        f"(n={n_fr} frames, ROI mask: CCNet-4class, evaluator: PIDNet-L-4class)",
        fontsize=12, fontweight="bold", y=0.995,
    )

    info_txt = (
        f"BD-rate(SAC vs H.265) = {bd_rate:.2f}%   "
        f"(âm = SAC tốn ít bitrate hơn tại cùng accuracy)\n"
        f"BD-accuracy(SAC vs H.265) = {(bd_acc or 0)*100:+.3f}%   "
        f"(dương = SAC mIoU cao hơn tại cùng bitrate)\n"
        f"Method: {method}    "
        f"SAC mean Δbitrate = {meta.get('mean_sac_ccnet_delta_br_pct'):+.2f}%"
    )
    fig.text(0.01, 0.01, info_txt, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#888", lw=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

    # ─── iso-bitrate CSV ────────────────────────────────────────────────────
    iso_csv = args.out.with_suffix(".csv")
    iso_df.to_csv(iso_csv, index=False, float_format="%.4f")
    print(f"saved {iso_csv}")

    # quick text summary
    print("\nIso-bitrate comparison (mIoU at same bitrate):")
    for _, r in iso_df.iterrows():
        print(f"  @ {r['bitrate_mbps']:6.2f} Mbps:  "
              f"H.265 = {r['miou_h265']:6.3f}%   SAC = {r['miou_sac']:6.3f}%   "
              f"Δ = {r['delta_miou']:+.3f}%   |   "
              f"ROI: Δ = {r['delta_roi']:+.3f}%")


if __name__ == "__main__":
    main()
