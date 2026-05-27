"""
Vẽ 1 đồ thị RD curve (Bitrate vs mIoU) từ results.csv.

Usage:
  conda run -n sac python3 new_feature/plot_rd.py \
      --csv outputs/new_feature/rd_x264/20260527_123707/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STYLE = {
    "H.264":     {"color": "#1f77b4", "marker": "o",  "ls": "-",  "lw": 2.0},
    "H.265":     {"color": "#1f77b4", "marker": "o",  "ls": "-",  "lw": 2.0},
    "SA-CCNet":  {"color": "#ff7f0e", "marker": "s",  "ls": "--", "lw": 1.8},
    "SA-PIDNet": {"color": "#2ca02c", "marker": "^",  "ls": "--", "lw": 1.8},
}


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    codec    = rows[0].get("codec", "libx264")
    group    = "H.265/HEVC" if "x265" in codec else "H.264/AVC"
    title    = f"Bitrate vs mIoU — {group}"
    out_png  = csv_path.parent / "rd_miou.png"

    # ordered unique methods
    methods: List[str] = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)

    for method in methods:
        pts = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda r: float(r["bitrate_kbps"]),
        )
        xs = [float(r["bitrate_kbps"]) / 1000.0 for r in pts]   # → Mbps
        ys = [float(r["miou_pct"])               for r in pts]
        s  = _STYLE.get(method, {"color": "#9467bd", "marker": "D", "ls": ":", "lw": 1.6})

        ax.plot(xs, ys, s["ls"],
                color=s["color"], marker=s["marker"],
                markersize=8, linewidth=s["lw"], label=method)

        for r, x, y in zip(pts, xs, ys):
            crf_roi, crf_non = int(r["crf_roi"]), int(r["crf_non"])
            lbl = f"CRF{crf_roi}" if crf_roi == crf_non else f"({crf_roi},{crf_non})"
            ax.annotate(lbl, (x, y),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=s["color"], alpha=0.9)

    ax.set_xlabel("Bitrate (Mbps)", fontsize=12)
    ax.set_ylabel("mIoU (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
