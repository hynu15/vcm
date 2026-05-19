#!/usr/bin/env python3
"""Compose 4 frames from outputs/two_stream/test_smoke into one A4 figure.

Layout: 4 rows x 3 columns
    cols (left -> right): ORIG, IA (ROI), BA (non-ROI)
    rows: frame_00000 .. frame_00003
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "outputs" / "two_stream" / "test_smoke"
OUT_PATH = SMOKE_DIR / "composite_a4.png"

FRAME_INDICES = [0, 1, 2, 3]
COLS = [
    ("Original",  "ORIG"),
    ("ROI",       "IA"),
    ("Non-ROI",   "BA"),
]

A4_INCHES = (8.27, 6.5)  # ~ hơn nửa trang A4 chiều cao


def main() -> None:
    fig, axes = plt.subplots(
        nrows=len(FRAME_INDICES),
        ncols=len(COLS),
        figsize=A4_INCHES,
        dpi=200,
    )
    fig.patch.set_facecolor("white")

    for row, idx in enumerate(FRAME_INDICES):
        fname = f"frame_{idx:05d}.png"
        for col, (title, subdir) in enumerate(COLS):
            ax = axes[row, col]
            img_path = SMOKE_DIR / subdir / fname
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"Frame {idx}", fontsize=10, rotation=90,
                              labelpad=8, fontweight="bold")

    fig.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.01,
                        wspace=0.04, hspace=0.01)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
