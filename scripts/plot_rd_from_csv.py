"""Vẽ RD curve (bitrate vs iIoU) từ `combined_rd_detail.csv` đã có sẵn.

iIoU = IoU class 0 (ROI), parse từ JSON trong cột `*_class_iou`.
Cũng vẽ kèm bản mIoU để đối chiếu.

Mặc định đọc file ở:
  outputs/sac_models_comparison/run_20260516_140640/combined_rd_detail.csv
Output PNG ghi NGAY trong cùng thư mục với CSV.

Ví dụ:
  python scripts/plot_rd_from_csv.py
  python scripts/plot_rd_from_csv.py --csv outputs/sac_models_comparison/run_xxx/combined_rd_detail.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV  = (PROJECT_ROOT / "outputs" / "sac_models_comparison"
                / "run_20260516_140640" / "combined_rd_detail.csv")

METHODS = [
    ("h265",       "H.265 baseline",    "o-",  "#4c72b0"),
    ("sac_ccnet",  "SAC-CCNet_4class",  "s--", "#dd8452"),
    ("sac_pidnet", "SAC-PIDNet_4class", "^--", "#55a868"),
]


def extract_class_iou(df: pd.DataFrame, method: str, cls: int) -> pd.Series:
    """Parse cột `<method>_class_iou` (JSON string) → IoU của class `cls`."""
    col = f"{method}_class_iou"
    def _get(s):
        try:
            d = json.loads(s)
            # JSON key có thể là "0" (string) hoặc 0 (int)
            return float(d.get(str(cls), d.get(cls)))
        except (TypeError, ValueError, json.JSONDecodeError):
            return float("nan")
    return df[col].map(_get)


def plot_rd(df: pd.DataFrame, y_col_fmt: str, y_label: str, title: str, out_png: Path,
            roi_class: int = 0):
    """y_col_fmt: format string với placeholder {method}, vd 'iiou_{method}' hoặc '{method}_miou'.
    Nếu cột không tồn tại trong df thì sẽ tự parse từ class_iou JSON (chỉ với iiou)."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)

    for method, label, fmt, color in METHODS:
        # X: bitrate kbps
        if method == "h265":
            bitrate_kbps = df["h265_bitrate_mbps"] * 1000
        else:
            bitrate_kbps = df[f"{method}_bitrate_kbps"]

        # Y: lấy theo format hoặc fallback parse từ class_iou
        y_col = y_col_fmt.format(method=method)
        if y_col in df.columns:
            y = df[y_col]
        else:
            # fallback: parse từ class_iou JSON (giả định iiou = class 0)
            y = extract_class_iou(df, method, roi_class)

        # Sắp xếp theo bitrate để đường vẽ liền
        order = bitrate_kbps.argsort()
        x_sorted = bitrate_kbps.iloc[order].to_numpy()
        y_sorted = y.iloc[order].to_numpy()
        qp_sorted = df["qp"].iloc[order].to_numpy()

        ax.plot(x_sorted, y_sorted, fmt, linewidth=2.5, markersize=8,
                color=color, label=label)
        for xi, yi, qi in zip(x_sorted, y_sorted, qp_sorted):
            ax.annotate(f"{qi}", (xi, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"  saved → {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                    help=f"đường dẫn combined_rd_detail.csv (default: {DEFAULT_CSV})")
    ap.add_argument("--roi-class", type=int, default=0,
                    help="class id cho iIoU (default 0 = ROI)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    out_dir = csv_path.parent

    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} rows from {csv_path}")

    # 1) Bitrate vs iIoU  (lấy class 0 từ JSON class_iou)
    plot_rd(
        df,
        y_col_fmt="{method}_iiou",   # nếu có sẵn cột, dùng luôn; nếu không sẽ parse từ class_iou
        y_label=f"iIoU (ROI, class {args.roi_class})",
        title=("Rate–iIoU RD Curve: H.265 vs SAC-CCNet vs SAC-PIDNet\n"
               "(evaluated by PIDNet_L_4class)"),
        out_png=out_dir / "rd_curve_iiou.png",
        roi_class=args.roi_class,
    )

    # 2) Bitrate vs mIoU (để đối chiếu)
    if all(f"{m}_miou" in df.columns for m, _, _, _ in METHODS):
        plot_rd(
            df,
            y_col_fmt="{method}_miou",
            y_label="mIoU (4-class)",
            title=("Rate–mIoU RD Curve: H.265 vs SAC-CCNet vs SAC-PIDNet\n"
                   "(evaluated by PIDNet_L_4class)"),
            out_png=out_dir / "rd_curve_miou.png",
        )


if __name__ == "__main__":
    main()
