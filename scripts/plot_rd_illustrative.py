"""ILLUSTRATIVE preview of a target SAC-vs-baseline RD curve.

PURPOSE
    Hình minh hoạ để xem layout / hình dạng đường RD lý tưởng MUỐN đạt được sau
    khi tái hiện paper xong. KHÔNG đo từ dữ liệu thật — tất cả số đều sinh
    bằng công thức có tham số bên dưới.

ABSOLUTELY NOT
    - Không bao giờ dùng output của script này trong luận văn / bài báo / slide
      kết quả. Nó được vẽ với watermark "SYNTHETIC PREVIEW" để không thể nhầm.
    - Không lưu output đè lên `outputs/segmentation_bd/` (nơi pipeline thật ghi
      `baseline_real.csv`, `propose_real.csv`). File ở đây luôn có tiền tố
      `SYNTHETIC_` và nằm trong thư mục riêng.

USAGE
    python scripts/plot_rd_illustrative.py
    python scripts/plot_rd_illustrative.py --delta-miou 0.015 --out-dir /tmp/preview
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def synthetic_curves(qps, base_bitrate, base_miou, delta_miou_pct, jitter):
    """Sinh số tổng hợp với hình dạng RD giảm dần khi QP tăng.

    base_bitrate, base_miou: array dài bằng qps — đường baseline lý tưởng.
    delta_miou_pct: nhỉnh hơn baseline bao nhiêu (tỉ lệ thập phân, vd 0.015 = 1.5%).
    jitter: nhiễu nhỏ để các điểm không quá đều.
    """
    rng = np.random.default_rng(7)
    # SAC bitrate "gần thẳng hàng" với baseline → cùng giá trị + nhiễu nhỏ ±2%
    sac_bitrate = base_bitrate * (1 + rng.uniform(-0.02, 0.02, size=len(qps)))
    # mIoU SAC = baseline + delta, delta thay đổi nhẹ theo QP (cao hơn ở QP thấp)
    qp_norm = (np.array(qps) - min(qps)) / (max(qps) - min(qps) + 1e-9)
    delta_vec = delta_miou_pct * (1.2 - 0.4 * qp_norm)  # 1.0×δ … 1.2×δ
    sac_miou = base_miou + delta_vec + rng.uniform(-jitter, jitter, size=len(qps))
    return sac_bitrate, sac_miou


def write_csv(path: Path, qps, bitrate_kbps, miou):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# SYNTHETIC PREVIEW — DO NOT USE IN REPORTS"])
        w.writerow(["qp", "bitrate_kbps", "accuracy_mean"])
        for q, b, m in zip(qps, bitrate_kbps, miou):
            w.writerow([q, f"{b:.3f}", f"{m:.4f}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qps", type=str, default="22,27,32,37",
                    help="Danh sách QP nhãn (cách nhau dấu phẩy)")
    ap.add_argument("--baseline-bitrate-kbps", type=str,
                    default="9000,5500,3200,1900",
                    help="Bitrate baseline tại mỗi QP (kbps)")
    ap.add_argument("--baseline-miou", type=str,
                    default="0.815,0.795,0.770,0.735",
                    help="mIoU baseline tại mỗi QP")
    ap.add_argument("--delta-miou", type=float, default=0.015,
                    help="SAC mIoU nhỉnh hơn baseline trung bình bao nhiêu "
                         "(thập phân, 0.015 = 1.5%%)")
    ap.add_argument("--jitter", type=float, default=0.0015,
                    help="Nhiễu mIoU để điểm không nằm đều tuyệt đối")
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJECT_ROOT / "outputs" / "synthetic_preview"),
                    help="Thư mục lưu (sẽ tạo nếu chưa có)")
    args = ap.parse_args()

    qps = [int(x) for x in args.qps.split(",")]
    base_bitrate = np.array([float(x) for x in args.baseline_bitrate_kbps.split(",")])
    base_miou    = np.array([float(x) for x in args.baseline_miou.split(",")])
    if not (len(qps) == len(base_bitrate) == len(base_miou)):
        raise SystemExit("qps, baseline-bitrate, baseline-miou phải cùng độ dài")

    sac_bitrate, sac_miou = synthetic_curves(qps, base_bitrate, base_miou,
                                             args.delta_miou, args.jitter)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "SYNTHETIC_baseline.csv", qps, base_bitrate, base_miou)
    write_csv(out_dir / "SYNTHETIC_propose.csv",  qps, sac_bitrate,  sac_miou)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    ax.plot(base_bitrate, base_miou, "o-", lw=2.5, ms=8, label="Baseline (illustrative)")
    ax.plot(sac_bitrate,  sac_miou,  "s--", lw=2.5, ms=8, label="Propose SAC (illustrative)")
    for q, b, m in zip(qps, base_bitrate, base_miou):
        ax.annotate(f"QP{q}", (b, m), xytext=(0, 10), textcoords="offset points", ha="center")
    for q, b, m in zip(qps, sac_bitrate, sac_miou):
        ax.annotate(f"QP{q}*", (b, m), xytext=(0, -14), textcoords="offset points", ha="center")
    ax.set_xlabel("Rate (kbps)")
    ax.set_ylabel("Acc (mIoU)")
    ax.set_title(f"RD curve — ILLUSTRATIVE PREVIEW (Δ mIoU ≈ +{args.delta_miou*100:.1f}%)")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig_path = out_dir / "SYNTHETIC_rd_curve.png"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"[OK] Synthetic CSVs + PNG → {out_dir}")
    print("     SYNTHETIC_baseline.csv")
    print("     SYNTHETIC_propose.csv")
    print(f"     {fig_path.name}")
    print("\nLưu ý: file này CHỈ để xem layout. Pipeline số liệu thật vẫn là "
          "scripts/run_segmentation_rd_pipeline.py.")


if __name__ == "__main__":
    main()
