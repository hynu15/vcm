#!/usr/bin/env python3
"""RD pipeline và biểu đồ với ROI IoU (class 0) làm accuracy metric.

Giống run_segmentation_rd_pipeline.py nhưng dùng IoU của class ROI
(class 0 = đường/xe/người) thay vì mIoU tổng thể.

Lý do: SAC được thiết kế để bảo toàn chất lượng vùng ROI,
nên ROI IoU phản ánh đúng hơn hiệu quả của SAC so với mIoU.

Chạy:
    python run_rd_roi_iou.py --split val --num-frames 20
    python run_rd_roi_iou.py --split test --cities strasbourg,ulm -n 30
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from run_segmentation_rd_pipeline import (
    CLASS_NAMES_4,
    IMAGE_ROOT_TEST,
    IMAGE_ROOT_VAL,
    LABEL_ROOT_TEST,
    LABEL_ROOT_VAL,
    MODEL_PATH_DEFAULT,
    NARROW_ROI_POINTS,
    OUTPUT_ROOT,
    OperatingPoint,
    REQUESTED_POINTS,
    bd_rate_and_acc,
    bitrate_mbps,
    build_roi_masks,
    build_roi_masks_narrow_gt,
    build_split_frames,
    build_transform,
    encode_two_streams,
    list_eval_pairs,
    load_model_auto,
    mean_iou,
    predict_masks,
    read_video_frames,
    to_np_label,
)

ROI_CLASS_ID = 0  # class 0 = ROI trong 4-class mapping


def plot_rd_roi_iou(
    baseline_df: pd.DataFrame,
    propose_df: pd.DataFrame,
    output_png: Path,
    bd_rate: Optional[float] = None,
    bd_acc: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    baseline_df = baseline_df.sort_values("bitrate_kbps")
    propose_df  = propose_df.sort_values("bitrate_kbps")

    ax.plot(
        baseline_df["bitrate_kbps"], baseline_df["roi_iou"],
        "o-", linewidth=2.5, markersize=8, color="steelblue", label="Baseline (Traditional)",
    )
    ax.plot(
        propose_df["bitrate_kbps"], propose_df["roi_iou"],
        "s--", linewidth=2.5, markersize=8, color="tomato", label="Propose (SAC)",
    )

    for _, row in baseline_df.iterrows():
        ax.annotate(
            f"QP{row['qp']}", (row["bitrate_kbps"], row["roi_iou"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=9, color="steelblue",
        )
    for _, row in propose_df.iterrows():
        ax.annotate(
            f"QP{row['qp']}*", (row["bitrate_kbps"], row["roi_iou"]),
            xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=9, color="tomato",
        )

    ax.set_xlabel("Bitrate (kbps)", fontsize=13)
    ax.set_ylabel("ROI IoU  (class 0: road/vehicle/person)", fontsize=13)
    ax.set_title("ROI IoU – Rate RD Curve on Decoded Video", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=12)

    if bd_rate is not None and bd_acc is not None:
        info = f"BD-Rate: {bd_rate:+.2f}%   BD-ROI-IoU: {bd_acc:+.4f}"
        ax.text(
            0.98, 0.04, info,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    print(f"RD curve saved: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RD pipeline với ROI IoU (class 0) thay vì mIoU")
    parser.add_argument("-n", "--num-frames", type=int, default=20,
                        help="Số frame dùng để encode và đánh giá (mặc định: 20)")
    parser.add_argument("--fps",     type=int,   default=30)
    parser.add_argument("--preset",  type=str,   default="slow")
    parser.add_argument("--split",   type=str,   default="val", choices=("val", "test"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Checkpoint model (mặc định: models/best_pidnet_l_4class.pth)")
    parser.add_argument("--num-classes", type=int, default=4, choices=(2, 4))
    parser.add_argument("--cities", type=str, default=None,
                        help="Lọc city, ví dụ: strasbourg,ulm")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--roi-mode", type=str, default="model", choices=("model", "narrow-gt"),
        help=(
            "'model' (mặc định): dùng model predictions (class 0 = broad ROI ~60%%). "
            "'narrow-gt': dùng GT labelIds, ROI = vehicles+pedestrians (IDs 24-33, ~10-15%%)."
        ),
    )
    parser.add_argument(
        "--crf-pairs", type=str, default=None,
        help="Ghi đè operating points bằng 'roi:non,...'. Ví dụ: '18:22,22:26,26:30,30:34'.",
    )
    args = parser.parse_args()

    # ── Operating points ───────────────────────────────────────────────────────
    if args.crf_pairs:
        op_points = []
        for pair_str in args.crf_pairs.split(","):
            roi_crf, non_crf = pair_str.strip().split(":")
            label = str((int(roi_crf) + int(non_crf)) // 2)
            op_points.append(OperatingPoint(label, int(roi_crf), int(non_crf)))
        print(f"Custom CRF pairs: {[(o.crf_roi, o.crf_non) for o in op_points]}")
    elif args.roi_mode == "narrow-gt":
        op_points = NARROW_ROI_POINTS
        print("Operating points: NARROW_ROI_POINTS (optimised for ~10% ROI)")
    else:
        op_points = REQUESTED_POINTS

    # ── Model ──────────────────────────────────────────────────────────────────
    model_path = Path(args.model_path) if args.model_path else MODEL_PATH_DEFAULT
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    image_root = IMAGE_ROOT_VAL  if args.split == "val" else IMAGE_ROOT_TEST
    label_root = LABEL_ROOT_VAL  if args.split == "val" else LABEL_ROOT_TEST
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    out_root = OUTPUT_ROOT.parent / "rd_roi_iou"
    out_root.mkdir(parents=True, exist_ok=True)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {model_path}")
    model, model_name, num_classes = load_model_auto(model_path, device, args.num_classes)
    print(f"Model: {model_name}  |  num_classes: {num_classes}  |  "
          f"classes: {CLASS_NAMES_4 if num_classes == 4 else ['ROI','non_ROI']}")
    transform = build_transform()

    # ── Data ───────────────────────────────────────────────────────────────────
    city_filter = [c.strip() for c in args.cities.split(",")] if args.cities else None
    if city_filter:
        print(f"City filter: {city_filter}")

    pairs = list_eval_pairs(image_root, label_root, args.num_frames, num_classes, city_filter)
    if not pairs:
        raise RuntimeError(f"No images found in {image_root}"
                           + (f" (cities={city_filter})" if city_filter else ""))

    images = [np.array(Image.open(p).convert("RGB")) for p, _ in pairs]
    gts    = [to_np_label(lp, num_classes) for _, lp in pairs]

    valid_indices = [i for i, gt in enumerate(gts) if int(gt.max()) > 0]
    skip_metric   = len(valid_indices) == 0
    if skip_metric:
        print("WARNING: Không có frame nào có GT label hợp lệ → skip ROI IoU.\n"
              "  → Thử: --split val  hoặc  --split test --cities strasbourg,ulm")
    else:
        invalid = len(gts) - len(valid_indices)
        print(f"GT: {len(valid_indices)}/{len(gts)} frames hợp lệ"
              + (f"  ({invalid} frames zeros bị loại)" if invalid else ""))

    # ── Encode / split frames ──────────────────────────────────────────────────
    if args.roi_mode == "narrow-gt":
        print("ROI mode: narrow-gt (vehicles+pedestrians, IDs 24-33)")
        roi_masks = build_roi_masks_narrow_gt(pairs, images)
    else:
        print("ROI mode: model (class 0 = broad catch-all ROI)")
        roi_masks = build_roi_masks(model, device, transform, images)
        ratios = [m.mean() * 100 for m in roi_masks]
        print(f"  Broad ROI ratio: mean={np.mean(ratios):.1f}%  "
              f"min={np.min(ratios):.1f}%  max={np.max(ratios):.1f}%")
    frame_dir = output_dir / "tmp_frames"
    build_split_frames(images, roi_masks, frame_dir)

    duration_sec  = len(images) / float(args.fps)
    baseline_rows: List[Dict] = []
    propose_rows:  List[Dict] = []
    detail_rows:   List[Dict] = []

    for op in op_points:
        combo_name = f"crf_{op.label.replace('/', '_')}"
        combo_dir  = output_dir / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        crf_trad = (op.crf_roi + op.crf_non) // 2
        encode_two_streams(
            frame_dir=frame_dir,
            output_dir=combo_dir,
            fps=args.fps,
            crf_roi=op.crf_roi,
            crf_non=op.crf_non,
            crf_trad=crf_trad,
            preset=args.preset,
        )
        roi_video  = combo_dir / "roi.mp4"
        non_video  = combo_dir / "nonroi.mp4"
        sac_video  = combo_dir / "sac_x265.mp4"
        trad_video = combo_dir / "traditional_x265.mp4"

        sac_bitrate  = bitrate_mbps(roi_video, duration_sec) + bitrate_mbps(non_video, duration_sec)
        trad_bitrate = bitrate_mbps(trad_video, duration_sec)

        if skip_metric:
            sac_roi_iou = trad_roi_iou = float("nan")
            sac_class_iou = trad_class_iou = {}
        else:
            sac_frames  = read_video_frames(sac_video,  len(images))
            trad_frames = read_video_frames(trad_video, len(images))

            sac_preds  = predict_masks(model, device, transform, sac_frames)
            trad_preds = predict_masks(model, device, transform, trad_frames)

            valid_gts        = [gts[i]        for i in valid_indices]
            valid_sac_preds  = [sac_preds[i]  for i in valid_indices]
            valid_trad_preds = [trad_preds[i] for i in valid_indices]

            _, sac_class_iou  = mean_iou(valid_sac_preds,  valid_gts, num_classes=num_classes)
            _, trad_class_iou = mean_iou(valid_trad_preds, valid_gts, num_classes=num_classes)

            sac_roi_iou  = sac_class_iou.get(ROI_CLASS_ID, 0.0)
            trad_roi_iou = trad_class_iou.get(ROI_CLASS_ID, 0.0)

        baseline_rows.append({
            "qp": op.label,
            "bitrate_kbps": trad_bitrate * 1000.0,
            "roi_iou": trad_roi_iou,
        })
        propose_rows.append({
            "qp": op.label,
            "bitrate_kbps": sac_bitrate * 1000.0,
            "roi_iou": sac_roi_iou,
        })
        detail_rows.append({
            "qp": op.label,
            "crf_roi": op.crf_roi, "crf_non": op.crf_non, "crf_trad": crf_trad,
            "trad_bitrate_mbps": trad_bitrate,  "sac_bitrate_mbps": sac_bitrate,
            "trad_roi_iou": trad_roi_iou,        "sac_roi_iou": sac_roi_iou,
            "delta_roi_iou": (sac_roi_iou - trad_roi_iou) if not skip_metric else float("nan"),
            "delta_bitrate_mbps": sac_bitrate - trad_bitrate,
            "trad_class_iou": json.dumps(trad_class_iou, ensure_ascii=False),
            "sac_class_iou":  json.dumps(sac_class_iou,  ensure_ascii=False),
        })

        if skip_metric:
            print(f"  {op.label}: trad {trad_bitrate:.3f} Mbps / ROI IoU N/A | "
                  f"sac {sac_bitrate:.3f} Mbps / ROI IoU N/A")
        else:
            delta = sac_roi_iou - trad_roi_iou
            print(f"  {op.label}: trad {trad_bitrate:.3f} Mbps / ROI IoU {trad_roi_iou:.4f} | "
                  f"sac {sac_bitrate:.3f} Mbps / ROI IoU {sac_roi_iou:.4f}  "
                  f"(Δ={delta:+.4f})")

    # ── BD metrics & plot ──────────────────────────────────────────────────────
    baseline_df = pd.DataFrame(baseline_rows).sort_values("qp")
    propose_df  = pd.DataFrame(propose_rows).sort_values("qp")
    detail_df   = pd.DataFrame(detail_rows).sort_values("qp")

    baseline_df.to_csv(output_dir / "baseline_roi_iou.csv", index=False)
    propose_df.to_csv( output_dir / "propose_roi_iou.csv",  index=False)
    detail_df.to_csv(  output_dir / "detail_roi_iou.csv",   index=False)

    bd_rate = bd_acc = overlap = None
    bd_error = None
    if skip_metric:
        bd_error = "ROI IoU bị bỏ qua (không có GT hợp lệ)"
    else:
        try:
            bl_for_bd = baseline_df.rename(columns={"roi_iou": "accuracy_mean"})
            pr_for_bd = propose_df.rename( columns={"roi_iou": "accuracy_mean"})
            bd_rate, bd_acc, overlap = bd_rate_and_acc(bl_for_bd, pr_for_bd)
        except RuntimeError as exc:
            bd_error = str(exc)
        plot_rd_roi_iou(baseline_df, propose_df,
                        output_dir / "rd_curve_roi_iou.png",
                        bd_rate, bd_acc)

    summary = {
        "model":        model_name,
        "model_path":   str(model_path),
        "num_classes":  num_classes,
        "metric":       "ROI IoU (class 0: road/vehicle/person)",
        "split":        args.split,
        "num_frames":   len(images),
        "valid_frames": len(valid_indices),
        "bd_rate_percent":       bd_rate,
        "bd_roi_iou":            bd_acc,
        "bitrate_overlap_kbps":  overlap,
        "bd_error":              bd_error,
        "baseline_mean_roi_iou": float(np.nanmean(baseline_df["roi_iou"])),
        "propose_mean_roi_iou":  float(np.nanmean(propose_df["roi_iou"])),
    }
    (output_dir / "summary_roi_iou.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n── Baseline ──")
    print(baseline_df.to_string(index=False))
    print("\n── Propose ──")
    print(propose_df.to_string(index=False))
    if bd_error:
        print(f"\nBD metrics error: {bd_error}")
    else:
        print(f"\nBD-Rate    : {bd_rate:+.2f}%")
        print(f"BD-ROI-IoU : {bd_acc:+.4f}")
    print(f"\nOutput: {output_dir}")

    if not args.keep_artifacts:
        shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
