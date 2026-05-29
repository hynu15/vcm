"""
Step 5: Evaluate reconstructed frames against original frames.
Computes PSNR, SSIM, SA-PSNR, SA-SSIM, bitrate and saves to CSV.

Usage:
    python evaluate.py --config ../../config.yaml \
        --original_dir   /data/frames \
        --reconstructed_dir /data/reconstructed/crf22_28 \
        --masks_dir      /data/masks \
        --crf_roi 22 --crf_nonroi 28 \
        --bitrate_kb 1234.5 \
        --output_csv /data/results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.new.utils.io import load_config, setup_logger, ensure_dirs, sorted_frame_paths


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return psnr(img1, img2, data_range=255)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, channel_axis=2, data_range=255)


def compute_sa_metric(img_orig: np.ndarray, img_recon: np.ndarray,
                      roi_mask: np.ndarray, fn):
    """Compute fn only on ROI pixels, returns (sa_roi, sa_nonroi)."""
    mask3 = roi_mask[..., None].astype(bool)
    roi_orig = np.where(mask3, img_orig, 0).astype(np.uint8)
    roi_recon = np.where(mask3, img_recon, 0).astype(np.uint8)
    nonroi_orig = np.where(~mask3, img_orig, 0).astype(np.uint8)
    nonroi_recon = np.where(~mask3, img_recon, 0).astype(np.uint8)
    return fn(roi_orig, roi_recon), fn(nonroi_orig, nonroi_recon)


def evaluate_dir(original_dir: str, reconstructed_dir: str, masks_dir: str,
                 roi_class_ids: list) -> dict:
    orig_paths = sorted_frame_paths(original_dir)
    metrics = {"psnr": [], "ssim": [], "sa_psnr_roi": [], "sa_psnr_nonroi": [],
               "sa_ssim_roi": [], "sa_ssim_nonroi": []}

    for idx, orig_path in enumerate(tqdm(orig_paths, desc="Evaluating")):
        recon_path = Path(reconstructed_dir) / f"{idx:06d}.png"
        mask_path = Path(masks_dir) / orig_path.name
        if not recon_path.exists() or not mask_path.exists():
            continue

        orig = cv2.imread(str(orig_path))
        recon = cv2.imread(str(recon_path))
        label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if orig is None or recon is None or label is None:
            continue

        recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

        roi_mask = np.isin(label, roi_class_ids).astype(np.uint8)

        metrics["psnr"].append(compute_psnr(orig, recon))
        metrics["ssim"].append(compute_ssim(orig, recon))
        sa_p_roi, sa_p_nonroi = compute_sa_metric(orig, recon, roi_mask, compute_psnr)
        sa_s_roi, sa_s_nonroi = compute_sa_metric(orig, recon, roi_mask, compute_ssim)
        metrics["sa_psnr_roi"].append(sa_p_roi)
        metrics["sa_psnr_nonroi"].append(sa_p_nonroi)
        metrics["sa_ssim_roi"].append(sa_s_roi)
        metrics["sa_ssim_nonroi"].append(sa_s_nonroi)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


CSV_FIELDS = [
    "method", "codec", "crf_roi", "crf_nonroi",
    "bitrate_kb", "psnr", "ssim",
    "sa_psnr_roi", "sa_psnr_nonroi",
    "sa_ssim_roi", "sa_ssim_nonroi",
]


def append_csv(csv_path: str, row: dict):
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC reconstructed frames")
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--reconstructed_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--crf_roi", type=int, required=True)
    parser.add_argument("--crf_nonroi", type=int, required=True)
    parser.add_argument("--bitrate_kb", type=float, default=0.0)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("evaluation")

    roi_class_ids = cfg["segmentation"]["roi_class_ids"]
    codec = cfg["compression"]["codec"]

    logger.info(f"Evaluating CRF pair ({args.crf_roi}, {args.crf_nonroi})")
    metrics = evaluate_dir(args.original_dir, args.reconstructed_dir,
                           args.masks_dir, roi_class_ids)

    row = {
        "method": "SAC",
        "codec": codec,
        "crf_roi": args.crf_roi,
        "crf_nonroi": args.crf_nonroi,
        "bitrate_kb": args.bitrate_kb,
        **metrics,
    }
    logger.info(f"Results: {row}")
    ensure_dirs(str(Path(args.output_csv).parent))
    append_csv(args.output_csv, row)
    logger.info(f"Saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
