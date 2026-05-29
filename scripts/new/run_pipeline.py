"""
Full SAC pipeline runner.
Chains all steps: segmentation -> stream generation -> compression ->
reconstruction -> evaluation, for every CRF pair in config.

Usage:
    python run_pipeline.py --config config.yaml \
        --frames_dir /data/frames \
        --skip_segmentation   # if masks already exist
"""

import argparse
import subprocess
import sys
from pathlib import Path

from utils.io import load_config, setup_logger, ensure_dirs


def run(cmd: list, logger):
    logger.info("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full SAC pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames_dir", required=True, help="Original input frames")
    parser.add_argument("--skip_segmentation", action="store_true",
                        help="Skip segmentation if masks already exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("pipeline", log_file="pipeline.log")

    base = Path(cfg["paths"]["output_root"])
    masks_dir = base / cfg["paths"]["masks_dir"]
    roi_dir = base / cfg["paths"]["roi_dir"]
    nonroi_dir = base / cfg["paths"]["nonroi_dir"]
    compressed_dir = base / cfg["paths"]["compressed_dir"]
    recon_dir = base / cfg["paths"]["reconstructed_dir"]
    results_csv = base / cfg["paths"]["results_csv"]

    ensure_dirs(str(masks_dir), str(roi_dir), str(nonroi_dir),
                str(compressed_dir), str(recon_dir))

    here = Path(__file__).parent

    # Step 1: Segmentation
    if not args.skip_segmentation:
        logger.info("=== Step 1: Segmentation ===")
        run([sys.executable, here / "segmentation" / "run_segmentation.py",
             "--config", args.config,
             "--frames_dir", args.frames_dir,
             "--output_dir", str(masks_dir)], logger)
    else:
        logger.info("=== Step 1: Segmentation SKIPPED ===")

    # Step 2: Stream generation
    logger.info("=== Step 2: Stream generation ===")
    run([sys.executable, here / "stream_generation" / "generate_streams.py",
         "--config", args.config,
         "--frames_dir", args.frames_dir,
         "--masks_dir", str(masks_dir),
         "--roi_dir", str(roi_dir),
         "--nonroi_dir", str(nonroi_dir)], logger)

    # Step 3–5: Compress, reconstruct, evaluate per CRF pair
    crf_pairs = cfg["compression"]["crf_pairs"]
    for crf_roi, crf_nonroi in crf_pairs:
        tag = f"crf{crf_roi}_{crf_nonroi}"
        logger.info(f"=== CRF pair ({crf_roi}, {crf_nonroi}) ===")

        pair_compressed = compressed_dir / tag
        pair_recon = recon_dir / tag

        # Step 3: Compression
        logger.info(f"  Step 3: Compression")
        run([sys.executable, here / "compression" / "compress_streams.py",
             "--config", args.config,
             "--roi_dir", str(roi_dir),
             "--nonroi_dir", str(nonroi_dir),
             "--output_dir", str(compressed_dir)], logger)

        # Step 4: Reconstruction
        logger.info(f"  Step 4: Reconstruction")
        roi_video = pair_compressed / "roi.mp4"
        nonroi_video = pair_compressed / "nonroi.mp4"
        run([sys.executable, here / "reconstruction" / "combine_streams.py",
             "--config", args.config,
             "--roi_video", str(roi_video),
             "--nonroi_video", str(nonroi_video),
             "--output_dir", str(pair_recon)], logger)

        # Step 5: Evaluation
        logger.info(f"  Step 5: Evaluation")
        total_kb = (roi_video.stat().st_size + nonroi_video.stat().st_size) / 1024
        run([sys.executable, here / "evaluation" / "evaluate.py",
             "--config", args.config,
             "--original_dir", args.frames_dir,
             "--reconstructed_dir", str(pair_recon),
             "--masks_dir", str(masks_dir),
             "--crf_roi", str(crf_roi),
             "--crf_nonroi", str(crf_nonroi),
             "--bitrate_kb", str(total_kb),
             "--output_csv", str(results_csv)], logger)

    logger.info(f"Pipeline complete. Results -> {results_csv}")


if __name__ == "__main__":
    main()
