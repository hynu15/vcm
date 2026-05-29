"""
Step 3: Compress ROI and non-ROI streams with FFmpeg.
Iterates over all CRF pairs defined in config and saves compressed videos.

Usage:
    python compress_streams.py --config ../../config.yaml \
        --roi_dir   /data/roi \
        --nonroi_dir /data/nonroi \
        --output_dir /data/compressed
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.new.utils.io import load_config, setup_logger, ensure_dirs


def make_concat_list(frames_dir: str, list_path: str) -> list:
    """Tạo file concat list cho FFmpeg, trả về danh sách frames đã sort."""
    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    with open(list_path, "w") as f:
        for p in frames:
            f.write(f"file '{p.resolve()}'\n")
    return frames


def run_ffmpeg(frames_dir: str, output_video: str, crf: int, cfg: dict) -> int:
    """Encode frames bằng FFmpeg concat demuxer (hỗ trợ mọi kiểu tên file).
    Trả về file size (bytes)."""
    codec = cfg["compression"]["codec"]
    framerate = cfg["compression"]["framerate"]
    preset = cfg["compression"]["preset"]
    gop = cfg["compression"]["gop"]
    pix_fmt = cfg["compression"]["pix_fmt"]

    list_path = str(Path(output_video).with_suffix(".txt"))
    make_concat_list(frames_dir, list_path)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-r", str(framerate),
        "-i", list_path,
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-g", str(gop),
        "-pix_fmt", pix_fmt,
        output_video,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Path(output_video).stat().st_size


def compress_pair(roi_dir: str, nonroi_dir: str, output_dir: str,
                  crf_roi: int, crf_nonroi: int, cfg: dict, logger) -> dict:
    tag = f"crf{crf_roi}_{crf_nonroi}"
    pair_dir = Path(output_dir) / tag
    ensure_dirs(str(pair_dir))

    roi_video = str(pair_dir / "roi.mp4")
    nonroi_video = str(pair_dir / "nonroi.mp4")

    logger.info(f"  Compressing ROI    CRF={crf_roi}  -> {roi_video}")
    roi_size = run_ffmpeg(roi_dir, roi_video, crf_roi, cfg)

    logger.info(f"  Compressing nonROI CRF={crf_nonroi} -> {nonroi_video}")
    nonroi_size = run_ffmpeg(nonroi_dir, nonroi_video, crf_nonroi, cfg)

    total_kb = (roi_size + nonroi_size) / 1024
    logger.info(f"  Total size: {total_kb:.1f} KB")

    return {
        "crf_roi": crf_roi,
        "crf_nonroi": crf_nonroi,
        "roi_video": roi_video,
        "nonroi_video": nonroi_video,
        "total_size_kb": total_kb,
    }


def main():
    parser = argparse.ArgumentParser(description="Compress ROI and non-ROI streams")
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--roi_dir", required=True)
    parser.add_argument("--nonroi_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("compression")
    ensure_dirs(args.output_dir)

    crf_pairs = cfg["compression"]["crf_pairs"]
    logger.info(f"CRF pairs: {crf_pairs}")

    for crf_roi, crf_nonroi in crf_pairs:
        logger.info(f"Processing pair (ROI={crf_roi}, nonROI={crf_nonroi})")
        compress_pair(args.roi_dir, args.nonroi_dir, args.output_dir,
                      crf_roi, crf_nonroi, cfg, logger)

    logger.info("Compression done.")


if __name__ == "__main__":
    main()
