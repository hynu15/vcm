"""Khối S3 (paper §III-D, Eq.5-7): nén hai mức bằng FFmpeg + giải nén ghép lại.

Hỗ trợ:
- `compress_traditional(frames, codec, crf)`: nén cả frame với 1 CRF chung (H.264/H.265 baseline).
- `compress_sac(frames, roi_masks, codec, crf_roi, crf_non)`: tách 2 luồng, nén song song
  với CRF khác nhau, giải nén, ghép lại bằng `np.where(mask, roi_dec, non_dec)`.

`frames`: list các np.uint8 (H,W,3). Trả về list frame tái tạo cùng kích thước.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {p.returncode}):\n{' '.join(cmd)}\n{p.stderr[-2000:]}")


def _save_pngs(frames: list[np.ndarray], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(dst_dir / f"frame_{i:05d}.png")


def _read_video(video_path: Path) -> list[np.ndarray]:
    """Giải nén bằng ffmpeg → đọc PNG ra list (H,W,3) uint8."""
    out_dir = video_path.parent / (video_path.stem + "_decoded")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _run_ffmpeg([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        str(out_dir / "frame_%05d.png"),
    ])
    frames = []
    for f in sorted(out_dir.glob("frame_*.png")):
        frames.append(np.array(Image.open(f).convert("RGB")))
    return frames


def _encode(in_dir: Path, codec: str, crf: int, out_path: Path,
            framerate: int = 30, keyint: int | None = 30, preset: str = "medium") -> int:
    """Encode list PNG → 1 video; trả về byte size của file.

    framerate=30, keyint=30 mặc định phù hợp pipeline eval slideshow hiện tại
    (mỗi frame test = 1 scene khác nhau). Nếu input là video sequence thực
    (leftImg8bit_sequence — 30 frame liên tục cùng scene) thì nên đặt
    framerate=17, keyint=None để codec tận dụng inter-prediction.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(framerate),
        "-i", str(in_dir / "frame_%05d.png"),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
    ]
    if keyint is not None and keyint > 0:
        params_flag = "-x264-params" if codec == "libx264" else "-x265-params"
        cmd += [params_flag, f"keyint={keyint}:min-keyint={keyint}"]
    cmd.append(str(out_path))
    _run_ffmpeg(cmd)
    return out_path.stat().st_size


def compress_traditional(frames: list[np.ndarray], codec: str, crf: int,
                         work_dir: Path | None = None, framerate: int = 30,
                         keyint: int | None = 30,
                         ) -> tuple[list[np.ndarray], int]:
    """Nén baseline cả frame với 1 CRF chung. Trả về (frames_decoded, bytes)."""
    tmp_ctx = tempfile.TemporaryDirectory(prefix="sac_trad_") if work_dir is None else None
    base = Path(tmp_ctx.name) if tmp_ctx else work_dir
    base.mkdir(parents=True, exist_ok=True)
    try:
        png_dir = base / "frames"
        _save_pngs(frames, png_dir)
        vid = base / "out.mp4"
        size = _encode(png_dir, codec, crf, vid, framerate=framerate, keyint=keyint)
        dec = _read_video(vid)
        return dec, size
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def compress_sac(frames: list[np.ndarray], roi_masks: list[np.ndarray], codec: str,
                 crf_roi: int, crf_non: int, work_dir: Path | None = None,
                 framerate: int = 30, keyint: int | None = 30,
                 ) -> tuple[list[np.ndarray], int, int]:
    """Pipeline SAC: tách luồng → nén 2 nhánh → giải nén → ghép. Trả về (frames, b_roi, b_non)."""
    assert len(frames) == len(roi_masks), "frames và masks phải cùng số lượng"
    tmp_ctx = tempfile.TemporaryDirectory(prefix="sac_") if work_dir is None else None
    base = Path(tmp_ctx.name) if tmp_ctx else work_dir
    base.mkdir(parents=True, exist_ok=True)
    try:
        roi_dir = base / "stream_roi"
        non_dir = base / "stream_non"
        roi_dir.mkdir(parents=True, exist_ok=True)
        non_dir.mkdir(parents=True, exist_ok=True)
        for i, (f, m) in enumerate(zip(frames, roi_masks)):
            mi = m[..., None]
            Si = (f * mi).astype(np.uint8)
            Sn = (f * (1 - mi)).astype(np.uint8)
            Image.fromarray(Si).save(roi_dir / f"frame_{i:05d}.png")
            Image.fromarray(Sn).save(non_dir / f"frame_{i:05d}.png")

        vid_roi = base / "out_roi.mp4"
        vid_non = base / "out_non.mp4"
        b_roi = _encode(roi_dir, codec, crf_roi, vid_roi, framerate=framerate, keyint=keyint)
        b_non = _encode(non_dir, codec, crf_non, vid_non, framerate=framerate, keyint=keyint)
        dec_roi = _read_video(vid_roi)
        dec_non = _read_video(vid_non)

        # Ghép bằng mask gốc (xem CLAUDE.md §8.4 – đảm bảo không bị nhiễu codec ở pixel = 0)
        out = []
        for f_roi, f_non, m in zip(dec_roi, dec_non, roi_masks):
            mi3 = m[..., None].astype(bool)
            recombined = np.where(mi3, f_roi, f_non).astype(np.uint8)
            out.append(recombined)
        return out, b_roi, b_non
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
