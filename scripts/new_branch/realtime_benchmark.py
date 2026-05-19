"""Benchmark realtime end-to-end pipeline cho 2 model phân đoạn.

So sánh CCNet 4-class vs PIDNet-L 4-class trên 4 cấu hình codec
(H.264 / SA-X264 / H.265 / SA-X265). Đo riêng từng khối:

    S1  – segmentation forward (model-specific)
    S2  – sinh ROI mask + macroblock 16x16
    S3  – encode (SAC: hai luồng song song qua ThreadPoolExecutor;
                 baseline: một luồng duy nhất)
    DEC – decode + combine

Output:
    {out_dir}/per_frame.csv         – mỗi dòng = (frame, model, codec, t_*)
    {out_dir}/summary.csv           – mean ± std per (model, codec)
    {out_dir}/summary.json          – cùng thông tin, dạng JSON
    Bảng tóm tắt in ra stdout

Chạy:
    cd /home/huy/sac_project
    conda activate sac
    python -m scripts.new_branch.realtime_benchmark \
        --num-frames 50 --warmup 5 \
        --ccnet-ckpt models/best_ccnet_4class.pth \
        --pidnet-ckpt models/best_pidnet_l_4class.pth

Trên RunPod (RTX 4090): nhớ kiểm tra ffmpeg đa luồng đã bật:
    ffmpeg -filters | head -5
    nproc
"""
from __future__ import annotations

if __package__ in (None, ""):
    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "scripts.new_branch"

import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .compression import _encode, _read_video, _save_pngs
from .dataset import _list_split, _normalize, _rgb_to_gray3
from .model import build_pidnet_l, build_segnet
from .paths import (CRF_CONFIGS, EVAL_DIR, FRAME_HW, NUM_CLASSES,
                    PIDNET_L_CHECKPOINT, SCRIPTS_ROOT, SEG_INPUT_HW)
from .stream_separation import build_roi_mask

# CCNet4Class (kiến trúc thay thế dùng cho best_ccnet_4class.pth):
# ResNet-101 chuẩn torchvision + grayscale conversion BÊN TRONG forward.
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from new_feature.ccnet_4class import CCNet4Class  # noqa: E402


CODEC_ORDER = ["H264", "SA-X264", "H265", "SA-X265"]


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed() -> Iterator[list[float]]:
    """Thời gian giữa __enter__ và __exit__; cuda.synchronize() được gọi 2 đầu."""
    cuda_sync()
    t0 = time.perf_counter()
    box: list[float] = [0.0]
    try:
        yield box
    finally:
        cuda_sync()
        box[0] = time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_sd(sd):
    if isinstance(sd, dict) and "model_state_dict" in sd:
        return sd["model_state_dict"]
    return sd


def load_ccnet(ckpt_path: Path, device: torch.device, arch: str):
    """arch in {'4class','sac'}.

    - '4class': dùng CCNet4Class (ResNet-101 torchvision chuẩn, conv1 7x7,
                grayscale conversion BÊN TRONG forward).
                → gray_input=False khi gọi predict.
    - 'sac':    dùng Seg_Model (CCNet gốc, deep stem 3xConv3, layer1 in=128).
                → gray_input=True khi gọi predict.
    """
    if arch == "4class":
        model = CCNet4Class(num_classes=NUM_CLASSES, pretrained=False)
    elif arch == "sac":
        model = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=False)
    else:
        raise ValueError(f"Unknown ccnet arch: {arch!r}")
    sd = _unwrap_sd(torch.load(ckpt_path, map_location=device))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [ccnet/{arch}] missing={len(missing)} unexpected={len(unexpected)}",
              file=sys.stderr)
    return model.to(device).eval()


def load_pidnet(ckpt_path: Path, device: torch.device):
    model = build_pidnet_l(num_classes=NUM_CLASSES)
    sd = _unwrap_sd(torch.load(ckpt_path, map_location=device))
    # Checkpoint PIDNet lưu RAW (không prefix "backbone.") → load vào .backbone
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [pidnet] missing={len(missing)} unexpected={len(unexpected)}",
              file=sys.stderr)
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Forward helpers (mỗi model có preprocessing khác)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_seg_lr(model, img_rgb_full: np.ndarray, device: torch.device,
                   gray_input: bool) -> np.ndarray:
    """Trả về seg lowres (H', W') uint8 ở kích thước SEG_INPUT_HW."""
    lr = np.array(Image.fromarray(img_rgb_full).resize(
        (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.BILINEAR))
    inp = _rgb_to_gray3(lr) if gray_input else lr
    t = _normalize(inp).unsqueeze(0).to(device)
    out = model(t)
    logits = out[0] if isinstance(out, (list, tuple)) else out
    logits = F.interpolate(logits.float(), size=SEG_INPUT_HW,
                           mode="bilinear", align_corners=True)
    return logits.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Parallel encode/decode cho SAC
# ─────────────────────────────────────────────────────────────────────────────

def encode_two_streams_parallel(
    roi_png_dir: Path, non_png_dir: Path,
    codec: str, crf_roi: int, crf_non: int,
    out_roi: Path, out_non: Path,
    framerate: int, keyint: int | None,
) -> tuple[int, int]:
    """Encode hai luồng SONG SONG; trả về (bytes_roi, bytes_non).

    Tổng thời gian gọi ≈ max(t_roi, t_non) vì ffmpeg chạy trong subprocess
    (GIL được giải phóng khi block I/O).
    """
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_r = ex.submit(_encode, roi_png_dir, codec, crf_roi, out_roi,
                        framerate, keyint)
        f_n = ex.submit(_encode, non_png_dir, codec, crf_non, out_non,
                        framerate, keyint)
        b_r = f_r.result()
        b_n = f_n.result()
    return b_r, b_n


def decode_two_streams_parallel(out_roi: Path, out_non: Path
                                ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_r = ex.submit(_read_video, out_roi)
        f_n = ex.submit(_read_video, out_non)
        dec_r = f_r.result()
        dec_n = f_n.result()
    return dec_r, dec_n


# ─────────────────────────────────────────────────────────────────────────────
# Một lượt benchmark cho 1 frame, 1 model, 1 codec config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    frame_idx: int
    model: str
    codec: str
    crf_roi: int
    crf_non: int
    t_s1: float
    t_s2: float
    t_s3: float
    t_decode_combine: float
    t_total: float
    bytes_total: int


def bench_one_frame(
    frame: np.ndarray,
    model_name: str, model, gray_input: bool,
    codec_name: str, cfg: dict,
    work_dir: Path,
    device: torch.device,
    framerate: int = 30,
    keyint: int | None = 30,
) -> FrameRecord:
    codec = cfg["codec"]
    crf_r, crf_n = cfg["crf_roi"], cfg["crf_non"]
    is_sac = (crf_r != crf_n)

    # ── S1: segmentation ────────────────────────────────────────────────────
    with timed() as ts1:
        seg_lr = predict_seg_lr(model, frame, device, gray_input=gray_input)

    # ── S2: stream separation ───────────────────────────────────────────────
    with timed() as ts2:
        roi_mask = build_roi_mask(seg_lr, target_hw=FRAME_HW)

    # Chuẩn bị thư mục PNG (KHÔNG tính vào S3 — đây là I/O pipeline-level)
    work_dir.mkdir(parents=True, exist_ok=True)
    if is_sac:
        roi_dir = work_dir / "roi"
        non_dir = work_dir / "non"
        roi_dir.mkdir(parents=True, exist_ok=True)
        non_dir.mkdir(parents=True, exist_ok=True)
        m3 = roi_mask[..., None]
        Si = (frame * m3).astype(np.uint8)
        Sn = (frame * (1 - m3)).astype(np.uint8)
        Image.fromarray(Si).save(roi_dir / "frame_00000.png")
        Image.fromarray(Sn).save(non_dir / "frame_00000.png")
        vid_roi = work_dir / "out_roi.mp4"
        vid_non = work_dir / "out_non.mp4"

        # ── S3: encode hai luồng song song ──────────────────────────────────
        with timed() as ts3:
            b_r, b_n = encode_two_streams_parallel(
                roi_dir, non_dir, codec, crf_r, crf_n,
                vid_roi, vid_non, framerate, keyint)
        bytes_total = b_r + b_n

        # ── Decode + Combine ────────────────────────────────────────────────
        with timed() as tdc:
            dec_r, dec_n = decode_two_streams_parallel(vid_roi, vid_non)
            m3b = roi_mask[..., None].astype(bool)
            _ = np.where(m3b, dec_r[0], dec_n[0]).astype(np.uint8)

    else:
        # Baseline: 1 luồng
        png_dir = work_dir / "trad"
        png_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(png_dir / "frame_00000.png")
        vid = work_dir / "out_trad.mp4"

        with timed() as ts3:
            bytes_total = _encode(png_dir, codec, crf_r, vid, framerate, keyint)

        with timed() as tdc:
            _ = _read_video(vid)

    t_total = ts1[0] + ts2[0] + ts3[0] + tdc[0]
    return FrameRecord(
        frame_idx=-1, model=model_name, codec=codec_name,
        crf_roi=crf_r, crf_non=crf_n,
        t_s1=ts1[0], t_s2=ts2[0], t_s3=ts3[0],
        t_decode_combine=tdc[0],
        t_total=t_total, bytes_total=bytes_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggRow:
    model: str
    codec: str
    n: int
    s1_mean: float
    s2_mean: float
    s3_mean: float
    dec_mean: float
    total_mean: float
    total_std: float
    total_p50: float
    total_p95: float
    fps: float
    bytes_mean: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def aggregate(records: list[FrameRecord]) -> list[AggRow]:
    keys = sorted({(r.model, r.codec) for r in records})
    out: list[AggRow] = []
    for model, codec in keys:
        rs = [r for r in records if r.model == model and r.codec == codec]
        totals = [r.t_total for r in rs]
        out.append(AggRow(
            model=model, codec=codec, n=len(rs),
            s1_mean=statistics.fmean(r.t_s1 for r in rs),
            s2_mean=statistics.fmean(r.t_s2 for r in rs),
            s3_mean=statistics.fmean(r.t_s3 for r in rs),
            dec_mean=statistics.fmean(r.t_decode_combine for r in rs),
            total_mean=statistics.fmean(totals),
            total_std=statistics.pstdev(totals) if len(totals) > 1 else 0.0,
            total_p50=statistics.median(totals),
            total_p95=float(np.percentile(totals, 95)) if totals else 0.0,
            fps=1.0 / statistics.fmean(totals),
            bytes_mean=statistics.fmean(r.bytes_total for r in rs),
        ))
    return out


def print_summary_table(rows: list[AggRow]) -> None:
    hdr = (f"{'Model':<8} {'Codec':<10} "
           f"{'S1':>7} {'S2':>7} {'S3':>7} {'Dec':>7} "
           f"{'Total':>8} {'±std':>7} {'p95':>7} {'FPS':>6} {'kB':>8}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.model:<8} {r.codec:<10} "
              f"{r.s1_mean*1000:>6.1f}m {r.s2_mean*1000:>6.1f}m "
              f"{r.s3_mean*1000:>6.1f}m {r.dec_mean*1000:>6.1f}m "
              f"{r.total_mean*1000:>7.1f}m {r.total_std*1000:>6.1f}m "
              f"{r.total_p95*1000:>6.1f}m "
              f"{r.fps:>5.2f} {r.bytes_mean/1024:>7.1f}")
    print()


def save_outputs(records: list[FrameRecord], rows: list[AggRow], out_dir: Path,
                 meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "per_frame.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "model", "codec", "crf_roi", "crf_non",
                    "t_s1", "t_s2", "t_s3", "t_decode_combine",
                    "t_total", "bytes_total"])
        for r in records:
            w.writerow([r.frame_idx, r.model, r.codec, r.crf_roi, r.crf_non,
                        f"{r.t_s1:.6f}", f"{r.t_s2:.6f}", f"{r.t_s3:.6f}",
                        f"{r.t_decode_combine:.6f}",
                        f"{r.t_total:.6f}", r.bytes_total])

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "codec", "n",
                    "s1_mean_ms", "s2_mean_ms", "s3_mean_ms", "dec_mean_ms",
                    "total_mean_ms", "total_std_ms", "total_p50_ms",
                    "total_p95_ms", "fps", "bytes_mean"])
        for r in rows:
            w.writerow([r.model, r.codec, r.n,
                        f"{r.s1_mean*1000:.3f}", f"{r.s2_mean*1000:.3f}",
                        f"{r.s3_mean*1000:.3f}", f"{r.dec_mean*1000:.3f}",
                        f"{r.total_mean*1000:.3f}",
                        f"{r.total_std*1000:.3f}",
                        f"{r.total_p50*1000:.3f}",
                        f"{r.total_p95*1000:.3f}",
                        f"{r.fps:.3f}", f"{r.bytes_mean:.1f}"])

    summary = {
        "meta": meta,
        "rows": [r.as_dict() for r in rows],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-frames", type=int, default=50,
                    help="Số frame để đo (sau warm-up). Default 50.")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Số frame warm-up bỏ qua khỏi thống kê.")
    ap.add_argument("--split", default="test",
                    choices=("train", "val", "test"))
    ap.add_argument("--ccnet-ckpt", type=Path,
                    default=Path("models/best_ccnet_4class.pth"))
    ap.add_argument("--ccnet-arch", choices=("4class", "sac"), default="4class",
                    help=("Kiến trúc CCNet: '4class' cho best_ccnet_4class.pth "
                          "(ResNet-101 torchvision + grayscale bên trong); "
                          "'sac' cho best_ccnet_sac.pth (Seg_Model deep stem)."))
    ap.add_argument("--pidnet-ckpt", type=Path,
                    default=PIDNET_L_CHECKPOINT)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Mặc định outputs/new_branch/eval/realtime_{timestamp}.")
    ap.add_argument("--codecs", nargs="+", default=CODEC_ORDER,
                    choices=CODEC_ORDER,
                    help="Tập codec cấu hình cần đo.")
    ap.add_argument("--models", nargs="+", default=["ccnet", "pidnet"],
                    choices=["ccnet", "pidnet"])
    ap.add_argument("--framerate", type=int, default=30)
    ap.add_argument("--keyint", type=int, default=30,
                    help="GOP size; đặt 0 để bỏ ràng buộc keyint.")
    ap.add_argument("--device", default=None,
                    help="cuda|cpu; mặc định auto.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg không có trong PATH. Trên RunPod chạy:\n"
                 "    apt-get update && apt-get install -y ffmpeg")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  |  PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    keyint = args.keyint if args.keyint > 0 else None

    # Load các model được yêu cầu
    print("\nLoading models …")
    models: dict[str, tuple[torch.nn.Module, bool]] = {}
    if "ccnet" in args.models:
        if not args.ccnet_ckpt.exists():
            raise FileNotFoundError(f"CCNet checkpoint not found: {args.ccnet_ckpt}")
        # arch '4class' tự grayscale bên trong → gray_input=False;
        # arch 'sac' cần grayscale từ ngoài → gray_input=True.
        cc_gray = (args.ccnet_arch == "sac")
        models["ccnet"] = (load_ccnet(args.ccnet_ckpt, device, args.ccnet_arch),
                           cc_gray)
        print(f"  ccnet/{args.ccnet_arch}  ← {args.ccnet_ckpt.name}  "
              f"(gray_input={cc_gray})")
    if "pidnet" in args.models:
        if not args.pidnet_ckpt.exists():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {args.pidnet_ckpt}")
        models["pidnet"] = (load_pidnet(args.pidnet_ckpt, device), False)
        print(f"  pidnet ← {args.pidnet_ckpt.name}  (gray_input=False)")

    # Đọc danh sách frame
    pairs = _list_split(args.split)
    total_needed = args.num_frames + args.warmup
    if len(pairs) < total_needed:
        raise RuntimeError(
            f"Split {args.split!r} chỉ có {len(pairs)} frame, "
            f"cần {total_needed} (num_frames + warmup).")
    selected = pairs[:total_needed]
    print(f"\nFrames: warmup={args.warmup}, measured={args.num_frames}, "
          f"split={args.split}")

    # Đọc tất cả frame vào RAM trước để loại biến thiên I/O đĩa
    frames: list[np.ndarray] = []
    for img_path, _ in selected:
        arr = np.array(Image.open(img_path).convert("RGB"))
        if arr.shape[:2] != FRAME_HW:
            arr = np.array(Image.fromarray(arr).resize(
                (FRAME_HW[1], FRAME_HW[0]), Image.BILINEAR))
        frames.append(arr)

    # Setup output
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = EVAL_DIR / f"realtime_{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out_dir}")

    records: list[FrameRecord] = []

    # Tạo thư mục tạm cho ffmpeg (mỗi run riêng biệt để không đè file)
    with tempfile.TemporaryDirectory(prefix="rtbench_") as tmp:
        tmp_path = Path(tmp)
        for model_name, (model, gray_input) in models.items():
            print(f"\n── Model: {model_name}")
            for codec_name in args.codecs:
                cfg = CRF_CONFIGS[codec_name]
                print(f"   codec={codec_name}  CRF=({cfg['crf_roi']},{cfg['crf_non']})",
                      flush=True)

                # Warm-up
                for w in range(args.warmup):
                    wdir = tmp_path / f"warm_{model_name}_{codec_name}_{w}"
                    _ = bench_one_frame(
                        frames[w], model_name, model, gray_input,
                        codec_name, cfg, wdir, device,
                        framerate=args.framerate, keyint=keyint)
                    shutil.rmtree(wdir, ignore_errors=True)

                # Measured frames
                for i in range(args.num_frames):
                    frame = frames[args.warmup + i]
                    wdir = tmp_path / f"run_{model_name}_{codec_name}_{i}"
                    rec = bench_one_frame(
                        frame, model_name, model, gray_input,
                        codec_name, cfg, wdir, device,
                        framerate=args.framerate, keyint=keyint)
                    rec.frame_idx = args.warmup + i
                    records.append(rec)
                    shutil.rmtree(wdir, ignore_errors=True)

                # In nhanh trung bình cho cặp này
                last = [r for r in records
                        if r.model == model_name and r.codec == codec_name]
                mean_ms = statistics.fmean(r.t_total for r in last) * 1000
                fps = 1000.0 / mean_ms
                print(f"     → mean total = {mean_ms:.1f} ms/frame  ({fps:.2f} FPS)")

    rows = aggregate(records)
    print_summary_table(rows)

    meta = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "num_frames": args.num_frames,
        "warmup": args.warmup,
        "split": args.split,
        "framerate": args.framerate,
        "keyint": keyint,
        "frame_hw": list(FRAME_HW),
        "seg_input_hw": list(SEG_INPUT_HW),
        "ccnet_ckpt": str(args.ccnet_ckpt),
        "ccnet_arch": args.ccnet_arch,
        "pidnet_ckpt": str(args.pidnet_ckpt),
        "codecs": list(args.codecs),
        "models": list(args.models),
        "crf_configs": CRF_CONFIGS,
    }
    save_outputs(records, rows, args.out_dir, meta)
    print(f"Saved → {args.out_dir}/{{per_frame.csv,summary.csv,summary.json}}")


if __name__ == "__main__":
    main()
