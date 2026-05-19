"""Real-time SAC-X265 webcam demo — low latency, single window.

Encode/decode ROI và non-ROI qua ffmpeg pipe (không ghi file ra disk),
hai stream chạy song song trên ThreadPoolExecutor.

Phím tắt:
  q  — thoát
  s  — toggle split view  (SAC only  ↔  Original | SAC side-by-side)
  m  — toggle ROI mask overlay (hiện đường viền ROI lên ảnh SAC)

Chạy:
    conda activate sac
    cd /home/huy/sac_project/scripts
    python realtime_sac_x265_lowlatency.py --camera 0

Tối ưu tốc độ:
    python realtime_sac_x265_lowlatency.py --camera 0 \\
        --width 640 --height 360 \\
        --chunk-size 1 --mask-refresh-interval 3 \\
        --input-size 256 512
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_segmentation import load_segmentation_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-latency SAC-X265 demo — pipe-based, single window."
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width",  type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--chunk-size", type=int, default=10,
                   help="Frames per encoding chunk (larger = higher throughput).")
    p.add_argument("--crf-roi",  type=int, default=25)
    p.add_argument("--crf-non",  type=int, default=32)
    p.add_argument("--preset", default="ultrafast",
                   choices=["ultrafast","superfast","veryfast","faster","fast","medium"])
    p.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models", "best_pidnet.pth"))
    p.add_argument("--input-size", type=int, nargs=2, default=[320, 640], metavar=("H","W"),
                   help="Segmentation inference size. Nhỏ hơn = nhanh hơn.")
    p.add_argument("--mask-refresh-interval", type=int, default=3,
                   help="Chạy segmentation mỗi N frame, tái dùng mask ở các frame xen giữa.")
    p.add_argument("--job-queue-size", type=int, default=2,
                   help="Số chunk tối đa đang chờ xử lý. Nhỏ giữ latency bounded.")
    p.add_argument("--display-queue-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=1,
                   help="Số background encode thread.")
    return p.parse_args()


# ─────────────────────────────── Data classes ────────────────────────────────

@dataclass
class ChunkJob:
    chunk_id: int
    frames:   List[np.ndarray]
    masks:    List[np.ndarray]
    capture_times: List[float]

@dataclass
class ChunkResult:
    chunk_id: int
    originals: List[np.ndarray]
    sac_frames: List[np.ndarray]
    roi_masks:  List[np.ndarray]
    capture_times: List[float]
    process_latency_s: float


# ─────────────────────────────── Model ──────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    if not os.path.isfile(model_path):
        fallback = os.path.join(PROJECT_ROOT, "models", "best_pidnet.pth")
        if not os.path.isfile(fallback):
            raise FileNotFoundError(f"Model không tìm thấy: {model_path}")
        model_path = fallback
    model, name = load_segmentation_model(model_path, device=device, num_classes=2)
    print(f"[seg] {name} | {model_path}")
    return model


def build_transform(h: int, w: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def infer_roi_mask(model, frame_rgb: np.ndarray, transform, device) -> np.ndarray:
    fh, fw = frame_rgb.shape[:2]
    inp = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(inp)
    mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return (mask == 0).astype(np.uint8)


# ─────────────────────────────── Pipe encode/decode ─────────────────────────

_FFMPEG_LOG = "log-level=error"

def _encode_decode_pipe(
    frames: List[np.ndarray],
    crf: int,
    preset: str,
    fps: int,
    x265_params: str,
) -> List[np.ndarray]:
    """Encode chuỗi frame RGB → x265 → decode về RGB, hoàn toàn qua stdin/stdout pipe.
    Không ghi file nào ra disk.
    """
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    n = len(frames)

    raw_in = b"".join(f.astype(np.uint8).tobytes() for f in frames)

    enc_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx265",
        "-preset", preset,
        "-crf", str(crf),
        "-x265-params", x265_params,
        "-f", "matroska", "pipe:1",
    ]
    proc_enc = subprocess.run(enc_cmd, input=raw_in, capture_output=True)
    if proc_enc.returncode != 0:
        raise RuntimeError(f"Encode lỗi (crf={crf}): {proc_enc.stderr.decode()[:200]}")

    dec_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "matroska", "-i", "pipe:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc_dec = subprocess.run(dec_cmd, input=proc_enc.stdout, capture_output=True)
    if proc_dec.returncode != 0:
        raise RuntimeError(f"Decode lỗi: {proc_dec.stderr.decode()[:200]}")

    frame_bytes = h * w * 3
    raw_out = proc_dec.stdout
    result = []
    for i in range(n):
        chunk = raw_out[i * frame_bytes : (i + 1) * frame_bytes]
        if len(chunk) < frame_bytes:
            result.append(frames[i])  # fallback: trả ảnh gốc nếu decode thiếu
        else:
            result.append(np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3).copy())
    return result


def process_chunk(
    frames: List[np.ndarray],
    roi_masks: List[np.ndarray],
    fps: int,
    crf_roi: int,
    crf_non: int,
    preset: str,
) -> List[np.ndarray]:
    """Tách ROI/non-ROI, encode song song 2 stream qua pipe, merge về."""
    n = len(frames)
    roi_streams, non_streams = [], []
    for raw, m in zip(frames, roi_masks):
        m3 = np.repeat(m[:, :, None], 3, axis=2)
        roi_streams.append((raw * m3).astype(np.uint8))
        non_streams.append((raw * (1 - m3)).astype(np.uint8))

    kf = max(1, n)
    x265_params = (
        f"aq-mode=0:rc-lookahead=0:bframes=0"
        f":keyint={kf}:min-keyint={kf}:{_FFMPEG_LOG}"
    )

    # Hai stream encode song song qua ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_roi = ex.submit(_encode_decode_pipe, roi_streams, crf_roi, preset, fps, x265_params)
        fut_non = ex.submit(_encode_decode_pipe, non_streams, crf_non, preset, fps, x265_params)
        dec_roi = fut_roi.result()
        dec_non = fut_non.result()

    sac = []
    for i in range(n):
        m3 = np.repeat(roi_masks[i][:, :, None], 3, axis=2).astype(bool)
        merged = np.where(m3, dec_roi[i], dec_non[i]).astype(np.uint8)
        sac.append(merged)
    return sac


# ─────────────────────────────── Worker thread ───────────────────────────────

def chunk_worker(
    stop_event: threading.Event,
    job_queue: "queue.Queue[Optional[ChunkJob]]",
    result_queue: "queue.Queue[ChunkResult]",
    fps: int,
    crf_roi: int,
    crf_non: int,
    preset: str,
) -> None:
    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if job is None:
            job_queue.task_done()
            break
        t0 = time.perf_counter()
        try:
            sac_frames = process_chunk(
                frames=job.frames,
                roi_masks=job.masks,
                fps=fps,
                crf_roi=crf_roi,
                crf_non=crf_non,
                preset=preset,
            )
            latency = time.perf_counter() - t0
            result_queue.put(ChunkResult(
                chunk_id=job.chunk_id,
                originals=job.frames,
                sac_frames=sac_frames,
                roi_masks=job.masks,
                capture_times=job.capture_times,
                process_latency_s=latency,
            ))
        except Exception as exc:
            print(f"[worker] chunk {job.chunk_id} failed: {exc}")
        finally:
            job_queue.task_done()


# ─────────────────────────────── Display ─────────────────────────────────────

def _draw_hud(
    img: np.ndarray,
    proc_ms: float,
    e2e_ms: float,
    fps: float,
    crf_roi: int,
    crf_non: int,
    split: bool,
    mask_on: bool,
    dropped: int,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    def txt(text, pos, scale=0.65, color=(240, 240, 240), thick=2):
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick + 2)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    # Nhãn góc trên trái
    txt(f"SAC-X265  ROI={crf_roi}  NON={crf_non}", (12, 32), 0.70, (0, 220, 255))

    # HUD dưới
    bar_h = 52
    cv2.rectangle(out, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    txt(f"Proc: {proc_ms:.0f}ms | E2E: {e2e_ms:.0f}ms | FPS: {fps:.1f}", (12, h - bar_h + 20), 0.60)
    hints = "[s] split" + (" ON" if split else "") + "  [m] mask" + (" ON" if mask_on else "") + f"  [q] quit  drop={dropped}"
    txt(hints, (12, h - bar_h + 42), 0.52, (180, 220, 180))
    return out


def _overlay_roi_contour(img: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Vẽ đường viền ROI lên ảnh (không fill mask để không che nội dung)."""
    out = img.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 128), 2)
    return out


def build_display_frame(
    original: np.ndarray,
    sac: np.ndarray,
    roi_mask: np.ndarray,
    proc_ms: float,
    e2e_ms: float,
    fps: float,
    crf_roi: int,
    crf_non: int,
    split_view: bool,
    mask_overlay: bool,
    dropped: int,
    max_w: int = 1280,
    max_h: int = 720,
) -> np.ndarray:
    def to_bgr(x): return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    sac_bgr = to_bgr(sac)
    if mask_overlay:
        sac_bgr = _overlay_roi_contour(sac_bgr, roi_mask)

    if split_view:
        orig_bgr = to_bgr(original)
        h = min(sac_bgr.shape[0], orig_bgr.shape[0])
        w = min(sac_bgr.shape[1], orig_bgr.shape[1])
        orig_bgr = cv2.resize(orig_bgr, (w, h))
        sac_bgr  = cv2.resize(sac_bgr,  (w, h))
        cv2.putText(orig_bgr, "Original", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (40, 255, 40), 2)
        frame = np.hstack([orig_bgr, sac_bgr])
    else:
        frame = sac_bgr

    frame = _draw_hud(frame, proc_ms, e2e_ms, fps,
                      crf_roi, crf_non, split_view, mask_overlay, dropped)

    # Giới hạn kích thước hiển thị
    fh, fw = frame.shape[:2]
    scale = min(max_w / fw, max_h / fh, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
    return frame


# ─────────────────────────────── Main loop ───────────────────────────────────

def main() -> None:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg không tìm thấy. Cài: sudo apt install ffmpeg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model     = load_model(args.model, device)
    transform = build_transform(args.input_size[0], args.input_size[1])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)  # giảm buffer camera để bắt frame mới nhất

    job_queue:    "queue.Queue[Optional[ChunkJob]]" = queue.Queue(maxsize=max(1, args.job_queue_size))
    result_queue: "queue.Queue[ChunkResult]"        = queue.Queue()
    stop_event = threading.Event()

    workers = []
    for _ in range(args.num_workers):
        t = threading.Thread(
            target=chunk_worker,
            args=(stop_event, job_queue, result_queue,
                  args.fps, args.crf_roi, args.crf_non, args.preset),
            daemon=True,
        )
        t.start()
        workers.append(t)

    frame_buffer:   List[np.ndarray] = []
    mask_buffer:    List[np.ndarray] = []
    capture_times:  List[float]      = []
    show_queue: List[Tuple] = []

    cached_mask: Optional[np.ndarray] = None
    chunk_id       = 0
    frame_idx      = 0
    fps_ema        = 0.0
    last_disp      = time.perf_counter()
    dropped_chunks = 0
    dropped_stale  = 0
    last_panel: Optional[np.ndarray] = None

    split_view   = False
    mask_overlay = False

    cv2.namedWindow("SAC X265", cv2.WINDOW_NORMAL)
    print("[demo] Bắt đầu. Nhấn q=thoát | s=split | m=mask")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[demo] Không đọc được frame từ camera.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Segmentation (có thể skip nếu không đến lượt refresh)
            if cached_mask is None or (frame_idx % args.mask_refresh_interval == 0):
                cached_mask = infer_roi_mask(model, frame_rgb, transform, device)
            frame_idx += 1

            frame_buffer.append(frame_rgb)
            mask_buffer.append(cached_mask)
            capture_times.append(time.perf_counter())

            # Gửi chunk khi đủ kích thước
            if len(frame_buffer) >= args.chunk_size:
                job = ChunkJob(chunk_id, frame_buffer, mask_buffer, capture_times)
                chunk_id += 1
                if job_queue.full():
                    dropped_chunks += 1
                else:
                    job_queue.put_nowait(job)
                frame_buffer, mask_buffer, capture_times = [], [], []

            # Thu kết quả từ worker (non-blocking)
            while True:
                try:
                    res = result_queue.get_nowait()
                except queue.Empty:
                    break
                ready_t = time.perf_counter()
                n = min(len(res.originals), len(res.sac_frames), len(res.capture_times))
                for i in range(n):
                    e2e = ready_t - res.capture_times[i]
                    show_queue.append((
                        res.originals[i], res.sac_frames[i], res.roi_masks[i],
                        res.process_latency_s, e2e,
                    ))

            # Trim show_queue
            if len(show_queue) > args.display_queue_size:
                dropped_stale += len(show_queue) - args.display_queue_size
                show_queue = show_queue[-args.display_queue_size:]

            # Hiển thị
            if show_queue:
                original, sac, roi_mask, proc_lat, e2e_lat = show_queue.pop(0)
                now = time.perf_counter()
                dt  = max(now - last_disp, 1e-6)
                fps_ema = (1.0 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
                last_disp = now

                panel = build_display_frame(
                    original=original, sac=sac, roi_mask=roi_mask,
                    proc_ms=proc_lat * 1000, e2e_ms=e2e_lat * 1000,
                    fps=fps_ema, crf_roi=args.crf_roi, crf_non=args.crf_non,
                    split_view=split_view, mask_overlay=mask_overlay,
                    dropped=dropped_chunks,
                )
                last_panel = panel
                cv2.imshow("SAC X265", panel)

            elif last_panel is not None:
                # Giữ nguyên frame cuối, ghi chú đang chờ
                hold = last_panel.copy()
                cv2.putText(hold, f"Waiting... buf={len(frame_buffer)}/{args.chunk_size}",
                            (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
                cv2.imshow("SAC X265", hold)

            else:
                # Chưa có frame nào từ worker — hiện preview camera thô
                preview = frame_bgr.copy()
                cv2.putText(preview, "Initializing SAC pipeline...", (12, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2)
                cv2.putText(preview,
                            f"buf={len(frame_buffer)}/{args.chunk_size}  queue={job_queue.qsize()}",
                            (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                cv2.imshow("SAC X265", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                split_view = not split_view
                print(f"[demo] split_view = {split_view}")
            elif key == ord("m"):
                mask_overlay = not mask_overlay
                print(f"[demo] mask_overlay = {mask_overlay}")

    finally:
        stop_event.set()
        for _ in workers:
            try:
                job_queue.put_nowait(None)
            except queue.Full:
                pass
        for t in workers:
            t.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()
        print(f"[demo] Kết thúc. Dropped chunks={dropped_chunks} stale={dropped_stale}")


if __name__ == "__main__":
    main()
