import argparse
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_segmentation import CCNet


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class ChunkJob:
    chunk_id: int
    frames: List[np.ndarray]
    masks: List[np.ndarray]
    capture_times: List[float]


@dataclass
class ChunkResult:
    chunk_id: int
    originals: List[np.ndarray]
    sac_frames: List[np.ndarray]
    trad_frames: List[np.ndarray]
    capture_times: List[float]
    process_latency_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-latency SAC-X265 webcam demo using background chunk processing."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=960, help="Capture width.")
    parser.add_argument("--height", type=int, default=540, help="Capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS for chunk encoding.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Frames per encoding chunk. Use 1-2 for lower latency.",
    )
    parser.add_argument("--crf-roi", type=int, default=25, help="CRF for ROI stream.")
    parser.add_argument("--crf-non", type=int, default=32, help="CRF for non-ROI stream.")
    parser.add_argument("--crf-trad", type=int, default=28, help="CRF for traditional X265 stream.")
    parser.add_argument(
        "--preset",
        type=str,
        default="ultrafast",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
        ],
        help="x265 preset, ultrafast is recommended for low-latency preview.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "best_ccnet.pth"),
        help="Path to trained segmentation model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[384, 768],
        metavar=("H", "W"),
        help="Model inference size (H W). Smaller is faster.",
    )
    parser.add_argument(
        "--job-queue-size",
        type=int,
        default=2,
        help="Max queued chunks for background processing. Smaller queue keeps latency bounded.",
    )
    parser.add_argument(
        "--display-queue-size",
        type=int,
        default=8,
        help="Max processed frames waiting for display.",
    )
    parser.add_argument(
        "--disable-traditional",
        action="store_true",
        help="Disable traditional x265 branch in hot path to reduce latency.",
    )
    parser.add_argument(
        "--mask-refresh-interval",
        type=int,
        default=2,
        help="Run segmentation every N frames and reuse the latest mask in between.",
    )
    parser.add_argument(
        "--max-e2e-latency-ms",
        type=int,
        default=0,
        help="Drop queued frames older than this end-to-end latency threshold (milliseconds).",
    )
    return parser.parse_args()


def run_ffmpeg(cmd: List[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def encode_x265_from_pattern(
    input_pattern: str,
    output_video: str,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-x265-params",
        "log-level=error:rc-lookahead=0:bframes=0:keyint=15:min-keyint=15",
        output_video,
    ]
    run_ffmpeg(cmd)


def read_video_frames(video_path: str, expected_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while len(frames) < expected_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_model(model_path: str, device: torch.device) -> CCNet:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = CCNet(num_classes=4, backbone_weights=None)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_transform(input_h: int, input_w: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_h, input_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_roi_mask(
    model: CCNet,
    frame_rgb: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    pil = Image.fromarray(frame_rgb)
    input_tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(input_tensor)
    mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    roi = (mask == 0).astype(np.uint8)
    return roi


def save_chunk_images(
    chunk_dir: str,
    originals: List[np.ndarray],
    rois: List[np.ndarray],
    nons: List[np.ndarray],
) -> None:
    raw_dir = os.path.join(chunk_dir, "raw")
    roi_dir = os.path.join(chunk_dir, "roi")
    non_dir = os.path.join(chunk_dir, "non")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(non_dir, exist_ok=True)

    for i, (raw, roi, non) in enumerate(zip(originals, rois, nons)):
        cv2.imwrite(os.path.join(raw_dir, f"{i:04d}.png"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(roi_dir, f"{i:04d}.png"), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(non_dir, f"{i:04d}.png"), cv2.cvtColor(non, cv2.COLOR_RGB2BGR))


def process_chunk(
    chunk_dir: str,
    originals: List[np.ndarray],
    roi_masks: List[np.ndarray],
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
    disable_traditional: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    roi_frames = []
    non_frames = []
    for raw, roi_m in zip(originals, roi_masks):
        roi_3 = np.repeat(roi_m[:, :, None], 3, axis=2)
        non_3 = 1 - roi_3
        roi_frames.append((raw * roi_3).astype(np.uint8))
        non_frames.append((raw * non_3).astype(np.uint8))

    save_chunk_images(chunk_dir, originals, roi_frames, non_frames)

    roi_video = os.path.join(chunk_dir, "roi.mp4")
    non_video = os.path.join(chunk_dir, "non.mp4")
    trad_video = os.path.join(chunk_dir, "trad.mp4")

    encode_x265_from_pattern(os.path.join(chunk_dir, "roi", "%04d.png"), roi_video, fps, crf_roi, preset)
    encode_x265_from_pattern(os.path.join(chunk_dir, "non", "%04d.png"), non_video, fps, crf_non, preset)
    if not disable_traditional:
        encode_x265_from_pattern(os.path.join(chunk_dir, "raw", "%04d.png"), trad_video, fps, crf_trad, preset)

    dec_roi = read_video_frames(roi_video, len(originals))
    dec_non = read_video_frames(non_video, len(originals))
    if disable_traditional:
        dec_trad = originals
    else:
        dec_trad = read_video_frames(trad_video, len(originals))

    n = min(len(dec_roi), len(dec_non), len(dec_trad), len(originals), len(roi_masks))
    sac = []
    trad = []
    for i in range(n):
        roi_3 = np.repeat(roi_masks[i][:, :, None], 3, axis=2)
        merged = np.where(roi_3 == 1, dec_roi[i], dec_non[i]).astype(np.uint8)
        sac.append(merged)
        trad.append(dec_trad[i])

    return sac, trad


def draw_panel(
    original: np.ndarray,
    sac: np.ndarray,
    trad: np.ndarray,
    process_latency_s: float,
    e2e_latency_s: float,
    fps_avg: float,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    disable_traditional: bool,
    dropped_chunks: int,
    dropped_stale_frames: int,
) -> np.ndarray:
    def to_bgr(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    a = to_bgr(original)
    b = to_bgr(sac)
    c = to_bgr(trad)
    h = min(a.shape[0], b.shape[0], c.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)

    cv2.putText(a, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 255, 40), 2)
    cv2.putText(b, f"SAC-X265 ROI={crf_roi} NON={crf_non}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    trad_label = "Traditional disabled" if disable_traditional else f"Traditional X265 CRF={crf_trad}"
    cv2.putText(c, trad_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 40), 2)

    # Display order requested for easier visual comparison: Original -> Traditional -> SAC.
    panel = np.hstack([a, c, b])
    cv2.putText(
        panel,
        f"Proc latency: {process_latency_s:.3f}s | E2E latency: {e2e_latency_s:.3f}s | Display FPS: {fps_avg:.2f}",
        (20, panel.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
    )
    cv2.putText(
        panel,
        f"Dropped chunks: {dropped_chunks} | Dropped stale frames: {dropped_stale_frames}",
        (20, panel.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 220, 255),
        2,
    )

    # Keep 3-stream panel visible on common laptop screens.
    max_w, max_h = 1600, 900
    ph, pw = panel.shape[:2]
    scale = min(max_w / pw, max_h / ph, 1.0)
    if scale < 1.0:
        panel = cv2.resize(panel, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
    return panel


def chunk_worker(
    stop_event: threading.Event,
    job_queue: "queue.Queue[Optional[ChunkJob]]",
    result_queue: "queue.Queue[ChunkResult]",
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
    disable_traditional: bool,
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
            with tempfile.TemporaryDirectory(prefix="sac_rt_ll_") as chunk_dir:
                sac_frames, trad_frames = process_chunk(
                    chunk_dir=chunk_dir,
                    originals=job.frames,
                    roi_masks=job.masks,
                    fps=fps,
                    crf_roi=crf_roi,
                    crf_non=crf_non,
                    crf_trad=crf_trad,
                    preset=preset,
                    disable_traditional=disable_traditional,
                )
            latency = time.perf_counter() - t0
            result_queue.put(
                ChunkResult(
                    chunk_id=job.chunk_id,
                    originals=job.frames,
                    sac_frames=sac_frames,
                    trad_frames=trad_frames,
                    capture_times=job.capture_times,
                    process_latency_s=latency,
                )
            )
        except Exception as exc:  # keep UI alive even if one chunk fails
            print(f"[worker] failed on chunk {job.chunk_id}: {exc}")
        finally:
            job_queue.task_done()


def main() -> None:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg with libx265 support.")

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.mask_refresh_interval < 1:
        raise ValueError("--mask-refresh-interval must be >= 1")
    if args.max_e2e_latency_ms < 0:
        raise ValueError("--max-e2e-latency-ms must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, device)
    transform = build_transform(args.input_size[0], args.input_size[1])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    frame_buffer: List[np.ndarray] = []
    mask_buffer: List[np.ndarray] = []
    capture_times: List[float] = []
    show_queue: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []

    job_queue: "queue.Queue[Optional[ChunkJob]]" = queue.Queue(maxsize=max(1, args.job_queue_size))
    result_queue: "queue.Queue[ChunkResult]" = queue.Queue()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=chunk_worker,
        args=(
            stop_event,
            job_queue,
            result_queue,
            args.fps,
            args.crf_roi,
            args.crf_non,
            args.crf_trad,
            args.preset,
            args.disable_traditional,
        ),
        daemon=True,
    )
    worker.start()

    last_disp = time.perf_counter()
    fps_ema = 0.0
    chunk_id = 0
    frame_idx = 0
    cached_roi_mask: Optional[np.ndarray] = None
    dropped_chunks = 0
    dropped_stale_frames = 0
    last_rendered_panel: Optional[np.ndarray] = None

    max_e2e_latency_s = args.max_e2e_latency_ms / 1000.0

    print("Starting low-latency preview. Press 'q' to quit.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera stream ended or frame read failed.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if cached_roi_mask is None or (frame_idx % args.mask_refresh_interval == 0):
                cached_roi_mask = infer_roi_mask(model, frame_rgb, transform, device)
            roi_mask = cached_roi_mask
            frame_idx += 1
            frame_buffer.append(frame_rgb)
            mask_buffer.append(roi_mask)
            capture_times.append(time.perf_counter())

            if len(frame_buffer) >= args.chunk_size:
                job = ChunkJob(
                    chunk_id=chunk_id,
                    frames=frame_buffer,
                    masks=mask_buffer,
                    capture_times=capture_times,
                )
                chunk_id += 1

                if job_queue.full():
                    # Drop this chunk to keep tail latency bounded under overload.
                    dropped_chunks += 1
                    frame_buffer = []
                    mask_buffer = []
                    capture_times = []
                else:
                    job_queue.put(job)
                    frame_buffer = []
                    mask_buffer = []
                    capture_times = []

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break

                n = min(
                    len(result.originals),
                    len(result.sac_frames),
                    len(result.trad_frames),
                    len(result.capture_times),
                )
                ready_time = time.perf_counter()
                for i in range(n):
                    e2e_latency = ready_time - result.capture_times[i]
                    show_queue.append(
                        (
                            result.originals[i],
                            result.sac_frames[i],
                            result.trad_frames[i],
                            result.process_latency_s,
                            e2e_latency,
                        )
                    )

                # Keep display queue small to avoid showing stale frames.
                if len(show_queue) > args.display_queue_size:
                    dropped_stale_frames += len(show_queue) - args.display_queue_size
                    show_queue = show_queue[-args.display_queue_size :]

            if show_queue and max_e2e_latency_s > 0:
                fresh_queue: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
                for item in show_queue:
                    if item[4] <= max_e2e_latency_s:
                        fresh_queue.append(item)
                    else:
                        dropped_stale_frames += 1
                show_queue = fresh_queue

            if show_queue:
                original, sac, trad, process_latency, e2e_latency = show_queue.pop(0)
                now = time.perf_counter()
                dt = max(now - last_disp, 1e-6)
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps
                last_disp = now

                panel = draw_panel(
                    original=original,
                    sac=sac,
                    trad=trad,
                    process_latency_s=process_latency,
                    e2e_latency_s=e2e_latency,
                    fps_avg=fps_ema,
                    crf_roi=args.crf_roi,
                    crf_non=args.crf_non,
                    crf_trad=args.crf_trad,
                    disable_traditional=args.disable_traditional,
                    dropped_chunks=dropped_chunks,
                    dropped_stale_frames=dropped_stale_frames,
                )
                last_rendered_panel = panel
                cv2.imshow("SAC X265 Low-Latency Demo", panel)
            else:
                if last_rendered_panel is not None:
                    hold = last_rendered_panel.copy()
                    cv2.putText(
                        hold,
                        f"Waiting new chunk... queue {job_queue.qsize()}/{args.job_queue_size} | buffer {len(frame_buffer)}/{args.chunk_size}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("SAC X265 Low-Latency Demo", hold)
                else:
                    preview = frame_bgr.copy()
                    cv2.putText(
                        preview,
                        f"Capturing... frame buffer {len(frame_buffer)}/{args.chunk_size}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        preview,
                        f"Chunk queue: {job_queue.qsize()}/{args.job_queue_size} | Press q to quit",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (220, 220, 220),
                        2,
                    )
                    cv2.putText(
                        preview,
                        f"Mask refresh every {args.mask_refresh_interval} frame(s) | Max E2E {args.max_e2e_latency_ms} ms",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 240, 180),
                        2,
                    )
                    cv2.putText(
                        preview,
                        f"Dropped chunks: {dropped_chunks} | Dropped stale frames: {dropped_stale_frames}",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 220, 255),
                        2,
                    )
                    cv2.imshow("SAC X265 Low-Latency Demo", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_event.set()
        try:
            job_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
