import argparse
import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_segmentation import CCNet


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Near real-time SAC-X265 webcam demo (ROI/non-ROI two-stream compression)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS for chunk encoding.")
    parser.add_argument("--chunk-size", type=int, default=12, help="Frames per encoding chunk.")
    parser.add_argument("--crf-roi", type=int, default=23, help="CRF for ROI stream.")
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
        default=[512, 1024],
        metavar=("H", "W"),
        help="Model inference size (H W). Smaller is faster.",
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
        "log-level=error",
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
    encode_x265_from_pattern(os.path.join(chunk_dir, "raw", "%04d.png"), trad_video, fps, crf_trad, preset)

    dec_roi = read_video_frames(roi_video, len(originals))
    dec_non = read_video_frames(non_video, len(originals))
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
    latency_s: float,
    fps_avg: float,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
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
    cv2.putText(c, f"Traditional X265 CRF={crf_trad}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 40), 2)

    panel = np.hstack([a, b, c])
    cv2.putText(
        panel,
        f"Chunk latency: {latency_s:.3f}s | Display FPS: {fps_avg:.2f} | Press q to quit",
        (20, panel.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
    )
    return panel


def main() -> None:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg with libx265 support.")

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
    show_queue: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    last_disp = time.perf_counter()
    fps_ema = 0.0

    print("Starting near real-time preview. Press 'q' to quit.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Camera stream ended or frame read failed.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi_mask = infer_roi_mask(model, frame_rgb, transform, device)
        frame_buffer.append(frame_rgb)
        mask_buffer.append(roi_mask)

        if len(frame_buffer) >= args.chunk_size:
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="sac_rt_") as chunk_dir:
                sac_frames, trad_frames = process_chunk(
                    chunk_dir=chunk_dir,
                    originals=frame_buffer,
                    roi_masks=mask_buffer,
                    fps=args.fps,
                    crf_roi=args.crf_roi,
                    crf_non=args.crf_non,
                    crf_trad=args.crf_trad,
                    preset=args.preset,
                )
            latency = time.perf_counter() - t0

            n = min(len(frame_buffer), len(sac_frames), len(trad_frames))
            for i in range(n):
                show_queue.append((frame_buffer[i], sac_frames[i], trad_frames[i], latency))

            frame_buffer.clear()
            mask_buffer.clear()

        if show_queue:
            original, sac, trad, latency = show_queue.pop(0)
            now = time.perf_counter()
            dt = max(now - last_disp, 1e-6)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps
            last_disp = now

            panel = draw_panel(
                original=original,
                sac=sac,
                trad=trad,
                latency_s=latency,
                fps_avg=fps_ema,
                crf_roi=args.crf_roi,
                crf_non=args.crf_non,
                crf_trad=args.crf_trad,
            )
            cv2.imshow("SAC X265 Real-time Demo", panel)
        else:
            preview = frame_bgr.copy()
            cv2.putText(
                preview,
                f"Buffering {len(frame_buffer)}/{args.chunk_size} frames...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(preview, "Press q to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
            cv2.imshow("SAC X265 Real-time Demo", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
