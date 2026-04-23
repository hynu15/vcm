"""
Test SAC-X265 pipeline with custom video file (not Cityscapes).

Usage:
    ./.venv/bin/python scripts/test_video_sac.py --video /path/to/video.mp4 --num-frames 20
or
    ./.venv/bin/python scripts/test_video_sac.py --video /path/to/video.mp4 --num-frames 50 --crf-roi 23 --crf-non 32
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from train_segmentation import load_segmentation_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# thêm chọn classs roi linh hoạt, ép sky ra non-roi bằng hearistic cho video thực tế (đôi khi model nhầm sky thành ROI)

def parse_int_list(text):
    return [int(v.strip()) for v in text.split(',') if v.strip()]


def build_roi_mask(mask, orig_rgb, roi_classes, force_sky_nonroi=False):
    """Build ROI mask from predicted labels with optional sky suppression heuristic."""
    roi_mask = np.isin(mask, roi_classes).astype(np.uint8)

    if force_sky_nonroi:
        h, w = orig_rgb.shape[:2]
        hsv = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2HSV)

        # Heuristic for daylight sky in top region.
        sky_color = (
            (hsv[..., 0] >= 85) & (hsv[..., 0] <= 130) &
            (hsv[..., 1] >= 20) &
            (hsv[..., 2] >= 80)
        )
        top_region = np.zeros((h, w), dtype=bool)
        top_region[: int(0.5 * h), :] = True
        sky_like = sky_color & top_region

        # If model predicts class 1 (sky), prioritize removing it from ROI.
        sky_like = sky_like | (mask == 1)
        roi_mask[sky_like] = 0

    return (roi_mask * 255).astype(np.uint8)


def extract_frames_from_video(video_path, num_frames=20, output_dir=None):
    """Extract frames from video file."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video: {total_frames} total frames")
    
    # Extract every N frames to get `num_frames` samples
    step = max(1, total_frames // num_frames)
    frames = []
    frame_idx = 0
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every `step` frames
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            path = os.path.join(output_dir, f"frame_{len(frames):04d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    print(f"✅ Extracted {len(frames)} frames to {output_dir}")
    return frames, output_dir


def run_ffmpeg(cmd, step_name):
    """Run FFmpeg command."""
    print(f"[FFmpeg] {step_name}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {step_name} failed")
        print(result.stderr)
        raise RuntimeError(f"FFmpeg failed: {step_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Test SAC-X265 compression with custom video file."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (mp4, avi, mov, etc.)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of frames to extract and process.",
    )
    parser.add_argument(
        "--crf-roi",
        type=int,
        default=23,
        help="CRF for ROI stream (lower = better quality).",
    )
    parser.add_argument(
        "--crf-non",
        type=int,
        default=32,
        help="CRF for non-ROI stream (higher = more aggressive compression).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: outputs/test_video/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to segmentation model (default: auto-detect best_pidnet or best_ccnet).",
    )
    parser.add_argument(
        "--roi-classes",
        type=str,
        default="0",
        help="Comma-separated class IDs kept as ROI (default: 0). Example: 0,2",
    )
    parser.add_argument(
        "--force-sky-nonroi",
        action="store_true",
        help="Apply heuristic to force sky-like pixels to non-ROI on real-world videos.",
    )
    
    args = parser.parse_args()
    roi_classes = parse_int_list(args.roi_classes)
    if not roi_classes:
        raise ValueError("--roi-classes is empty. Example: --roi-classes 0")

    # Normalize user-provided video path (supports ~ and relative paths).
    video_arg = args.video.strip()
    video_path = os.path.abspath(os.path.expanduser(video_arg))
    
    # Validate video file
    if not os.path.isfile(video_path):
        hint = ""
        if video_arg.startswith("~") and not video_arg.startswith("~/"):
            hint = " Hint: use ~/my_video.mp4 or absolute path /home/huy/my_video.mp4"
        raise FileNotFoundError(f"Video not found: {video_arg}.{hint}")
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "outputs", "test_video")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    tmp_frame_dir = os.path.join(output_dir, "tmp_frames")
    os.makedirs(tmp_frame_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if args.model is None:
        # Auto-detect best model
        candidates = [
            os.path.join(PROJECT_ROOT, 'models', 'best_pidnet.pth'),
            os.path.join(PROJECT_ROOT, 'models', 'best_ccnet.pth'),
        ]
        model_path = next((p for p in candidates if os.path.isfile(p)), None)
        if model_path is None:
            raise FileNotFoundError("No model found in models/best_pidnet.pth or models/best_ccnet.pth")
    else:
        model_path = args.model
    
    model, model_name = load_segmentation_model(model_path, device=device, num_classes=4)
    print(f"📊 Model: {model_name} | {os.path.basename(model_path)}")
    print(f"🎯 ROI classes: {roi_classes} | force_sky_nonroi={args.force_sky_nonroi}")
    
    # Extract frames
    frames, frame_dir = extract_frames_from_video(video_path, num_frames=args.num_frames)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process frames: segment and split into ROI/non-ROI
    print("🔄 Processing frames...")
    roi_frames = []
    non_roi_frames = []
    
    for idx, frame_path in enumerate(sorted(os.listdir(frame_dir))):
        if not frame_path.endswith('.png'):
            continue
        
        full_path = os.path.join(frame_dir, frame_path)
        orig = Image.open(full_path).convert('RGB')
        orig_np = np.array(orig)
        h, w = orig_np.shape[:2]
        
        # Segment
        input_tensor = transform(orig).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_mask = build_roi_mask(
            mask,
            orig_np,
            roi_classes=roi_classes,
            force_sky_nonroi=args.force_sky_nonroi,
        )
        
        # Create ROI and non-ROI frames
        roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi_mask)
        non_img = cv2.bitwise_and(orig_np, orig_np, mask=255 - roi_mask)
        
        # Save temporary frames for encoding
        cv2.imwrite(
            os.path.join(tmp_frame_dir, f"frame_{idx:04d}_roi.png"),
            cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(tmp_frame_dir, f"frame_{idx:04d}_non.png"),
            cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(tmp_frame_dir, f"frame_{idx:04d}_orig.png"),
            cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        )
        
        roi_frames.append(roi_img)
        non_roi_frames.append(non_img)
        print(f"  Frame {idx+1}/{len(frames)} ✓")
    
    # Encode videos
    print(f"\n🎬 Encoding videos (ROI CRF={args.crf_roi}, Non-ROI CRF={args.crf_non})...")
    
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", "30",
            "-i", os.path.join(tmp_frame_dir, "frame_%04d_roi.png"),
            "-c:v", "libx265", "-crf", str(args.crf_roi), "-preset", "medium",
            os.path.join(output_dir, "roi.mp4"),
        ],
        "Encode ROI stream"
    )
    
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", "30",
            "-i", os.path.join(tmp_frame_dir, "frame_%04d_non.png"),
            "-c:v", "libx265", "-crf", str(args.crf_non), "-preset", "medium",
            os.path.join(output_dir, "nonroi.mp4"),
        ],
        "Encode non-ROI stream"
    )
    
    # Merge ROI and non-ROI
    run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", os.path.join(output_dir, "roi.mp4"),
            "-i", os.path.join(output_dir, "nonroi.mp4"),
            "-filter_complex", "[0:v][1:v]blend=all_mode=addition",
            os.path.join(output_dir, "sac_x265.mp4"),
        ],
        "Merge ROI and non-ROI"
    )
    
    # Traditional X265
    total_crf = int((args.crf_roi + args.crf_non) / 2)
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", "30",
            "-i", os.path.join(tmp_frame_dir, "frame_%04d_orig.png"),
            "-c:v", "libx265", "-crf", str(total_crf), "-preset", "medium",
            os.path.join(output_dir, "traditional_x265.mp4"),
        ],
        "Encode traditional X265"
    )
    
    print(f"\n✅ Compression complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"   - roi.mp4: ROI stream only")
    print(f"   - nonroi.mp4: Non-ROI stream only")
    print(f"   - sac_x265.mp4: SAC merged (ROI + Non-ROI)")
    print(f"   - traditional_x265.mp4: Traditional H.265")
    print(f"\n💡 Next: Run metrics comparison with:")
    print(f"   ./.venv/bin/python scripts/calculate_metrics.py --video-dir {output_dir}")


if __name__ == "__main__":
    main()
