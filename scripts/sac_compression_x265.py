"""
SAC-X265 compression pipeline.

Usage:
  cd /home/huy/sac_project/scripts
  conda run -n sac python3 sac_compression_x265.py --model ccnet
  conda run -n sac python3 sac_compression_x265.py --model pidnet_l --crf-roi 20 --crf-non 30
  conda run -n sac python3 sac_compression_x265.py --model pidnet_l --max-frames 5
"""

import argparse
import os
import sys

import cv2
import numpy as np
import subprocess
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


# ── argparse ──────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC-X265 compression pipeline")
    p.add_argument("--model", choices=["ccnet", "pidnet_l"], default="ccnet",
                   help="Segmentation model (default: ccnet)")
    p.add_argument("--checkpoint", default=None,
                   help="Custom checkpoint path. Bỏ trống để dùng mặc định.")
    p.add_argument("--crf-roi",    type=int, default=23,
                   help="CRF cho ROI stream (default: 23)")
    p.add_argument("--crf-non",    type=int, default=32,
                   help="CRF cho non-ROI stream (default: 32)")
    p.add_argument("--max-frames", type=int, default=20,
                   help="Số frame tối đa (default: 20, dùng 0 để lấy hết)")
    p.add_argument("--image-dir",  default=None,
                   help="Thư mục ảnh đầu vào (default: val split Cityscapes)")
    p.add_argument("--output-dir", default=None,
                   help="Thư mục lưu kết quả (default: outputs/compressed)")
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, checkpoint: str, device: torch.device):
    if model_name == "ccnet":
        from new_feature.ccnet_4class import load_ccnet_4class
        ckpt = checkpoint or os.path.join(PROJECT_ROOT, "models", "best_ccnet_4class.pth")
        model = load_ccnet_4class(ckpt, device=device)
    elif model_name == "pidnet_l":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.new_branch.model import build_pidnet_l
        ckpt = checkpoint or os.path.join(PROJECT_ROOT, "models", "best_pidnet_l_4class.pth")
        model = build_pidnet_l(num_classes=4)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Model: {model_name} | checkpoint: {ckpt}")
    return model


# ── helpers ───────────────────────────────────────────────────────────────────

def macroblock_align_filter(mask_2d, block_size=16):
    h, w = mask_2d.shape
    pad_h = (h + block_size - 1) // block_size * block_size
    pad_w = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((pad_h, pad_w), dtype=np.uint8)
    padded[:h, :w] = mask_2d
    blocks = padded.reshape(pad_h // block_size, block_size,
                            pad_w // block_size, block_size)
    roi_max = blocks.max(axis=(1, 3))
    aligned = np.repeat(np.repeat(roi_max, block_size, axis=0), block_size, axis=1)
    return aligned[:h, :w]


def run_ffmpeg(cmd, step_name):
    print(f"[FFMPEG] {step_name}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[FFMPEG][ERROR] {step_name} failed (exit {result.returncode})")
        print(result.stderr.strip())
        raise RuntimeError(f"FFmpeg failed at step: {step_name}")


def segment(model, tensor: torch.Tensor, orig_hw: tuple, device) -> np.ndarray:
    with torch.no_grad():
        pred = model(tensor.to(device))
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_dir = args.image_dir or os.path.join(
        PROJECT_ROOT, "data", "gt_4class",
        "leftImg8bit_trainvaltest", "leftImg8bit", "val",
    )
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "compressed")
    tmp_dir = os.path.join(output_dir, "tmp_frames")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Thu thập danh sách file
    files = []
    for city in sorted(os.listdir(image_dir)):
        city_dir = os.path.join(image_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for f in sorted(os.listdir(city_dir)):
            if f.endswith("_leftImg8bit.png"):
                files.append(os.path.join(city_dir, f))

    if args.max_frames > 0:
        files = files[: args.max_frames]

    print(f"Frames : {len(files)} | CRF ROI={args.crf_roi} non={args.crf_non}")

    # Segmentation + stream split
    for idx, img_path in enumerate(files):
        orig = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig)
        tensor = transform(orig).unsqueeze(0)

        mask = segment(model, tensor, orig_np.shape[:2], device)
        roi_mask = macroblock_align_filter((mask == 0).astype(np.uint8)) * 255

        roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi_mask)
        non_img = cv2.bitwise_and(orig_np, orig_np, mask=255 - roi_mask)

        cv2.imwrite(os.path.join(tmp_dir, f"frame_{idx:04d}_orig.png"),
                    cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{idx:04d}_roi.png"),
                    cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{idx:04d}_non.png"),
                    cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))

        if (idx + 1) % 10 == 0 or idx == len(files) - 1:
            print(f"  Segmented {idx+1}/{len(files)}")

    # Nén ROI stream
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(tmp_dir, "frame_%04d_roi.png"),
        "-c:v", "libx265", "-crf", str(args.crf_roi), "-preset", "medium",
        os.path.join(output_dir, "roi.mp4"),
    ], f"Encode ROI (CRF={args.crf_roi})")

    # Nén non-ROI stream
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(tmp_dir, "frame_%04d_non.png"),
        "-c:v", "libx265", "-crf", str(args.crf_non), "-preset", "medium",
        os.path.join(output_dir, "nonroi.mp4"),
    ], f"Encode non-ROI (CRF={args.crf_non})")

    # Merge
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", os.path.join(output_dir, "roi.mp4"),
        "-i", os.path.join(output_dir, "nonroi.mp4"),
        "-filter_complex", "[0:v][1:v]blend=all_mode=addition",
        os.path.join(output_dir, "sac_x265.mp4"),
    ], "Merge SAC-X265")

    # Traditional X265 baseline
    total_crf = int((args.crf_roi + args.crf_non) / 2)
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(tmp_dir, "frame_%04d_orig.png"),
        "-c:v", "libx265", "-crf", str(total_crf), "-preset", "medium",
        os.path.join(output_dir, "traditional_x265.mp4"),
    ], f"Encode traditional X265 (CRF={total_crf})")

    print(f"\nHoan thanh!")
    print(f"  SAC-X265      : {output_dir}/sac_x265.mp4")
    print(f"  Traditional   : {output_dir}/traditional_x265.mp4")


if __name__ == "__main__":
    main()
