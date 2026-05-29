"""
Step 1: Semantic segmentation với PIDNet-L (4 class).
Chạy inference trên toàn bộ frames, lưu mask (H,W) uint8 cùng tên file.

Usage:
    python run_segmentation.py --config ../../config.yaml \
        --frames_dir /data/frames --output_dir /data/masks
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Thêm scripts/ vào path để import new_branch và train_pidnet_l
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT))

from new_branch.model import build_pidnet_l, predict_segmask  # noqa: E402
from new_branch.paths import PIDNET_L_CHECKPOINT, SEG_INPUT_HW  # noqa: E402
from new.utils.io import load_config, setup_logger, ensure_dirs, sorted_frame_paths  # noqa: E402


def load_model(checkpoint: str, num_classes: int, device: torch.device):
    model = build_pidnet_l(num_classes=num_classes)
    state = torch.load(checkpoint, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess(frame_bgr: np.ndarray, input_hw: tuple) -> torch.Tensor:
    """BGR → normalized tensor (1,3,H,W)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_hw[1], input_hw[0]))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser(description="Segmentation với PIDNet-L 4-class")
    parser.add_argument("--config",      default="../../config.yaml")
    parser.add_argument("--frames_dir",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--checkpoint",  default=str(PIDNET_L_CHECKPOINT),
                        help="Path tới .pth (mặc định: best_pidnet_l_4class.pth)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = setup_logger("segmentation")
    ensure_dirs(args.output_dir)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["segmentation"]["num_classes"]
    logger.info(f"Device: {device} | model: pidnet_l | classes: {num_classes}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    model = load_model(args.checkpoint, num_classes, device)

    frames = sorted_frame_paths(args.frames_dir)
    logger.info(f"Tìm thấy {len(frames)} frames trong {args.frames_dir}")

    for frame_path in tqdm(frames, desc="Segmenting"):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            logger.warning(f"Không đọc được {frame_path}, bỏ qua.")
            continue

        orig_h, orig_w = frame_bgr.shape[:2]
        tensor = preprocess(frame_bgr, SEG_INPUT_HW).to(device)

        # predict_segmask trả về (1,H,W) argmax ở kích thước input mạng
        # → upsample về kích thước gốc bằng nearest-neighbor
        mask_lowres = predict_segmask(model, tensor)           # (1, SEG_H, SEG_W)
        mask_lowres = mask_lowres.squeeze(0).cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask_lowres, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        out_path = Path(args.output_dir) / frame_path.name
        cv2.imwrite(str(out_path), mask)

    logger.info(f"Masks đã lưu vào {args.output_dir}")


if __name__ == "__main__":
    main()
