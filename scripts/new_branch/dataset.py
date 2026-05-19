"""CityscapesSAC – dataset loader cho 4 lớp ROI/sky/construction/nature.

Khớp với paper §III-B và CLAUDE.md §5.1:
- Đầu vào 2048×1024 RGB → downsize 1024×512 → grayscale (Y = 0.299R+0.587G+0.114B)
  rồi repeat 3 kênh để dùng pretrained ResNet-101 nguyên vẹn.
- Chuẩn hóa theo ImageNet mean/std (vì pretrained là ImageNet).
- Nhãn `_gtFine_4class.png` đã được prepare_4class_labels.py map sẵn:
  0=ROI, 1=sky, 2=construction, 3=nature.
- Random horizontal flip + random scale (0.75–1.25) cho train, no-aug cho val/test
  – paper không nêu rõ augment nhưng đây là chuẩn CCNet.
"""
from __future__ import annotations

import glob
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .paths import IMG_ROOT, LBL_ROOT, SEG_INPUT_HW

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114


def _rgb_to_gray3(arr: np.ndarray) -> np.ndarray:
    """arr: (H,W,3) uint8 RGB → (H,W,3) uint8 grayscale repeated 3 channels."""
    y = (LUMA_R * arr[..., 0] + LUMA_G * arr[..., 1] + LUMA_B * arr[..., 2]).astype(np.uint8)
    return np.stack([y, y, y], axis=-1)


def _normalize(arr_u8: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (3,H,W) float32 normalized."""
    a = arr_u8.astype(np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def _list_split(split: str) -> list[tuple[Path, Path]]:
    img_dir = IMG_ROOT / split
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
        city = img_path.parent.name
        lbl_name = img_path.name.replace("_leftImg8bit.png", "_gtFine_4class.png")
        lbl_path = LBL_ROOT / split / city / lbl_name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    if not pairs:
        raise RuntimeError(f"Không tìm thấy cặp (img, label) cho split={split!r} dưới {img_dir}")
    return pairs


class CityscapesSAC(Dataset):
    """Trả về (image: (3,H,W) float, label: (H,W) long, original_rgb: (H,W,3) uint8)."""

    def __init__(self, split: str = "train", augment: bool | None = None,
                 size_hw: tuple[int, int] = SEG_INPUT_HW, return_original: bool = False):
        assert split in ("train", "val", "test"), split
        self.split = split
        self.pairs = _list_split(split)
        self.size_hw = size_hw  # (H, W)
        self.augment = augment if augment is not None else (split == "train")
        self.return_original = return_original

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.pairs[idx]
        H, W = self.size_hw

        # Load full-resolution; downsize
        img_full = Image.open(img_path).convert("RGB")
        lbl_full = Image.open(lbl_path)

        if self.augment:
            scale = random.uniform(0.75, 1.25)
            new_w, new_h = int(W * scale), int(H * scale)
            img = img_full.resize((new_w, new_h), Image.BILINEAR)
            lbl = lbl_full.resize((new_w, new_h), Image.NEAREST)
            # Random crop / pad to (H, W)
            img_np = np.array(img)
            lbl_np = np.array(lbl)
            img_np, lbl_np = self._random_crop_or_pad(img_np, lbl_np, H, W)
            if random.random() < 0.5:
                img_np = img_np[:, ::-1, :]
                lbl_np = lbl_np[:, ::-1]
        else:
            img = img_full.resize((W, H), Image.BILINEAR)
            lbl = lbl_full.resize((W, H), Image.NEAREST)
            img_np = np.array(img)
            lbl_np = np.array(lbl)

        original_rgb = img_np.copy()
        img_gray3 = _rgb_to_gray3(img_np)
        img_tensor = _normalize(img_gray3)
        lbl_tensor = torch.from_numpy(lbl_np.astype(np.int64))

        if self.return_original:
            return img_tensor, lbl_tensor, original_rgb, str(img_path), str(lbl_path)
        return img_tensor, lbl_tensor

    @staticmethod
    def _random_crop_or_pad(img: np.ndarray, lbl: np.ndarray, H: int, W: int):
        h, w = img.shape[:2]
        # Pad if smaller
        pad_h = max(H - h, 0)
        pad_w = max(W - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
            h, w = img.shape[:2]
        # Random crop
        top = random.randint(0, h - H)
        left = random.randint(0, w - W)
        img = img[top:top + H, left:left + W]
        lbl = lbl[top:top + H, left:left + W]
        return img, lbl


def collate_default(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    lbls = torch.stack([b[1] for b in batch], 0)
    return imgs, lbls
