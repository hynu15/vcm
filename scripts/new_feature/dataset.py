"""
Cityscapes 4-class dataset for SAC paper training / evaluation.

Label file naming: <city>/<stem>_gtFine_4class.png
Image file naming: <city>/<stem>_leftImg8bit.png

Class mapping (from prepare_4class_labels.py):
  0 = ROI          (road, vehicle, pedestrian, rider, traffic-light, traffic-sign,
                    unlabelled, dynamic, ground, …  — anything not in classes 1-3)
  1 = sky
  2 = construction (building, wall, fence, guard-rail, bridge, tunnel)
  3 = nature       (vegetation, terrain)

Augmentation (training only):
  • Random scale: resize to [0.75, 1.5]× then crop to base size
  • Random horizontal flip (p=0.5)
  • Color jitter (brightness, contrast, saturation ±0.4; hue ±0.1)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Cityscapes4Class(Dataset):
    """
    Paired (image, 4-class label) dataset for Cityscapes splits.

    Parameters
    ----------
    image_dir : str | Path
    label_dir : str | Path
    image_size : (H, W)   — paper uses (512, 1024)
    augment : bool        — True for training split
    """

    DEFAULT_SIZE: Tuple[int, int] = (512, 1024)

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        image_size: Tuple[int, int] = DEFAULT_SIZE,
        augment: bool = False,
        cities: list[str] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size   # (H, W)
        self.augment    = augment

        self._color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

        city_filter = set(cities) if cities is not None else None

        self.samples: list[Tuple[Path, Path]] = []
        for city_dir in sorted(self.label_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            if city_filter is not None and city_dir.name not in city_filter:
                continue
            city_img_dir = self.image_dir / city_dir.name
            for lbl_path in sorted(city_dir.glob("*_gtFine_4class.png")):
                stem     = lbl_path.name.replace("_gtFine_4class.png", "")
                img_path = city_img_dir / f"{stem}_leftImg8bit.png"
                if img_path.is_file():
                    self.samples.append((img_path, lbl_path))

        if not self.samples:
            raise RuntimeError(
                f"No samples found. image_dir={image_dir}, label_dir={label_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        H, W = self.image_size
        image = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        label = Image.open(lbl_path).resize((W, H), Image.NEAREST)

        if self.augment:
            image, label = self._augment(image, label, H, W)

        image_t = transforms.ToTensor()(image)
        image_t = transforms.Normalize(self.MEAN, self.STD)(image_t)
        label_t = torch.from_numpy(np.array(label)).long()

        return image_t, label_t, str(img_path)

    def _augment(self, image: Image.Image, label: Image.Image,
                 H: int, W: int) -> Tuple[Image.Image, Image.Image]:
        # 1. Random scale: [0.75, 1.5] × base size, always ≥ base size so we can crop
        scale   = random.uniform(0.75, 1.5)
        new_h   = max(int(H * scale), H)
        new_w   = max(int(W * scale), W)
        image   = image.resize((new_w, new_h), Image.BILINEAR)
        label   = label.resize((new_w, new_h), Image.NEAREST)

        # 2. Random crop back to (H, W)
        top  = random.randint(0, new_h - H)
        left = random.randint(0, new_w - W)
        image = image.crop((left, top, left + W, top + H))
        label = label.crop((left, top, left + W, top + H))

        # 3. Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. Color jitter (image only, not label)
        image = self._color_jitter(image)

        return image, label

    def get_paths(self, idx: int) -> Tuple[Path, Path]:
        return self.samples[idx]
