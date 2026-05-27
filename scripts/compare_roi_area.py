"""
So sánh diện tích ROI trung bình trên 10 frame giữa PIDNet-L và CCNet (4-class).

Chạy từ thư mục scripts/:
  conda run -n sac python3 compare_roi_area.py
  conda run -n sac python3 compare_roi_area.py --num-frames 20 --split val

Định nghĩa ROI:
  PIDNet-L (4-class): class 1=road, 2=vehicle, 3=pedestrian → ROI = pixel có class != 0
  CCNet    (4-class): class 0=ROI (road/vehicle/ped/...) → ROI = pixel có class == 0
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

PROJECT_ROOT = _SCRIPTS.parent

# ── Import model definitions ──────────────────────────────────────────────────

from train_pidnet_l import build_pidnet_l
from new_feature.ccnet_4class import CCNet4Class, load_ccnet_4class

# ── Paths ─────────────────────────────────────────────────────────────────────

PIDNET_CKPT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
CCNET_CKPT  = PROJECT_ROOT / "models" / "best_ccnet_4class.pth"

IMG_ROOTS = {
    "val":  PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val",
    "train": PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train",
}

# ── Preprocessing ─────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Load PIDNet-L 4-class ─────────────────────────────────────────────────────

def load_pidnet_l(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    num_classes = ckpt.get("meta", {}).get("num_classes", 4)
    model = build_pidnet_l(num_classes=num_classes).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, img_tensor: torch.Tensor, model_name: str) -> torch.Tensor:
    """Return [H, W] prediction map (class indices)."""
    x = img_tensor.unsqueeze(0)           # [1, 3, H, W]
    out = model(x)

    # PIDNet returns list when augment=True; take main logits (index 1)
    if isinstance(out, (list, tuple)):
        logits = out[1]
    else:
        logits = out

    # Upsample if output is smaller than input
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

    pred = logits.argmax(dim=1).squeeze(0).cpu()   # [H, W]
    return pred


def roi_stats(pred: torch.Tensor, roi_class_fn) -> dict:
    """Compute ROI pixel count and percentage from prediction map."""
    total = pred.numel()
    roi_mask = roi_class_fn(pred)
    roi_pixels = int(roi_mask.sum().item())
    return {
        "roi_pixels": roi_pixels,
        "total_pixels": total,
        "roi_pct": roi_pixels / total * 100.0,
    }


# ── ROI class definitions ─────────────────────────────────────────────────────

def pidnet_roi(pred: torch.Tensor) -> torch.Tensor:
    """ROI = class 0 (road/vehicle/ped — same convention as CCNet)."""
    return pred == 0


def ccnet_roi(pred: torch.Tensor) -> torch.Tensor:
    """ROI = class 0 (road/vehicle/ped/cyclist/dynamic)."""
    return pred == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_images(img_root: Path, num_frames: int) -> list[Path]:
    imgs = sorted(img_root.rglob("*_leftImg8bit.png"))
    if not imgs:
        imgs = sorted(img_root.rglob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong {img_root}")
    # Lấy đều từ toàn bộ dataset
    step = max(1, len(imgs) // num_frames)
    selected = imgs[::step][:num_frames]
    return selected


def run(num_frames: int = 10, split: str = "val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Load models
    print(f"\nLoading PIDNet-L 4-class từ {PIDNET_CKPT.name} ...")
    pidnet = load_pidnet_l(PIDNET_CKPT, device)
    print(f"Loading CCNet 4-class    từ {CCNET_CKPT.name} ...")
    ccnet  = load_ccnet_4class(str(CCNET_CKPT), device)

    # Collect images
    img_root = IMG_ROOTS[split]
    imgs = collect_images(img_root, num_frames)
    print(f"\nSplit    : {split}  ({len(imgs)} frame được chọn / {num_frames} yêu cầu)")

    # Per-frame results
    rows = []
    print(f"\n{'Frame':<5} {'Tên file':<45} {'PIDNet ROI%':>12} {'CCNet ROI%':>12}")
    print("-" * 78)

    pidnet_pcts, ccnet_pcts = [], []
    pidnet_px,   ccnet_px   = [], []

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).to(device)

        pred_pid = predict(pidnet, tensor, "pidnet")
        pred_cc  = predict(ccnet,  tensor, "ccnet")

        s_pid = roi_stats(pred_pid, pidnet_roi)
        s_cc  = roi_stats(pred_cc,  ccnet_roi)

        pidnet_pcts.append(s_pid["roi_pct"])
        ccnet_pcts.append(s_cc["roi_pct"])
        pidnet_px.append(s_pid["roi_pixels"])
        ccnet_px.append(s_cc["roi_pixels"])

        rows.append({
            "frame": i + 1,
            "file": img_path.name,
            "pidnet_roi_px": s_pid["roi_pixels"],
            "ccnet_roi_px":  s_cc["roi_pixels"],
            "total_px": s_pid["total_pixels"],
            "pidnet_roi_pct": s_pid["roi_pct"],
            "ccnet_roi_pct":  s_cc["roi_pct"],
        })

        print(f"{i+1:<5} {img_path.name:<45} {s_pid['roi_pct']:>11.2f}% {s_cc['roi_pct']:>11.2f}%")

    # Summary
    total_px = rows[0]["total_px"]
    avg_pid_pct = float(np.mean(pidnet_pcts))
    avg_cc_pct  = float(np.mean(ccnet_pcts))
    avg_pid_px  = float(np.mean(pidnet_px))
    avg_cc_px   = float(np.mean(ccnet_px))
    std_pid     = float(np.std(pidnet_pcts))
    std_cc      = float(np.std(ccnet_pcts))

    print("\n" + "=" * 78)
    print(f"{'THỐNG KÊ':^78}")
    print("=" * 78)
    print(f"{'Metric':<35} {'PIDNet-L':>20} {'CCNet':>20}")
    print("-" * 78)
    print(f"{'Kích thước frame (px)':<35} {total_px:>20,}")
    print(f"{'TB diện tích ROI (pixel)':<35} {avg_pid_px:>19,.0f}  {avg_cc_px:>19,.0f}")
    print(f"{'TB diện tích ROI (%)':<35} {avg_pid_pct:>19.2f}%  {avg_cc_pct:>19.2f}%")
    print(f"{'Độ lệch chuẩn (%)':<35} {std_pid:>19.2f}%  {std_cc:>19.2f}%")
    print(f"{'Min ROI (%)':<35} {min(pidnet_pcts):>19.2f}%  {min(ccnet_pcts):>19.2f}%")
    print(f"{'Max ROI (%)':<35} {max(pidnet_pcts):>19.2f}%  {max(ccnet_pcts):>19.2f}%")
    print("=" * 78)
    print(f"\nLưu ý định nghĩa ROI (cả hai model):")
    print(f"  PIDNet-L : class == 0  (ROI = road/vehicle/pedestrian)")
    print(f"  CCNet    : class == 0  (ROI = road/vehicle/pedestrian/cyclist/dynamic)")

    # Save CSV
    out_dir = PROJECT_ROOT / "scripts" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"roi_area_comparison_{split}_{num_frames}frames.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
        # Summary rows
        w.writerow({
            "frame": "AVG", "file": "",
            "pidnet_roi_px": f"{avg_pid_px:.0f}", "ccnet_roi_px": f"{avg_cc_px:.0f}",
            "total_px": total_px,
            "pidnet_roi_pct": f"{avg_pid_pct:.4f}", "ccnet_roi_pct": f"{avg_cc_pct:.4f}",
        })
        w.writerow({
            "frame": "STD", "file": "",
            "pidnet_roi_px": "", "ccnet_roi_px": "",
            "total_px": "",
            "pidnet_roi_pct": f"{std_pid:.4f}", "ccnet_roi_pct": f"{std_cc:.4f}",
        })
    print(f"\nKết quả đã lưu: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh diện tích ROI giữa PIDNet-L và CCNet")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Số frame để đánh giá (default: 10)")
    parser.add_argument("--split", choices=["val", "train"], default="val",
                        help="Dataset split (default: val)")
    args = parser.parse_args()
    run(num_frames=args.num_frames, split=args.split)
