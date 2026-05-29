"""
prepare_bdd_labels.py — Chuyển đổi BDD100K segmentation masks → 4-class SAC labels.

Mapping BDD100K (19 class + 255=ignore) → 4 class (theo quy ước paths.py new_branch):
  0 = ROI        : road(0), sidewalk(1), traffic_light(6), traffic_sign(7),
                   person(11), rider(12), car(13), truck(14), bus(15),
                   train(16), motorcycle(17), bicycle(18)
  1 = sky        : sky(10)
  2 = construction: building(2), wall(3), fence(4), pole(5)
  3 = nature     : vegetation(8), terrain(9)
  255 = ignore

Cách dùng:
  conda run -n sac python3 scripts/prepare_bdd_labels.py --dry-run   # kiểm tra cấu trúc
  conda run -n sac python3 scripts/prepare_bdd_labels.py             # chạy thật

Trên RunPod (sau khi giải nén archive.zip vào /workspace/data/archive/):
  conda run -n sac python3 /workspace/sac_project/scripts/prepare_bdd_labels.py \
      --data-root /workspace/data/archive \
      --out-root  /workspace/data/gt_4class_bdd
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_): return it

# ── Lookup table: BDD100K id (0-255) → 4-class id ────────────────────────────
# Mặc định toàn bộ = 0 (ROI); chỉ ghi đè các class NON-ROI
_LUT = np.zeros(256, dtype=np.uint8)
_LUT[2]  = 2   # building    → construction
_LUT[3]  = 2   # wall        → construction
_LUT[4]  = 2   # fence       → construction
_LUT[5]  = 2   # pole        → construction
_LUT[8]  = 3   # vegetation  → nature
_LUT[9]  = 3   # terrain     → nature
_LUT[10] = 1   # sky         → sky
_LUT[255] = 255  # ignore


def convert_mask(src: Path, dst: Path) -> None:
    arr = np.array(Image.open(src), dtype=np.uint8)
    out = _LUT[arr]
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst)


# ── Auto-detect thư mục label (hỗ trợ format cũ và mới) ──────────────────────

def _find_label_root(archive_root: Path):
    candidates = [
        archive_root / "bdd100k_seg" / "bdd100k" / "seg" / "labels",   # format thực tế
        archive_root / "bdd100k_seg" / "seg" / "labels",
        archive_root / "bdd100k_seg" / "labels" / "sem_seg" / "masks",
        archive_root / "bdd100k_seg" / "labels",
    ]
    for c in candidates:
        train_dir = c / "train"
        if train_dir.is_dir() and any(train_dir.glob("*.png")):
            return c
    return None


def _find_image_root(archive_root: Path):
    candidates = [
        archive_root / "bdd100k_seg" / "bdd100k" / "seg" / "images",   # format thực tế
        archive_root / "bdd100k_seg" / "seg" / "images",
        archive_root / "bdd100k_seg" / "images" / "10k",
        archive_root / "bdd100k" / "images" / "10k",
    ]
    for c in candidates:
        if (c / "train").is_dir():
            return c
    return None


def print_structure(archive_root: Path, depth: int = 3) -> None:
    seg_root = archive_root / "bdd100k_seg"
    if not seg_root.exists():
        print(f"  [!] Không tìm thấy {seg_root}")
        return
    print(f"\n  Cấu trúc {seg_root}:")
    for root, dirs, files in os.walk(seg_root):
        level = root.replace(str(seg_root), "").count(os.sep)
        if level >= depth:
            dirs[:] = []
            continue
        indent = "  " * (level + 2)
        print(f"{indent}{os.path.basename(root)}/")
        if level == depth - 1:
            sub = "  " * (level + 3)
            for f in files[:5]:
                print(f"{sub}{f}")
            if len(files) > 5:
                print(f"{sub}... (+{len(files) - 5} files)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="BDD100K seg → 4-class SAC labels")
    ap.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent.parent / "data" / "archive"),
        help="Thư mục chứa bdd100k_seg/ (mặc định: <project>/data/archive)",
    )
    ap.add_argument(
        "--out-root",
        default=str(Path(__file__).resolve().parent.parent / "data" / "gt_4class_bdd"),
        help="Thư mục đầu ra nhãn 4-class (mặc định: <project>/data/gt_4class_bdd)",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in cấu trúc và kiểm tra, không ghi file",
    )
    args = ap.parse_args()

    archive_root = Path(args.data_root)
    out_root = Path(args.out_root)

    if not archive_root.exists():
        print(f"[LỖI] data-root không tồn tại: {archive_root}")
        print("  Kiểm tra lại đường dẫn hoặc giải nén archive.zip trước.")
        sys.exit(1)

    print_structure(archive_root)

    lbl_root = _find_label_root(archive_root)
    img_root = _find_image_root(archive_root)

    if lbl_root is None:
        print("\n[LỖI] Không tìm thấy thư mục label PNG trong bdd100k_seg/")
        print("  Xem cấu trúc ở trên và truyền --data-root đúng thư mục.")
        sys.exit(1)

    print(f"\n  Label root : {lbl_root}")
    print(f"  Image root : {img_root or '(không tìm thấy)'}")
    print(f"  Output root: {out_root}")
    print("\n  4-class mapping:")
    print("    0 = ROI          (road, sidewalk, person, vehicle, ...)")
    print("    1 = sky          (BDD ID 10)")
    print("    2 = construction (BDD ID 2,3,4,5: building/wall/fence/pole)")
    print("    3 = nature       (BDD ID 8,9: vegetation/terrain)")
    print("    255 = ignore")

    if args.dry_run:
        print("\n[dry-run] Kết thúc. Bỏ --dry-run để chạy thật.")
        return

    total_ok = 0
    for split in args.splits:
        src_dir = lbl_root / split
        dst_dir = out_root / split

        if not src_dir.is_dir():
            print(f"\n[WARN] Không tìm thấy split '{split}' tại {src_dir}, bỏ qua.")
            continue

        png_files = sorted(src_dir.glob("*.png"))
        if not png_files:
            print(f"\n[WARN] Không có file .png trong {src_dir}, bỏ qua.")
            continue

        print(f"\n  [{split}] {len(png_files)} file → {dst_dir}/")
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src in tqdm(png_files, desc=f"  {split}"):
            convert_mask(src, dst_dir / src.name)
            total_ok += 1

        # Xác minh 1 file
        check = np.array(Image.open(dst_dir / png_files[0].name))
        unique_vals = sorted(np.unique(check).tolist())
        bad = [v for v in unique_vals if v not in (0, 1, 2, 3, 255)]
        status = "OK" if not bad else f"[WARN] có giá trị lạ: {bad}"
        print(f"    Kiểm tra {png_files[0].name}: unique={unique_vals} → {status}")

    print(f"\n  Hoàn thành: {total_ok} file đã convert.")
    print(f"  Labels lưu tại: {out_root}")
    print("\n  Bước tiếp theo: chạy train_pidnet_l_bdd.py")


if __name__ == "__main__":
    main()
