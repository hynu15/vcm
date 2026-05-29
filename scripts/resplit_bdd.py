"""
resplit_bdd.py — Chia lại BDD100K seg thành train/val/test có nhãn.

Hiện trạng:
  images/test/  : 2000 ảnh KHÔNG có nhãn  → xóa
  images/train/ : 7000 ảnh + labels/train/ : 7000 nhãn
  images/val/   : 1000 ảnh + labels/val/   : 1000 nhãn

Sau khi chạy:
  images/test/  : 800 ảnh  + labels/test/  : 800 nhãn
    └── 600 lấy từ train (sorted, lấy đầu danh sách)
    └── 200 lấy từ val   (sorted, lấy đầu danh sách)
  images/train/ : 6400 ảnh + labels/train/ : 6400 nhãn
  images/val/   : 800 ảnh  + labels/val/   : 800 nhãn

Cách dùng:
  conda run -n sac python3 scripts/resplit_bdd.py --dry-run   # xem trước
  conda run -n sac python3 scripts/resplit_bdd.py             # chạy thật

Trên RunPod:
  conda run -n sac python3 /workspace/sac_project/scripts/resplit_bdd.py \
      --seg-root /workspace/data/archive/bdd100k_seg/bdd100k/seg \
      --dry-run
  conda run -n sac python3 /workspace/sac_project/scripts/resplit_bdd.py \
      --seg-root /workspace/data/archive/bdd100k_seg/bdd100k/seg
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_SEG_ROOT = (
    Path(__file__).resolve().parent.parent
    / "data" / "archive" / "bdd100k_seg" / "bdd100k" / "seg"
)


def count(d: Path, ext: str = "*") -> int:
    return sum(1 for _ in d.glob(ext)) if d.is_dir() else 0


def move_pair(stem: str, src_img_dir: Path, src_lbl_dir: Path,
              dst_img_dir: Path, dst_lbl_dir: Path, dry_run: bool) -> bool:
    """Di chuyển 1 cặp (ảnh .jpg + nhãn _train_id.png). Trả về True nếu thành công."""
    img_src = src_img_dir / f"{stem}.jpg"
    lbl_src = src_lbl_dir / f"{stem}_train_id.png"

    if not img_src.exists():
        print(f"  [WARN] Không tìm thấy ảnh: {img_src}")
        return False
    if not lbl_src.exists():
        print(f"  [WARN] Không tìm thấy nhãn: {lbl_src}")
        return False

    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_src), dst_img_dir / img_src.name)
        shutil.move(str(lbl_src), dst_lbl_dir / lbl_src.name)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Chia lại BDD100K seg train/val/test")
    ap.add_argument(
        "--seg-root",
        default=str(DEFAULT_SEG_ROOT),
        help="Thư mục chứa images/ và labels/ (mặc định: data/archive/bdd100k_seg/bdd100k/seg)",
    )
    ap.add_argument("--n-train", type=int, default=600,
                    help="Số ảnh lấy từ train để tạo test (mặc định: 600)")
    ap.add_argument("--n-val", type=int, default=200,
                    help="Số ảnh lấy từ val để tạo test (mặc định: 200)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ in kế hoạch, không di chuyển file")
    args = ap.parse_args()

    seg_root = Path(args.seg_root)
    img_root = seg_root / "images"
    lbl_root = seg_root / "labels"

    if not seg_root.exists():
        print(f"[LỖI] seg-root không tồn tại: {seg_root}")
        sys.exit(1)

    # Kiểm tra thư mục cần thiết
    for d in (img_root / "train", img_root / "val", lbl_root / "train", lbl_root / "val"):
        if not d.is_dir():
            print(f"[LỖI] Không tìm thấy: {d}")
            sys.exit(1)

    # ── Thống kê trước khi chia ──────────────────────────────────────────────
    n_img_test_old = count(img_root / "test", "*.jpg")
    n_img_train    = count(img_root / "train", "*.jpg")
    n_img_val      = count(img_root / "val", "*.jpg")
    n_lbl_train    = count(lbl_root / "train", "*.png")
    n_lbl_val      = count(lbl_root / "val", "*.png")

    print("Trạng thái HIỆN TẠI:")
    print(f"  images/test/  : {n_img_test_old} ảnh (không có nhãn → sẽ xóa)")
    print(f"  images/train/ : {n_img_train} ảnh  | labels/train/ : {n_lbl_train} nhãn")
    print(f"  images/val/   : {n_img_val} ảnh  | labels/val/   : {n_lbl_val} nhãn")

    if n_lbl_train != n_img_train or n_lbl_val != n_img_val:
        print("\n[WARN] Số ảnh và nhãn không khớp — kiểm tra lại data.")

    if args.n_train > n_img_train:
        print(f"\n[LỖI] --n-train={args.n_train} > số ảnh train={n_img_train}")
        sys.exit(1)
    if args.n_val > n_img_val:
        print(f"\n[LỖI] --n-val={args.n_val} > số ảnh val={n_img_val}")
        sys.exit(1)

    print(f"\nKế hoạch:")
    print(f"  Xóa images/test/ cũ ({n_img_test_old} ảnh không nhãn)")
    print(f"  Lấy {args.n_train} ảnh đầu (sort) từ train  → test")
    print(f"  Lấy {args.n_val} ảnh đầu (sort) từ val    → test")
    print(f"\nSau khi chia:")
    print(f"  images/test/  + labels/test/  : {args.n_train + args.n_val}")
    print(f"  images/train/ + labels/train/ : {n_img_train - args.n_train}")
    print(f"  images/val/   + labels/val/   : {n_img_val - args.n_val}")

    if args.dry_run:
        print("\n[dry-run] Kết thúc. Bỏ --dry-run để thực sự di chuyển file.")
        return

    # ── Bước 1: Xóa images/test/ cũ ─────────────────────────────────────────
    old_test = img_root / "test"
    if old_test.is_dir():
        print(f"\nXóa {old_test} ({n_img_test_old} ảnh)...")
        shutil.rmtree(old_test)
        print("  Xong.")

    # ── Bước 2: Lấy stem list (sort để đảm bảo tái lập) ─────────────────────
    train_stems = sorted(p.stem for p in (img_root / "train").glob("*.jpg"))
    val_stems   = sorted(p.stem for p in (img_root / "val").glob("*.jpg"))

    stems_from_train = train_stems[:args.n_train]
    stems_from_val   = val_stems[:args.n_val]

    dst_img = img_root / "test"
    dst_lbl = lbl_root / "test"

    # ── Bước 3: Di chuyển từ train ───────────────────────────────────────────
    print(f"\nDi chuyển {args.n_train} cặp từ train → test...")
    ok = 0
    for stem in stems_from_train:
        if move_pair(stem, img_root / "train", lbl_root / "train",
                     dst_img, dst_lbl, dry_run=False):
            ok += 1
    print(f"  Hoàn thành: {ok}/{args.n_train}")

    # ── Bước 4: Di chuyển từ val ─────────────────────────────────────────────
    print(f"\nDi chuyển {args.n_val} cặp từ val → test...")
    ok = 0
    for stem in stems_from_val:
        if move_pair(stem, img_root / "val", lbl_root / "val",
                     dst_img, dst_lbl, dry_run=False):
            ok += 1
    print(f"  Hoàn thành: {ok}/{args.n_val}")

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    print("\nTrạng thái SAU KHI chia:")
    for split in ("train", "val", "test"):
        ni = count(img_root / split, "*.jpg")
        nl = count(lbl_root / split, "*.png")
        match = "✓" if ni == nl else "✗ KHÔNG KHỚP"
        print(f"  {split:6s}: {ni} ảnh | {nl} nhãn  {match}")

    print("\nXong. Chạy prepare_bdd_labels.py tiếp theo.")


if __name__ == "__main__":
    main()
