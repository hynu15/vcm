import os
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

# Đường dẫn
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.environ.get('SAC_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data'))
GT_FINE_ROOT = os.path.join(DATA_ROOT, 'gt_4class', 'gtFine_trainvaltest', 'gtFine')
GT_LABEL_ROOT = os.path.join(DATA_ROOT, 'gt_4class')

# Mapping Cityscapes labelIds → 4 class SAC
# Class 0 = ROI     : đường, xe, người — mọi pixel không thuộc 3 class dưới (default)
# Class 1 = sky     : labelId 23
# Class 2 = construction: labelId 11–16 (building, wall, fence, guard rail, bridge, tunnel)
# Class 3 = nature  : labelId 21–22 (vegetation, terrain)

SKY_IDS           = [23]
CONSTRUCTION_IDS  = [11, 12, 13, 14, 15, 16]
NATURE_IDS        = [21, 22]


def collect_label_files(split):
    split_root = os.path.join(GT_FINE_ROOT, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Không tìm thấy thư mục split: {split_root}")

    label_files = []
    for city in sorted(os.listdir(split_root)):
        city_dir = os.path.join(split_root, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in os.listdir(city_dir):
            if fname.endswith('_labelIds.png'):
                label_files.append((city, fname, os.path.join(city_dir, fname)))
    return label_files


def convert_to_4class(label_path, save_path):
    label = np.array(Image.open(label_path))
    mask = np.zeros_like(label, dtype=np.uint8)  # default = 0 (ROI)
    for sid in SKY_IDS:
        mask[label == sid] = 1
    for cid in CONSTRUCTION_IDS:
        mask[label == cid] = 2
    for nid in NATURE_IDS:
        mask[label == nid] = 3
    Image.fromarray(mask).save(save_path)


print("⚠️  Cityscapes test split không có ground truth semantic labels.")
print("    Chỉ xử lý train và val.\n")
print("4-class mapping:")
print("  0 = ROI          (road, vehicle, person, ...)")
print("  1 = sky          (labelId 23)")
print("  2 = construction (labelId 11-16: building/wall/fence/guard rail/bridge/tunnel)")
print("  3 = nature       (labelId 21-22: vegetation/terrain)\n")

for split in ("train", "val"):
    print(f"Đang chuẩn bị 4-class label cho {split} set...")
    split_list = collect_label_files(split)

    for city, fname, label_path in tqdm(split_list):
        save_path = os.path.join(
            GT_LABEL_ROOT,
            split,
            city,
            fname.replace('_gtFine_labelIds.png', '_gtFine_4class.png')
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        convert_to_4class(label_path, save_path)

    print(f"✅ Hoàn thành 4-class label cho {split} set!")
    print(f"   Số ảnh: {len(split_list)}")
    print(f"   Lưu tại: {GT_LABEL_ROOT}/{split}/")
