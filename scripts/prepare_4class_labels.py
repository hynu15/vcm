import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Đường dẫn
DATA_ROOT = os.path.expanduser('~/sac_project/data')
GT_FINE_ROOT = os.path.join(DATA_ROOT, 'gt_4class', 'gtFine_trainvaltest', 'gtFine')
GT_4CLASS_ROOT = os.path.join(DATA_ROOT, 'gt_4class')

# Class IDs theo Cityscapes (theo paper)
CONSTRUCTION_IDS = [11, 12, 13, 14, 15, 16]   # building, wall, fence, guard rail, bridge, tunnel
NATURE_IDS      = [21, 22]                    # vegetation, terrain
SKY_ID          = 23


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
    mask = np.zeros_like(label, dtype=np.uint8)
    
    # construction = 2
    for cid in CONSTRUCTION_IDS:
        mask[label == cid] = 2
    # nature = 3
    for nid in NATURE_IDS:
        mask[label == nid] = 3
    # sky = 1
    mask[label == SKY_ID] = 1
    # còn lại = 0 (ROI)
    
    # Lưu dưới dạng PNG 8-bit
    Image.fromarray(mask).save(save_path)


# Xử lý val trước (nhanh để test)
print("Đang chuẩn bị label cho val set...")
val_list = collect_label_files('val')

for city, fname, label_path in tqdm(val_list):
    save_path = os.path.join(
        GT_4CLASS_ROOT,
        'val',
        city,
        fname.replace('_labelIds.png', '_4class.png')
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    convert_to_4class(label_path, save_path)

print("✅ Hoàn thành label 4-class cho val set!")
print(f"   Số ảnh: {len(val_list)}")
print(f"   Lưu tại: {GT_4CLASS_ROOT}/val/")
