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

# Mapping Cityscapes labelIds → 4 class SAC (theo reference paper CCNet)
# Class 0 = background : mọi pixel không thuộc 3 class dưới (default) → BA stream
# Class 1 = road       : labelId 7–10 (road, sidewalk, parking, rail track)  → IA stream
# Class 2 = vehicle    : labelId 26–33 (car, truck, bus, train, motorcycle, bicycle, ...) → IA stream
# Class 3 = pedestrian : labelId 24–25 (person, rider) → IA stream

ROAD_IDS        = [7, 8, 9, 10]
VEHICLE_IDS     = [26, 27, 28, 29, 30, 31, 32, 33]
PEDESTRIAN_IDS  = [24, 25]


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
    mask = np.zeros_like(label, dtype=np.uint8)  # default = 0 (background)
    for rid in ROAD_IDS:
        mask[label == rid] = 1
    for vid in VEHICLE_IDS:
        mask[label == vid] = 2
    for pid in PEDESTRIAN_IDS:
        mask[label == pid] = 3
    Image.fromarray(mask).save(save_path)


print("⚠️  Cityscapes test split không có ground truth semantic labels.")
print("    Chỉ xử lý train và val.\n")
print("4-class mapping (CCNet reference paper):")
print("  0 = background   (catch-all: sky, construction, nature, ...)")
print("  1 = road         (labelId 7-10: road, sidewalk, parking, rail track)")
print("  2 = vehicle      (labelId 26-33: car, truck, bus, motorcycle, bicycle, ...)")
print("  3 = pedestrian   (labelId 24-25: person, rider)\n")

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
