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

# non_ROI: sky, construction, nature (mọi thứ không liên quan đến xe tự hành)
NON_ROI_IDS = [
    11, 12, 13, 14, 15, 16,  # construction: building, wall, fence, guard rail, bridge, tunnel
    21, 22,                   # nature: vegetation, terrain
    23,                       # sky
]


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


def convert_to_2class(label_path, save_path):
    label = np.array(Image.open(label_path))
    mask = np.zeros_like(label, dtype=np.uint8)  # default = 0 (ROI)
    for nid in NON_ROI_IDS:
        mask[label == nid] = 1  # non_ROI
    Image.fromarray(mask).save(save_path)


print("⚠️  Cityscapes test split không có ground truth semantic labels.")
print("    Chỉ xử lý train và val.\n")

for split in ("train", "val"):
    print(f"Đang chuẩn bị 2-class label cho {split} set...")
    split_list = collect_label_files(split)

    for city, fname, label_path in tqdm(split_list):
        save_path = os.path.join(
            GT_LABEL_ROOT,
            split,
            city,
            fname.replace('_gtFine_labelIds.png', '_gtFine_2class.png')
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        convert_to_2class(label_path, save_path)

    print(f"✅ Hoàn thành 2-class label cho {split} set!")
    print(f"   Số ảnh: {len(split_list)}")
    print(f"   Lưu tại: {GT_LABEL_ROOT}/{split}/")
