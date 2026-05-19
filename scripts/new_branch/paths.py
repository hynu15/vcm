"""Đường dẫn cố định cho pipeline SAC.

Mọi script trong new_branch/ import từ đây; tránh hard-code path rải rác.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # /home/huy/sac_project
SCRIPTS_ROOT = Path(__file__).resolve().parent.parent           # /home/huy/sac_project/scripts
# CCNet đặt CÙNG TRONG scripts/ để có thể sync chỉ riêng scripts/ lên RunPod.
CCNET_ROOT = SCRIPTS_ROOT / "CCNet"

DATA_ROOT = PROJECT_ROOT / "data" / "gt_4class"
IMG_ROOT = DATA_ROOT / "leftImg8bit_trainvaltest" / "leftImg8bit"
LBL_ROOT = DATA_ROOT  # train/ val/ test/ subdirs trực tiếp dưới đây

MODELS_DIR = PROJECT_ROOT / "models"
RESNET101_PRETRAINED = MODELS_DIR / "resnet101-imagenet.pth"
SAC_CHECKPOINT = MODELS_DIR / "best_ccnet_sac.pth"
PIDNET_L_CHECKPOINT = MODELS_DIR / "best_pidnet_l_4class.pth"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "new_branch"
LOG_DIR = OUTPUTS_DIR / "logs"
EVAL_DIR = OUTPUTS_DIR / "eval"
TMP_DIR = OUTPUTS_DIR / "tmp"

for d in (OUTPUTS_DIR, LOG_DIR, EVAL_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Class id trong nhãn 4-class (đã được prepare_4class_labels.py tạo sẵn):
# 0 = ROI (mọi thứ không phải 3 lớp dưới)
# 1 = sky
# 2 = construction (building, wall, fence, guard rail, bridge, tunnel)
# 3 = nature (vegetation, terrain)
NUM_CLASSES = 4
ROI_CLASS_ID = 0
NON_ROI_CLASS_IDS = (1, 2, 3)

# Input của mạng segmentation (paper Section III-B + Fig.3)
SEG_INPUT_HW = (512, 1024)  # (H, W) – downsize từ 1024×2048 gốc
FRAME_HW = (1024, 2048)     # độ phân giải gốc Cityscapes

# Cặp CRF của paper Bảng II / Bảng IV
CRF_CONFIGS = {
    "H264":     {"codec": "libx264", "crf_roi": 23, "crf_non": 23},
    "SA-X264":  {"codec": "libx264", "crf_roi": 18, "crf_non": 27},
    "H265":     {"codec": "libx265", "crf_roi": 28, "crf_non": 28},
    "SA-X265":  {"codec": "libx265", "crf_roi": 23, "crf_non": 32},
}
