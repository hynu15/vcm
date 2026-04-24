import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import subprocess
from train_segmentation import load_segmentation_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ====================== Load model ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(PROJECT_ROOT, 'models', 'best_best_ccnet.pth')
if not os.path.isfile(model_path):
    legacy_path = os.path.join(PROJECT_ROOT, 'models', 'best_ccnet.pth')
    if not os.path.isfile(legacy_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path} hoặc {legacy_path}")
    model_path = legacy_path

model, model_name = load_segmentation_model(model_path, device=device, num_classes=4)
print(f"Dùng segmentation model: {model_name} | {model_path}")

transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def run_ffmpeg(cmd, step_name):
    """Run an ffmpeg command and stop immediately on failure with full stderr."""
    print(f"[FFMPEG] {step_name}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[FFMPEG][ERROR] {step_name} failed with exit code {result.returncode}")
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"FFmpeg failed at step: {step_name}")

# ====================== Parameters ======================
CRF_ROI = 23      # thấp = chất lượng cao (theo paper)
CRF_NON = 32      # cao = nén mạnh
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'compressed')
TMP_DIR = os.path.join(OUTPUT_DIR, 'tmp_frames')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Dùng toàn bộ split val để đánh giá compression; có thể giới hạn lại khi cần test nhanh.
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'gt_4class', 'leftImg8bit_trainvaltest', 'leftImg8bit', 'val')
files = []
for city in sorted(os.listdir(IMAGE_DIR)):
    city_dir = os.path.join(IMAGE_DIR, city)
    if not os.path.isdir(city_dir):
        continue
    for f in sorted(os.listdir(city_dir)):
        if f.endswith('_leftImg8bit.png'):
            files.append(os.path.join(city_dir, f))

MAX_FRAMES = None  # đặt số nguyên nếu muốn giới hạn, ví dụ 20 để chạy thử nhanh
if MAX_FRAMES is not None:
    files = files[:MAX_FRAMES]

print(f"Đang xử lý {len(files)} frames với SAC-X265...")

original_frames = []
sac_frames = []
traditional_frames = []

for idx, img_path in enumerate(files):
    orig = Image.open(img_path).convert('RGB')
    orig_np = np.array(orig)
    original_frames.append(orig_np)

    # 1. Dự đoán segmentation
    input_tensor = transform(orig).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
    mask = torch.argmax(pred, dim=1)[0].cpu().numpy()  # 0=ROI, 1=sky, 2=construction, 3=nature
    # Bring prediction mask back to original image size for OpenCV bitwise ops.
    mask = cv2.resize(mask.astype(np.uint8), (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    roi_mask = (mask == 0).astype(np.uint8) * 255

    # 2. Tạo 2 stream (ROI & non-ROI)
    roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi_mask)
    non_img = cv2.bitwise_and(orig_np, orig_np, mask=255 - roi_mask)

    # Lưu tạm để nén bằng ffmpeg
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}_orig.png"), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}_roi.png"), cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}_non.png"), cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))

# 3. Nén 2 stream bằng X265
print("Đang nén ROI (CRF=", CRF_ROI, ") và non-ROI (CRF=", CRF_NON, ")...")

run_ffmpeg(
    [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(TMP_DIR, "frame_%04d_roi.png"),
        "-c:v",
        "libx265",
        "-crf",
        str(CRF_ROI),
        "-preset",
        "medium",
        os.path.join(OUTPUT_DIR, "roi.mp4"),
    ],
    "Encode ROI stream",
)
run_ffmpeg(
    [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(TMP_DIR, "frame_%04d_non.png"),
        "-c:v",
        "libx265",
        "-crf",
        str(CRF_NON),
        "-preset",
        "medium",
        os.path.join(OUTPUT_DIR, "nonroi.mp4"),
    ],
    "Encode non-ROI stream",
)

# 4. Giải nén và merge
run_ffmpeg(
    [
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(OUTPUT_DIR, "roi.mp4"),
        "-i",
        os.path.join(OUTPUT_DIR, "nonroi.mp4"),
        "-filter_complex",
        "[0:v][1:v]blend=all_mode=addition",
        os.path.join(OUTPUT_DIR, "sac_x265.mp4"),
    ],
    "Merge ROI and non-ROI into SAC-X265",
)

# 5. Traditional X265 (cùng tổng bitrate)
total_crf = int((CRF_ROI + CRF_NON)/2)   # xấp xỉ cùng tỷ lệ nén
run_ffmpeg(
    [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(TMP_DIR, "frame_%04d_orig.png"),
        "-c:v",
        "libx265",
        "-crf",
        str(total_crf),
        "-preset",
        "medium",
        os.path.join(OUTPUT_DIR, "traditional_x265.mp4"),
    ],
    "Encode traditional X265 stream",
)

print("✅ Hoàn thành nén!")
print(f"   SAC-X265 video: {OUTPUT_DIR}/sac_x265.mp4")
print(f"   Traditional X265: {OUTPUT_DIR}/traditional_x265.mp4")
