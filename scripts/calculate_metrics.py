import os
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from train_segmentation import load_segmentation_model


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resolve_existing_file(candidates):
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

# ====================== Load model để lấy mask ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = resolve_existing_file([
    os.path.join(PROJECT_ROOT, 'models', 'best_pidnet.pth'),
    os.path.join(PROJECT_ROOT, 'models', 'best_ccnet.pth'),
])
if model_path is None:
    raise FileNotFoundError("Không tìm thấy model: models/best_pidnet.pth hoặc models/best_ccnet.pth")

model, model_name = load_segmentation_model(model_path, device=device, num_classes=4)
print(f"Dùng segmentation model: {model_name} | {model_path}")

transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================== Parameters ======================
CRF_ROI = 23
CRF_NON = 32
r_roi = CRF_ROI / (CRF_ROI + CRF_NON)
r_non = CRF_NON / (CRF_ROI + CRF_NON)

VIDEO_SAC = resolve_existing_file([
    os.path.join(PROJECT_ROOT, "outputs", "compressed", "sac_x265.mp4"),
    os.path.join(PROJECT_ROOT, "outputs", "compressed", "roi.mp4"),
])
VIDEO_TRAD = os.path.join(PROJECT_ROOT, "outputs", "compressed", "traditional_x265.mp4")
IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "leftImg8bit_trainvaltest", "leftImg8bit", "val")

if not os.path.isdir(IMAGE_BASE_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {IMAGE_BASE_DIR}")
if VIDEO_SAC is None:
    raise FileNotFoundError(
        "Không tìm thấy video SAC. Cần một trong các file: "
        "outputs/compressed/sac_x265.mp4 hoặc outputs/compressed/roi.mp4"
    )
if not os.path.isfile(VIDEO_TRAD):
    raise FileNotFoundError(f"Không tìm thấy video traditional: {VIDEO_TRAD}")

print(f"Dùng video SAC: {VIDEO_SAC}")
print(f"Dùng video Traditional: {VIDEO_TRAD}")

# Lấy đúng 20 frame như compression (từ các city folders)
files = []
for city in sorted(os.listdir(IMAGE_BASE_DIR)):
    city_path = os.path.join(IMAGE_BASE_DIR, city)
    if os.path.isdir(city_path):
        for fname in sorted(os.listdir(city_path)):
            if fname.endswith('_leftImg8bit.png'):
                files.append(os.path.join(city_path, fname))
                if len(files) >= 20:
                    break
    if len(files) >= 20:
        break

print("Đang tính metrics cho 20 frames...")

metrics_data = []

cap_sac = cv2.VideoCapture(VIDEO_SAC)
cap_trad = cv2.VideoCapture(VIDEO_TRAD)

for idx, orig_path in enumerate(tqdm(files)):
    # Original frame
    orig = cv2.imread(orig_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]

    # Đọc frame từ 2 video
    ret_sac, sac_frame = cap_sac.read()
    ret_trad, trad_frame = cap_trad.read()
    if not ret_sac or not ret_trad:
        break
    sac_frame = cv2.cvtColor(sac_frame, cv2.COLOR_BGR2RGB)
    trad_frame = cv2.cvtColor(trad_frame, cv2.COLOR_BGR2RGB)

    # Dự đoán mask (ROI = 0)
    input_tensor = transform(Image.fromarray(orig)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
    mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    roi_mask = (mask == 0).astype(np.uint8)      # ROI
    non_mask = 1 - roi_mask

    # Tính metrics theo vùng
    def region_metrics(img1, img2, region_mask):
        if region_mask.sum() == 0:
            return 0, 0
        mask_3ch = np.repeat(region_mask[..., None], 3, axis=2)
        roi1 = img1 * mask_3ch
        roi2 = img2 * mask_3ch
        ps = compare_psnr(roi1, roi2, data_range=255)
        ss = compare_ssim(roi1, roi2, channel_axis=2, data_range=255)
        return ps, ss

    # Traditional metrics (toàn khung hình)
    psnr_trad = compare_psnr(orig, trad_frame, data_range=255)
    ssim_trad = compare_ssim(orig, trad_frame, channel_axis=2, data_range=255)

    # SAC metrics (toàn khung hình)
    psnr_sac = compare_psnr(orig, sac_frame, data_range=255)
    ssim_sac = compare_ssim(orig, sac_frame, channel_axis=2, data_range=255)

    # Per-region
    psnr_roi_sac, ssim_roi_sac = region_metrics(orig, sac_frame, roi_mask)
    psnr_non_sac, ssim_non_sac = region_metrics(orig, sac_frame, non_mask)

    # SA metrics (theo công thức paper)
    sa_psnr = r_non * psnr_roi_sac + r_roi * psnr_non_sac
    sa_ssim = r_non * ssim_roi_sac + r_roi * ssim_non_sac

    metrics_data.append({
        'Frame': idx+1,
        'PSNR_Trad': round(psnr_trad, 3),
        'SSIM_Trad': round(ssim_trad, 4),
        'PSNR_SAC': round(psnr_sac, 3),
        'SSIM_SAC': round(ssim_sac, 4),
        'SA-PSNR': round(sa_psnr, 3),
        'SA-SSIM': round(sa_ssim, 4),
    })

cap_sac.release()
cap_trad.release()

# ====================== In bảng so sánh ======================
df = pd.DataFrame(metrics_data)
if df.empty:
    raise RuntimeError("Không tính được metrics nào. Kiểm tra số frame và video đầu vào.")
print("\n" + "="*80)
print("📊 BẢNG SO SÁNH METRICS (SAC-X265 vs Traditional X265)")
print("="*80)
print(df.mean(numeric_only=True).round(3))

print("\nChi tiết từng frame (10 frame đầu):")
print(df.head(10).to_string(index=False))

# Lưu file CSV
metrics_csv = os.path.join(PROJECT_ROOT, "outputs", "metrics_comparison.csv")
os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
df.to_csv(metrics_csv, index=False)
print(f"\n✅ Đã lưu đầy đủ metrics vào: {metrics_csv}")

# So sánh trung bình
print(f"\n🎯 KẾT QUẢ TRUNG BÌNH:")
print(f"   SA-PSNR  SAC > Trad : {df['SA-PSNR'].mean() - df['PSNR_Trad'].mean():+.3f} dB")
print(f"   SA-SSIM  SAC > Trad : {df['SA-SSIM'].mean() - df['SSIM_Trad'].mean():+.4f}")
