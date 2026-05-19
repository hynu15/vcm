# scripts/new_branch — SAC reproduction (Wang et al., IEEE TIV 2023)

Tái hiện chính xác bài báo **Semantic-Aware Video Compression for Automotive
Cameras**, không cải tiến gì. Mọi tham số khớp với CLAUDE.md §3-§9.

## Tổ chức file

| File | Vai trò | Section paper |
|------|---------|----------------|
| `paths.py` | Hằng số đường dẫn, class id, cặp CRF | – |
| `dataset.py` | `CityscapesSAC` (4-class, grayscale-3ch, ImageNet norm) | §III-B.1 |
| `model.py` | Wrapper `Seg_Model` của CCNet (ResNet-101 dilated + RCCA R=2 + DSN) | §III-B.2-3 |
| `losses.py` | `BCEDiceWithAux` (main + 0.4×aux) | §III-B.4 (Eq.9-12) |
| `train.py` | Vòng train Adam lr=3e-4, batch 4, 47 epoch | §III-B.4 |
| `stream_separation.py` | Khối S2: binary mask → upsample → MB 16×16 → S_i/S_n | §III-C (Hình 4) |
| `compression.py` | Khối S3: FFmpeg libx264/libx265 hai luồng | §III-D (Eq.5-7) |
| `metrics.py` | PSNR/SSIM regional, SA-PSNR, SA-SSIM, mIoU, iIoU | §III-E (Eq.11-17) |
| `evaluate.py` | Reproduce Bảng II + Bảng IV trên test split | §IV |

## Phụ thuộc

- `$PROJECT_ROOT/scripts/CCNet/` (clone + patch `InPlaceABN` → `BN+ReLU`) — đặt trong `scripts/` để sync chung khi đẩy lên RunPod.
- `$PROJECT_ROOT/models/resnet101-imagenet.pth` (MIT CSAIL pretrained, 171MB)
- Test split = 2 thành phố `strasbourg` + `ulm` (460 ảnh), được tách thủ công khỏi
  train vì test gốc của Cityscapes không có nhãn.

## Cách chạy

> **QUAN TRỌNG**: luôn chạy bằng `python -m scripts.new_branch.<module>` từ
> thư mục project root (`/workspace/sac_project` hoặc `/home/huy/sac_project`).
> Không `cd scripts/new_branch && python train.py` — các module dùng relative
> import, sẽ báo `attempted relative import with no known parent package`.

```bash
conda activate sac
cd /workspace/sac_project        # hoặc /home/huy/sac_project trên máy local

# 0) Setup CCNet + pretrained (chạy MỘT LẦN trên mỗi máy mới):
python -m scripts.new_branch.setup_ccnet

# 1) Train 47 epoch (RTX 4090): khoảng 3-5 giờ. AMP bật mặc định.
python -m scripts.new_branch.train --epochs 47 --batch-size 4

# 2) Eval đầy đủ 4 phương pháp trên test split:
python -m scripts.new_branch.evaluate --split test

# 2b) Eval nhanh trên 20 frame để kiểm tra pipeline:
python -m scripts.new_branch.evaluate --split test --num-frames 20 --gop 10

# 2c) Chỉ 1 phương pháp:
python -m scripts.new_branch.evaluate --split test --methods SA-X264
```

## Output mong đợi (theo paper)

| Bảng | Method | C_roi | C_non | SA-PSNR | SA-SSIM | mIoU | iIoU |
|-----|--------|-------|-------|---------|---------|------|------|
| II/IV | H.264 | 23 | 23 | 45.18 | 0.98 | 87.86 | 92.00 |
| II/IV | **SA-X264** | **18** | **27** | **48.05** | **0.99** | **90.56** | **92.45** |
| II/IV | H.265 | 28 | 28 | 43.80 | 0.98 | 85.71 | 91.43 |
| II/IV | SA-X265 | 23 | 32 | 47.01 | 0.99 | 87.48 | 92.04 |

mIoU train/val/test paper: 87.97% / 80.89% / 79.08%.
