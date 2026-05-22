import os
import csv
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Re-use model definitions from train_segmentation
from train_segmentation import (
    build_segmentation_model,
    CompressionArtifactAugmentation,
    infer_model_name_from_state_dict,
    _auto_batch_size,
)

NUM_CLASSES = 4
# Class ID: 0=background, 1=road, 2=vehicle, 3=pedestrian

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ====================== Dataset 4-class ======================
class Cityscapes4Class(Dataset):
    """Dataset 4-class: background(0) / road(1) / vehicle(2) / pedestrian(3)."""

    def __init__(self, image_root, label_root, transform=None, image_size=None, cities=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.image_size = image_size
        self.samples = []

        city_filter = set(cities) if cities is not None else None

        for city in sorted(os.listdir(self.label_root)):
            if city_filter is not None and city not in city_filter:
                continue

            city_label_dir = os.path.join(self.label_root, city)
            city_image_dir = os.path.join(self.image_root, city)

            if not os.path.isdir(city_label_dir) or not os.path.isdir(city_image_dir):
                continue

            for fname in os.listdir(city_label_dir):
                if not fname.endswith('_gtFine_4class.png'):
                    continue

                stem = fname.replace('_gtFine_4class.png', '')
                img_path = os.path.join(city_image_dir, f"{stem}_leftImg8bit.png")
                label_path = os.path.join(city_label_dir, fname)

                if os.path.isfile(img_path):
                    self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.image_size is not None:
            h, w = self.image_size
            image = image.resize((w, h), Image.BILINEAR)
            label = label.resize((w, h), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label)).long()
        return image, label


def load_segmentation_model_4class(model_path, device, default_model='pidnet_s'):
    """Load checkpoint 4-class (num_classes=4)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model_name = default_model
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        meta = checkpoint.get('meta', {}) if isinstance(checkpoint.get('meta', {}), dict) else {}
        model_name = meta.get('model_name', infer_model_name_from_state_dict(state_dict))
    else:
        state_dict = checkpoint
        if isinstance(state_dict, dict):
            model_name = infer_model_name_from_state_dict(state_dict)

    model = build_segmentation_model(
        model_name=model_name, num_classes=NUM_CLASSES, backbone_weights='IMAGENET1K_V1'
    ).to(device)
    model.load_state_dict(state_dict)
    if hasattr(model, 'augment'):
        model.augment = False
    model.eval()
    return model, model_name


# ====================== Main ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-class segmentation model")
    parser.add_argument(
        "--model",
        default="pidnet_s",
        choices=["pidnet_s", "pidnet_l", "ccnet"],
        help="Model to train: pidnet_s, pidnet_l, or ccnet",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = auto-detect từ VRAM GPU")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"Đang dùng: {device} - {device_name}")
    print(f"Num classes: {NUM_CLASSES}  (0=background, 1=road, 2=vehicle, 3=pedestrian)")

    IMAGE_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "leftImg8bit_trainvaltest", "leftImg8bit", "train")
    LABEL_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "train")
    IMAGE_VAL_DIR   = os.path.join(PROJECT_ROOT, "data", "gt_4class", "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    LABEL_VAL_DIR   = os.path.join(PROJECT_ROOT, "data", "gt_4class", "val")
    MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
    METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    IMAGE_SIZE = (512, 1024)

    train_transform = transforms.Compose([
        CompressionArtifactAugmentation(p=0.7, min_quality=35, max_quality=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    TEST_CITIES = ['strasbourg', 'ulm']
    all_train_cities = sorted(os.listdir(LABEL_TRAIN_DIR))
    train_cities = [c for c in all_train_cities if c not in TEST_CITIES]

    train_ds = Cityscapes4Class(IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR, train_transform, image_size=IMAGE_SIZE, cities=train_cities)
    val_ds   = Cityscapes4Class(IMAGE_VAL_DIR,   LABEL_VAL_DIR,   val_transform,   image_size=IMAGE_SIZE)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Train dataset rỗng. Hãy chạy prepare_4class_labels.py để tạo nhãn 4-class."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            "Val dataset rỗng. Hãy kiểm tra dữ liệu tại data/gt_4class/val."
        )

    print(f"Train samples: {len(train_ds)} ({len(train_cities)} thành phố, bỏ {TEST_CITIES})")
    print(f"Val   samples: {len(val_ds)}")

    model_name = args.model
    batch_size = args.batch_size if args.batch_size > 0 else _auto_batch_size(model_name)
    print(f"Batch size: {batch_size}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # CCNet dùng ImageNet pretrained backbone; PIDNet không cần
    backbone_weights = 'IMAGENET1K_V1' if model_name in ('ccnet', 'resnet101') else None
    model = build_segmentation_model(model_name=model_name, num_classes=NUM_CLASSES, backbone_weights=backbone_weights).to(device)

    if model_name.startswith("pidnet"):
        best_name = f"best_{model_name}_4class.pth"
    else:
        best_name = "best_ccnet_4class.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    metrics_csv_path  = os.path.join(METRICS_DIR, f"train_metrics_{model_name}_4class.csv")
    best_result_path  = os.path.join(METRICS_DIR, f"best_result_{model_name}_4class.json")

    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'mean_iou', 'best_miou_so_far', 'is_best', 'model_name'])

    def ce_dice_loss(pred, target):
        """CE + Dice loss cho N class."""
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            ce = nn.CrossEntropyLoss()(pred_fp32, target)
            target_onehot = F.one_hot(target, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
            pred_soft = F.softmax(pred_fp32, dim=1)
            dice = 1 - (
                (2 * (pred_soft * target_onehot).sum(dim=(2, 3)) + 1)
                / (pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) + 1)
            )
            return ce + dice.mean()

    best_miou = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = ce_dice_loss(outputs, labels)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())

        # Validate
        model.eval()
        ious = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = torch.argmax(outputs, dim=1)
                for p, t in zip(pred, labels):
                    iou_per_class = []
                    for c in range(NUM_CLASSES):
                        inter = ((p == c) & (t == c)).sum().item()
                        union = ((p == c) | (t == c)).sum().item()
                        iou_per_class.append(inter / union if union > 0 else 0)
                    ious.append(np.mean(iou_per_class))

        mean_iou = np.mean(ious)
        is_best = mean_iou > best_miou
        print(f"Epoch {epoch+1} - mIoU: {mean_iou:.4f}" + (" ✅ best" if is_best else ""))

        with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(mean_iou),
                float(max(best_miou, mean_iou)),
                int(is_best),
                model_name,
            ])

        if is_best:
            best_miou  = mean_iou
            best_epoch = epoch + 1
            model_save_path = os.path.join(MODEL_DIR, best_name)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'meta': {
                        'model_name': model_name,
                        'num_classes': NUM_CLASSES,
                        'best_miou':   float(best_miou),
                        'best_epoch':  int(best_epoch),
                    },
                },
                model_save_path,
            )
            with open(best_result_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'model_name':      model_name,
                        'best_miou':       float(best_miou),
                        'best_epoch':      int(best_epoch),
                        'num_epochs':      int(args.epochs),
                        'num_classes':     NUM_CLASSES,
                        'checkpoint_path': model_save_path,
                    },
                    f, ensure_ascii=False, indent=2,
                )

    print(f"✅ Hoàn thành training! Model lưu tại: {os.path.join(MODEL_DIR, best_name)}")
    print(f"📊 Log mIoU từng epoch: {metrics_csv_path}")
    if best_epoch > 0:
        print(f"🏆 Best: epoch={best_epoch}, mIoU={best_miou:.4f}, file={best_result_path}")
