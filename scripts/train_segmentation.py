import io
import os
import csv
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from PIL import ImageFilter
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ====================== Legacy CCNet (ResNet101) ======================

def _make_layer_dilated(layer, dilation):
    """Remove stride and apply dilation to all 3×3 convs in a ResNet layer.
    Keeps spatial resolution at H/8 instead of H/32 (matches CCNet paper).
    """
    for i, block in enumerate(layer):
        # First block carries stride=2 in conv2 and downsample → remove it
        if i == 0 and block.downsample is not None:
            block.downsample[0].stride = (1, 1)
        block.conv2.stride = (1, 1)
        block.conv2.dilation = (dilation, dilation)
        block.conv2.padding = (dilation, dilation)


class CrissCrossAttention(nn.Module):
    """True Criss-Cross Attention (Huang et al., CCNet ICCV 2019).

    Each pixel attends only to the H+W-1 neighbors in its row and column,
    not to all N=H*W pixels. Applying this module twice gives full-image
    connectivity with far less memory than global self-attention.

    The H-direction diagonal is masked with -inf to avoid counting the
    anchor pixel twice (it also appears in the W-direction energy).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        Cp = C // 8  # query/key channels

        with torch.amp.autocast('cuda', enabled=False):
            xf = x.float()
            Q = self.query_conv(xf)   # [B, Cp, H, W]
            K = self.key_conv(xf)     # [B, Cp, H, W]
            V = self.value_conv(xf)   # [B, C,  H, W]

            # H-direction: for each of W columns, attend over H positions
            Q_H = Q.permute(0, 3, 1, 2).reshape(B*W, Cp, H).permute(0, 2, 1)  # [B*W, H, Cp]
            K_H = K.permute(0, 3, 1, 2).reshape(B*W, Cp, H)                   # [B*W, Cp, H]
            V_H = V.permute(0, 3, 1, 2).reshape(B*W,  C, H)                   # [B*W, C,  H]

            # W-direction: for each of H rows, attend over W positions
            Q_W = Q.permute(0, 2, 1, 3).reshape(B*H, Cp, W).permute(0, 2, 1)  # [B*H, W, Cp]
            K_W = K.permute(0, 2, 1, 3).reshape(B*H, Cp, W)                   # [B*H, Cp, W]
            V_W = V.permute(0, 2, 1, 3).reshape(B*H,  C, W)                   # [B*H, C,  W]

            E_H = torch.bmm(Q_H, K_H)   # [B*W, H, H]
            E_W = torch.bmm(Q_W, K_W)   # [B*H, W, W]

            # Mask diagonal of E_H so each pixel is counted only once
            # (it will still attend to itself via E_W)
            diag_inf = torch.zeros(H, H, device=x.device, dtype=torch.float32)
            diag_inf.fill_diagonal_(float('inf'))
            E_H = E_H - diag_inf.unsqueeze(0)   # [B*W, H, H]

            # Reshape and concatenate, then apply joint softmax over H+W neighbors
            E_H = E_H.view(B, W, H, H).permute(0, 2, 1, 3)  # [B, H, W, H]
            E_W = E_W.view(B, H, W, W)                       # [B, H, W, W]
            A = F.softmax(torch.cat([E_H, E_W], dim=3), dim=3)  # [B, H, W, H+W]

            A_H = A[:, :, :, :H].permute(0, 2, 1, 3).reshape(B*W, H, H)  # [B*W, H, H]
            A_W = A[:, :, :, H:].reshape(B*H, W, W)                       # [B*H, W, W]

            out_H = torch.bmm(V_H, A_H.permute(0, 2, 1)).view(B, W, C, H).permute(0, 2, 3, 1)
            out_W = torch.bmm(V_W, A_W.permute(0, 2, 1)).view(B, H, C, W).permute(0, 2, 1, 3)

            out = self.gamma.float() * (out_H + out_W) + xf

        return out.to(x.dtype)


class LegacyCCNetResNet101(nn.Module):
    def __init__(self, num_classes=2, backbone_weights='IMAGENET1K_V1'):
        super().__init__()
        resnet = models.resnet101(weights=backbone_weights)

        # Dilated backbone: layer3 dilation=2, layer4 dilation=4
        # → output at H/8 (not H/32), preserving 16× more spatial detail
        _make_layer_dilated(resnet.layer3, dilation=2)
        _make_layer_dilated(resnet.layer4, dilation=4)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        # Reduction: 2048 → 512 before CCA (matches CCNet paper Fig. 3)
        self.reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.cc1 = CrissCrossAttention(512)
        self.cc2 = CrissCrossAttention(512)
        self.head = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feat = self.backbone(x)       # [B, 2048, H/8, W/8]
        r    = self.reduction(feat)   # [B, 512,  H/8, W/8]  ← R(X)
        h    = self.cc2(self.cc1(r))  # [B, 512,  H/8, W/8]  ← H''
        out  = self.head(r + h)       # [B, num_classes, H/8, W/8]  ← f(R(X)+H'')
        return F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)


# ====================== Lightweight PID-style Segmentor ======================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, groups=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=stride, dilation=dilation, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class PIDNetSegmentor(nn.Module):
    """
    Lightweight PID-style segmentation network:
    - P branch: high-resolution detail cues
    - I branch: low-resolution semantic context
    - D branch: boundary/detail enhancement
    """

    def __init__(self, num_classes=2, channels=32):
        super().__init__()
        p_ch = channels
        i_ch = channels * 4
        d_ch = channels * 2

        # Shared stem to 1/4 resolution.
        self.stem = nn.Sequential(
            ConvBNReLU(3, channels, kernel_size=3, stride=2),
            ConvBNReLU(channels, channels * 2, kernel_size=3, stride=2),
        )

        # P (detail) branch keeps 1/4 resolution.
        self.p_branch = nn.Sequential(
            ConvBNReLU(channels * 2, p_ch, kernel_size=3, stride=1),
            ConvBNReLU(p_ch, p_ch, kernel_size=3, stride=1),
        )

        # I (context) branch goes to 1/8 resolution.
        self.i_branch = nn.Sequential(
            DepthwiseSeparableConv(channels * 2, i_ch // 2, stride=2),
            DepthwiseSeparableConv(i_ch // 2, i_ch, stride=1),
            DepthwiseSeparableConv(i_ch, i_ch, stride=1, dilation=2),
        )

        # D (boundary/detail enhancement) branch.
        self.d_branch = nn.Sequential(
            ConvBNReLU(channels * 2, d_ch, kernel_size=3, stride=1, dilation=2),
            ConvBNReLU(d_ch, d_ch, kernel_size=3, stride=1),
        )

        self.i_to_p = ConvBNReLU(i_ch, p_ch, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNReLU(p_ch + p_ch + d_ch, channels * 3, kernel_size=3),
            nn.Conv2d(channels * 3, num_classes, kernel_size=1),
        )

    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        feat_1_4 = self.stem(x)

        p_feat = self.p_branch(feat_1_4)
        d_feat = self.d_branch(feat_1_4)

        i_feat = self.i_branch(feat_1_4)
        i_up = F.interpolate(self.i_to_p(i_feat), size=p_feat.shape[-2:], mode='bilinear', align_corners=False)

        fused = torch.cat([p_feat, i_up, d_feat], dim=1)
        logits = self.fuse(fused)
        logits = F.interpolate(logits, size=(in_h, in_w), mode='bilinear', align_corners=False)
        return logits


class CCNet(nn.Module):
    """Backward-compatible alias: old code imports CCNet, now default to PIDNetSegmentor."""

    def __init__(self, num_classes=2, backbone_weights=None):
        super().__init__()
        self.model = PIDNetSegmentor(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def build_segmentation_model(model_name='pidnet_s', num_classes=2, backbone_weights=None):
    name = model_name.lower()
    if name in ('pidnet', 'pidnet_s', 'pidnet-small'):
        return PIDNetSegmentor(num_classes=num_classes, channels=32)
    if name in ('pidnet_l', 'pidnet-large'):
        from train_pidnet_l import build_pidnet_l
        return build_pidnet_l(num_classes=num_classes)
    if name in ('ccnet', 'resnet101', 'ccnet_resnet101', 'ccnet_4class'):
        return LegacyCCNetResNet101(num_classes=num_classes, backbone_weights=backbone_weights)
    raise ValueError(f"Unknown model_name={model_name}")


def infer_model_name_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    # Known CCNet checkpoint layouts use explicit prefixes
    if any(k.startswith('backbone.') or k.startswith('cc1.') or k.startswith('cc2.') for k in keys):
        return 'ccnet'
    # Official PIDNet-L has dfm/spp/seghead keys
    if any(k.startswith('dfm.') or k.startswith('spp.') or k.startswith('seghead_') for k in keys):
        return 'pidnet_l'
    # Older / raw ResNet-based checkpoints may expose standard ResNet layer names
    if any(k.startswith('conv1') or k.startswith('layer1') or k.startswith('layer2') or k.startswith('layer3') or k.startswith('layer4') for k in keys):
        return 'ccnet'
    return 'pidnet_s'


def load_segmentation_model(model_path, device, num_classes=2, default_model='pidnet_s'):
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

    model = build_segmentation_model(model_name=model_name, num_classes=num_classes, backbone_weights='IMAGENET1K_V1').to(device)
    model.load_state_dict(state_dict)
    # PIDNet-L returns a list of 3 tensors when augment=True (training mode).
    # Set augment=False so forward returns a single logit tensor at inference.
    if hasattr(model, 'augment'):
        model.augment = False
    model = model.to(device)
    model.eval()
    return model, model_name

# ====================== Dataset ======================
class CityscapesROI(Dataset):
    """Dataset 2-class: ROI (0) vs non_ROI (1) cho bài toán VCM xe tự hành."""

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
                if not fname.endswith('_gtFine_2class.png'):
                    continue

                stem = fname.replace('_gtFine_2class.png', '')
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


# Backward-compatible alias
Cityscapes4Class = CityscapesROI


class CompressionArtifactAugmentation:
    """Approximate decoded-video artifacts with a light JPEG re-encode and resize jitter."""

    def __init__(self, p: float = 0.7, min_quality: int = 35, max_quality: int = 90):
        self.p = p
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        width, height = image.size
        work = image

        if random.random() < 0.9:
            scale = random.uniform(0.6, 1.0)
            down_width = max(64, int(width * scale))
            down_height = max(64, int(height * scale))
            work = work.resize((down_width, down_height), Image.BICUBIC)
            work = work.resize((width, height), Image.BICUBIC)

        buffer = io.BytesIO()
        jpeg_quality = random.randint(self.min_quality, self.max_quality)
        work.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=2, optimize=False)
        buffer.seek(0)
        work = Image.open(buffer).convert("RGB")

        if random.random() < 0.5:
            work = work.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

        if random.random() < 0.3:
            contrast = random.uniform(0.95, 1.05)
            work = Image.fromarray(np.clip(np.array(work, dtype=np.float32) * contrast, 0, 255).astype(np.uint8))

        return work

def _auto_batch_size(model_name: str) -> int:
    """Chọn batch size dựa trên model và VRAM GPU."""
    if model_name not in ('ccnet', 'resnet101'):
        return 4  # PIDNet nhẹ, batch=4 thoải mái
    if not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram_gb >= 20:   # RTX 4090 (24 GB), A5000 (24 GB)
        return 4
    elif vram_gb >= 12:  # RTX 3080 Ti (12 GB), A4000 (16 GB)
        return 2
    return 1             # GPU nhỏ hơn


# ====================== Main ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--model",
        default="ccnet",
        choices=["pidnet_s", "pidnet_l", "ccnet"],
        help="Model to train: pidnet_s, pidnet_l, or ccnet",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = auto-detect từ VRAM GPU")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"Đang dùng: {device} - {device_name}")

    # Đường dẫn split chuẩn: train để học, val để đánh giá
    IMAGE_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "leftImg8bit_trainvaltest", "leftImg8bit", "train")
    LABEL_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "train")
    IMAGE_VAL_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    LABEL_VAL_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "val")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    IMAGE_SIZE = (512, 1024)
    NUM_EPOCHS = args.epochs

    train_transform = transforms.Compose([
        CompressionArtifactAugmentation(p=0.7, min_quality=35, max_quality=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Strasbourg và Ulm được giữ lại làm test set — loại khỏi train
    TEST_CITIES = ['strasbourg', 'ulm']
    all_train_cities = sorted(os.listdir(LABEL_TRAIN_DIR))
    train_cities = [c for c in all_train_cities if c not in TEST_CITIES]

    train_ds = CityscapesROI(IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR, train_transform, image_size=IMAGE_SIZE, cities=train_cities)
    val_ds   = CityscapesROI(IMAGE_VAL_DIR,   LABEL_VAL_DIR,   val_transform,   image_size=IMAGE_SIZE)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Train dataset rỗng. Hãy chạy prepare_2class_labels.py để tạo nhãn 2-class."
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # CCNet dùng ImageNet pretrained backbone; PIDNet không cần
    backbone_weights = 'IMAGENET1K_V1' if model_name in ('ccnet', 'resnet101') else None
    model = build_segmentation_model(model_name=model_name, num_classes=2, backbone_weights=backbone_weights).to(device)
    best_name = "best_pidnet.pth" if model_name.startswith("pidnet") else "best_ccnet.pth"
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    metrics_csv_path = os.path.join(METRICS_DIR, f"train_metrics_{model_name}.csv")
    best_result_path = os.path.join(METRICS_DIR, f"best_result_{model_name}.json")

    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'mean_iou',
            'best_miou_so_far',
            'is_best',
            'model_name',
        ])

    # Loss theo paper (BCE + Dice)
    def bce_dice_loss(pred, target):
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            target_onehot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
            bce = nn.CrossEntropyLoss()(pred_fp32, target)
            pred_soft = F.softmax(pred_fp32, dim=1)
            dice = 1 - (
                (2 * (pred_soft * target_onehot).sum(dim=(2, 3)) + 1)
                / (pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) + 1)
            )
            return bce + dice.mean()

    best_miou = 0
    best_epoch = -1
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = bce_dice_loss(outputs, labels)

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
                    iou = []
                    for c in range(2):
                        inter = ((p == c) & (t == c)).sum().item()
                        union = ((p == c) | (t == c)).sum().item()
                        iou.append(inter / union if union > 0 else 0)
                    ious.append(np.mean(iou))
        mean_iou = np.mean(ious)
        print(f"Epoch {epoch+1} - mIOU: {mean_iou:.4f}")

        is_best = mean_iou > best_miou

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
            best_miou = mean_iou
            best_epoch = epoch + 1
            model_save_path = os.path.join(MODEL_DIR, best_name)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'meta': {
                        'model_name': model_name,
                        'num_classes': 2,
                        'best_miou': float(best_miou),
                        'best_epoch': int(best_epoch),
                    },
                },
                model_save_path,
            )

            with open(best_result_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'model_name': model_name,
                        'best_miou': float(best_miou),
                        'best_epoch': int(best_epoch),
                        'num_epochs': int(NUM_EPOCHS),
                        'checkpoint_path': model_save_path,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"✅ Lưu model tốt nhất: mIOU = {best_miou:.4f}")

    print(f"✅ Hoàn thành training! Model lưu tại: {os.path.join(MODEL_DIR, best_name)}")
    print(f"📊 Log mIOU từng epoch: {metrics_csv_path}")
    if best_epoch > 0:
        print(f"🏆 Best result: epoch={best_epoch}, mIOU={best_miou:.4f}, file={best_result_path}")
