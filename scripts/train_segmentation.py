import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ====================== Legacy CCNet (ResNet101) ======================
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B, N, C'
        proj_key = self.key_conv(x).view(B, -1, H * W)                     # B, C', N
        energy = torch.bmm(proj_query, proj_key)                           # B, N, N
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)                 # B, C, N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class LegacyCCNetResNet101(nn.Module):
    def __init__(self, num_classes=4, backbone_weights='IMAGENET1K_V1'):
        super().__init__()
        resnet = models.resnet101(weights=backbone_weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # ResNet101 without last FC + avgpool
        self.cc1 = CrissCrossAttention(2048)
        self.cc2 = CrissCrossAttention(2048)
        self.conv = nn.Conv2d(2048, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)          # [B, 2048, H/32, W/32]
        x = self.cc1(x)
        x = self.cc2(x)
        x = self.conv(x)              # [B, 4, H/32, W/32]
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)
        return x


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

    def __init__(self, num_classes=4, channels=32):
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

    def __init__(self, num_classes=4, backbone_weights=None):
        super().__init__()
        self.model = PIDNetSegmentor(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def build_segmentation_model(model_name='pidnet_s', num_classes=4, backbone_weights=None):
    name = model_name.lower()
    if name in ('pidnet', 'pidnet_s', 'pidnet-small'):
        return PIDNetSegmentor(num_classes=num_classes)
    if name in ('ccnet', 'resnet101', 'ccnet_resnet101'):
        return LegacyCCNetResNet101(num_classes=num_classes, backbone_weights=backbone_weights)
    raise ValueError(f"Unknown model_name={model_name}")


def infer_model_name_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith('backbone.') or k.startswith('cc1.') or k.startswith('cc2.') for k in keys):
        return 'ccnet'
    return 'pidnet_s'


def load_segmentation_model(model_path, device, num_classes=4, default_model='pidnet_s'):
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

    model = build_segmentation_model(model_name=model_name, num_classes=num_classes, backbone_weights=None)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, model_name

# ====================== Dataset ======================
class Cityscapes4Class(Dataset):
    def __init__(self, image_root, label_root, transform=None, image_size=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.image_size = image_size
        self.samples = []

        for city in sorted(os.listdir(self.label_root)):
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

# ====================== Main ======================
if __name__ == "__main__":
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    IMAGE_SIZE = (512, 1024)
    NUM_EPOCHS = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = Cityscapes4Class(IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR, transform, image_size=IMAGE_SIZE)
    val_ds = Cityscapes4Class(IMAGE_VAL_DIR, LABEL_VAL_DIR, transform, image_size=IMAGE_SIZE)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Train dataset rỗng. Hãy chuẩn bị nhãn 4-class cho split train tại data/gt_4class/train."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            "Val dataset rỗng. Hãy kiểm tra dữ liệu tại data/gt_4class/val."
        )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    model_name = 'pidnet_s'
    model = build_segmentation_model(model_name=model_name, num_classes=4, backbone_weights=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Loss theo paper (BCE + Dice)
    def bce_dice_loss(pred, target):
        bce = nn.CrossEntropyLoss()(pred, target)
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=4).permute(0,3,1,2).float()
        dice = 1 - (2 * (pred_soft * target_onehot).sum(dim=(2,3)) + 1) / (pred_soft.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3)) + 1)
        return bce + dice.mean()

    best_miou = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = bce_dice_loss(outputs, labels)
            
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
                    for c in range(4):
                        inter = ((p == c) & (t == c)).sum().item()
                        union = ((p == c) | (t == c)).sum().item()
                        iou.append(inter / union if union > 0 else 0)
                    ious.append(np.mean(iou))
        mean_iou = np.mean(ious)
        print(f"Epoch {epoch+1} - mIOU: {mean_iou:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'meta': {
                        'model_name': model_name,
                        'num_classes': 4,
                    },
                },
                os.path.join(MODEL_DIR, 'best_pidnet.pth'),
            )
            print(f"✅ Lưu model tốt nhất: mIOU = {best_miou:.4f}")

    print(f"✅ Hoàn thành training! Model lưu tại: {os.path.join(MODEL_DIR, 'best_pidnet.pth')}")
