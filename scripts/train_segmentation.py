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

# ====================== CCNet Module (theo paper Fig.3) ======================
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

class CCNet(nn.Module):
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

    # Đường dẫn
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "gt_4class", "val")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    IMAGE_SIZE = (512, 1024)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Cityscapes4Class(IMAGE_DIR, LABEL_DIR, transform, image_size=IMAGE_SIZE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    model = CCNet(num_classes=4).to(device)
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
    for epoch in range(10):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
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
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_ccnet.pth"))
            print(f"✅ Lưu model tốt nhất: mIOU = {best_miou:.4f}")

    print(f"✅ Hoàn thành training! Model lưu tại: {os.path.join(MODEL_DIR, 'best_ccnet.pth')}")
