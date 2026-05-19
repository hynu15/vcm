"""
CCNet-4Class: ResNet101 backbone + two CrissCross Attention modules.

Reproduces the segmentation network from Fig. 3 of:
  Wang et al., "Semantic-Aware Video Compression for Automotive Cameras",
  IEEE Trans. Intelligent Vehicles, Vol. 8, No. 6, 2023.

Architecture:
  RGB input → greyscale conversion (inside forward) → ResNet101 backbone
  (layer3 dilated×2, layer4 dilated×4 → stride-8 output)
  → channel reduction (2048→512) → CrissCrossAttention × 2
  → concat(reduced, cc2_out) → 4-class segmentation head
  → bilinear upsample to original resolution

Classes:
  0 = ROI  (road, vehicle, pedestrian, cyclist, unlabelled dynamic…)
  1 = sky
  2 = construction  (building, wall, fence, bridge, tunnel, …)
  3 = nature  (vegetation, terrain)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# CrissCross Attention (Huang et al., ICCV 2019, reference [38] in paper)
# ─────────────────────────────────────────────────────────────────────────────

class CrissCrossAttention(nn.Module):
    """
    Row-wise + column-wise attention.

    For stride-8 features (64×128 at 512×1024 input) full N×N attention
    (N=8192) would require ~268 M weights per batch — infeasible.
    Row+col decomposition reduces this to O(N·(H+W)) and is equivalent in
    expressiveness when two CCA blocks are stacked (as in the paper).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        c8 = max(in_channels // 8, 1)
        self.q = nn.Conv2d(in_channels, c8, 1, bias=False)
        self.k = nn.Conv2d(in_channels, c8, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        c8 = self.q.out_channels
        scale = c8 ** -0.5

        Q = self.q(x)  # [B, c8, H, W]
        K = self.k(x)  # [B, c8, H, W]
        V = self.v(x)  # [B,  C, H, W]

        # Row-wise: each pixel attends to all pixels in its row
        Q_r = Q.permute(0, 2, 1, 3).reshape(B * H, c8, W)  # [B*H, c8, W]
        K_r = K.permute(0, 2, 1, 3).reshape(B * H, c8, W)
        V_r = V.permute(0, 2, 1, 3).reshape(B * H,  C, W)
        attn_r = F.softmax(torch.bmm(Q_r.permute(0, 2, 1), K_r) * scale, dim=-1)  # [B*H, W, W]
        out_r = torch.bmm(V_r, attn_r.permute(0, 2, 1))                           # [B*H, C, W]
        out_r = out_r.reshape(B, H, C, W).permute(0, 2, 1, 3)                     # [B, C, H, W]

        # Column-wise: each pixel attends to all pixels in its column
        Q_c = Q.permute(0, 3, 1, 2).reshape(B * W, c8, H)  # [B*W, c8, H]
        K_c = K.permute(0, 3, 1, 2).reshape(B * W, c8, H)
        V_c = V.permute(0, 3, 1, 2).reshape(B * W,  C, H)
        attn_c = F.softmax(torch.bmm(Q_c.permute(0, 2, 1), K_c) * scale, dim=-1)  # [B*W, H, H]
        out_c = torch.bmm(V_c, attn_c.permute(0, 2, 1))                           # [B*W, C, H]
        out_c = out_c.reshape(B, W, C, H).permute(0, 2, 3, 1)                     # [B, C, H, W]

        return self.gamma * (out_r + out_c) + x


# ─────────────────────────────────────────────────────────────────────────────
# Full CCNet-4Class model
# ─────────────────────────────────────────────────────────────────────────────

class CCNet4Class(nn.Module):
    """
    Paper-faithful CCNet for 4-class semantic segmentation.

    Key differences from a vanilla ResNet101 FCN:
      • layer3 / layer4 use dilated convolutions (dilation 2 / 4) so the
        backbone outputs stride-8 features (64×128 for 512×1024 input)
        instead of stride-32 (16×32).  This is standard for dense-prediction
        heads (DeepLab, CCNet original, etc.).
      • Two stacked CrissCrossAttention modules on the stride-8 features.
      • Input converted to greyscale before the backbone (paper Fig. 3).

    Usage
    -----
    model = CCNet4Class(num_classes=4, pretrained=True)
    logits = model(rgb_tensor)   # [B, 4, H, W]
    pred   = logits.argmax(1)    # [B, H, W]  values in {0,1,2,3}
    """

    TRAIN_SIZE = (512, 1024)

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── ResNet-101 backbone ──────────────────────────────────────────────
        weights = "IMAGENET1K_V1" if pretrained else None
        bb = models.resnet101(weights=weights)

        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # stride 4,  out_ch=256
        self.layer2 = bb.layer2   # stride 8,  out_ch=512
        self.layer3 = bb.layer3   # → dilated×2, stays stride-8,  out_ch=1024
        self.layer4 = bb.layer4   # → dilated×4, stays stride-8,  out_ch=2048

        # Replace stride with dilation so spatial resolution stays at H/8×W/8
        self._make_dilated(self.layer3, dilation=2)
        self._make_dilated(self.layer4, dilation=4)

        # ── Channel reduction: 2048 → 512 ───────────────────────────────────
        self.reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Two CrissCross Attention modules ────────────────────────────────
        self.cc1 = CrissCrossAttention(512)
        self.cc2 = CrissCrossAttention(512)

        # ── Segmentation head: concat(reduced=512, cc2=512) → num_classes ───
        self.seg_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_dilated(layer: nn.Sequential, dilation: int) -> None:
        """
        Convert a ResNet layer to dilated convolutions.
        Sets stride=1 and dilation on every block's 3×3 conv and its downsample.
        Weight shapes are unchanged — existing checkpoints remain loadable.
        """
        for block in layer:
            # 3×3 conv in Bottleneck is conv2
            block.conv2.dilation = (dilation, dilation)
            block.conv2.padding  = (dilation, dilation)
            block.conv2.stride   = (1, 1)
            # Remove stride from projection shortcut (first block only)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)

    @staticmethod
    def _to_grey3ch(x: torch.Tensor) -> torch.Tensor:
        """Convert RGB → greyscale, replicated across 3 channels (paper Fig. 3)."""
        grey = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return grey.expand(-1, 3, -1, -1)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 3, H, W] float32, ImageNet-normalised RGB

        Returns
        -------
        logits : [B, num_classes, H, W]
        """
        H_in, W_in = x.shape[-2:]

        x = self._to_grey3ch(x)

        # Backbone: stride-8 output thanks to dilated layer3/layer4
        x = self.stem(x)       # [B,   64, H/4,  W/4 ]
        x = self.layer1(x)     # [B,  256, H/4,  W/4 ]
        x = self.layer2(x)     # [B,  512, H/8,  W/8 ]
        x = self.layer3(x)     # [B, 1024, H/8,  W/8 ]  (dilated×2, no stride)
        x = self.layer4(x)     # [B, 2048, H/8,  W/8 ]  (dilated×4, no stride)

        x_red = self.reduction(x)          # [B, 512, H/8, W/8]

        h1 = self.cc1(x_red)
        h2 = self.cc2(h1)

        fused = torch.cat([x_red, h2], dim=1)  # [B, 1024, H/8, W/8]
        out   = self.seg_head(fused)            # [B, num_classes, H/8, W/8]

        # Upsample ×8 (vs ×32 before) — much finer output
        out = F.interpolate(out, size=(H_in, W_in), mode="bilinear", align_corners=True)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load a saved checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_ccnet_4class(ckpt_path: str, device: torch.device) -> CCNet4Class:
    """Load a CCNet4Class checkpoint saved by train.py."""
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt.get("meta", {}).get("num_classes", 4)
    model = CCNet4Class(num_classes=num_classes, pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model
