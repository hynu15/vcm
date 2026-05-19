# ------------------------------------------------------------------------------
# Official PIDNet-L training script for 2-class semantic segmentation
# (ROI vs non-ROI) on Cityscapes for VCM autonomous driving.
#
# Architecture: PIDNet-L (m=3, n=4, planes=64, ppm_planes=112, head_planes=256)
# Reference: Xu et al., "PIDNet: A Real-time Semantic Segmentation Network
#            Inspired from PID Controller", CVPR 2023.
#
# Key differences from lightweight PIDNetSegmentor in train_segmentation.py:
#   - Full three-branch P/I/D architecture from official code
#   - DAPPM (Dense Aggregated PPM) for large model
#   - Bag fusion module (not Light_Bag)
#   - Four-component loss: CE(P) + BCE(D boundary) + CE(main) + BAS-loss
#   - SGD with poly-LR + warmup (paper section 4.2)
# ------------------------------------------------------------------------------

import os
import csv
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import dataset and augmentation from existing module (sibling import)
from train_segmentation import CityscapesROI, CompressionArtifactAugmentation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ==============================================================================
# Official PIDNet building blocks (verbatim from Xu et al. CVPR 2023)
# https://github.com/XuJiacong/PIDNet
# ==============================================================================

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False  # align_corners=False as in official code


class BasicBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block with 1x3x1 pattern and expansion=2."""
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmenthead(nn.Module):
    """Segmentation head: BN -> Conv3x3 -> BN -> Conv1x1, optional upsample."""

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width],
                                mode='bilinear', align_corners=algc)
        return out


class DAPPM(nn.Module):
    """Densely Aggregated Pyramid Pooling Module (used in PIDNet-L)."""

    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.process1 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process2 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process3 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process4 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False))

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(self.process1(
            F.interpolate(self.scale1(x), size=[height, width],
                          mode='bilinear', align_corners=algc) + x_list[0]))
        x_list.append(self.process2(
            F.interpolate(self.scale2(x), size=[height, width],
                          mode='bilinear', align_corners=algc) + x_list[1]))
        x_list.append(self.process3(
            F.interpolate(self.scale3(x), size=[height, width],
                          mode='bilinear', align_corners=algc) + x_list[2]))
        x_list.append(self.process4(
            F.interpolate(self.scale4(x), size=[height, width],
                          mode='bilinear', align_corners=algc) + x_list[3]))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class PagFM(nn.Module):
    """Pixel-Attention Guided Feature Merge module."""

    def __init__(self, in_channels, mid_channels, after_relu=False,
                 with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels))
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels))
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y
        return x


class Bag(nn.Module):
    """Boundary-Aware Gate (full version, used in PIDNet-L)."""

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()
        self.conv = nn.Sequential(
            BatchNorm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


# ==============================================================================
# PIDNet model class (official architecture, adapted for num_classes=2)
# When augment=True, forward() returns [x_extra_p, x_, x_extra_d]
#   x_extra_p : P-branch auxiliary logits  [B, num_classes, H/8, W/8]
#   x_         : main semantic logits       [B, num_classes, H/8, W/8]
#   x_extra_d  : D-branch boundary logits  [B, 1,           H/8, W/8]
# ==============================================================================

class PIDNet(nn.Module):
    """Official PIDNet architecture (Xu et al., CVPR 2023).

    For PIDNet-L: m=3, n=4, planes=64, ppm_planes=112, head_planes=256.
    Uses DAPPM (not PAPPM) and Bag (not Light_Bag) per paper section 3.3.
    """

    def __init__(self, m=2, n=3, num_classes=19, planes=64,
                 ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment

        # I Branch (backbone)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True))

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom))
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom))
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch  (m==2 => PIDNet-S/M uses PAPPM+Light_Bag, else => PIDNet-L uses DAPPM+Bag)
        if m == 2:
            # PIDNet-S/M path (kept for completeness but not used for L)
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom))
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom))
            self.spp = _build_pappm(planes * 16, ppm_planes, planes * 4)
            self.dfm = _build_lightbag(planes * 4, planes * 4)
        else:
            # PIDNet-L path: DAPPM + Bag
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom))
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom))
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction heads
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        # Weight initialisation
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    # ------------------------------------------------------------------
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            no_relu = (i == blocks - 1)
            layers.append(block(inplanes, planes, stride=1, no_relu=no_relu))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        return block(inplanes, planes, stride, downsample, no_relu=True)

    # ------------------------------------------------------------------
    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
            self.diff3(x), size=[height_output, width_output],
            mode='bilinear', align_corners=algc)

        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x), size=[height_output, width_output],
            mode='bilinear', align_corners=algc)

        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


def _build_pappm(inplanes, branch_planes, outplanes):
    """Build PAPPM inline (for S/M variants, not used in L training)."""
    import torch.nn as _nn
    bn_m = 0.1

    class PAPPM(_nn.Module):
        def __init__(self):
            super().__init__()
            self.scale1 = _nn.Sequential(
                _nn.AvgPool2d(5, 2, 2), _nn.BatchNorm2d(inplanes, momentum=bn_m),
                _nn.ReLU(True), _nn.Conv2d(inplanes, branch_planes, 1, bias=False))
            self.scale2 = _nn.Sequential(
                _nn.AvgPool2d(9, 4, 4), _nn.BatchNorm2d(inplanes, momentum=bn_m),
                _nn.ReLU(True), _nn.Conv2d(inplanes, branch_planes, 1, bias=False))
            self.scale3 = _nn.Sequential(
                _nn.AvgPool2d(17, 8, 8), _nn.BatchNorm2d(inplanes, momentum=bn_m),
                _nn.ReLU(True), _nn.Conv2d(inplanes, branch_planes, 1, bias=False))
            self.scale4 = _nn.Sequential(
                _nn.AdaptiveAvgPool2d((1, 1)), _nn.BatchNorm2d(inplanes, momentum=bn_m),
                _nn.ReLU(True), _nn.Conv2d(inplanes, branch_planes, 1, bias=False))
            self.scale0 = _nn.Sequential(
                _nn.BatchNorm2d(inplanes, momentum=bn_m), _nn.ReLU(True),
                _nn.Conv2d(inplanes, branch_planes, 1, bias=False))
            self.scale_process = _nn.Sequential(
                _nn.BatchNorm2d(branch_planes * 4, momentum=bn_m), _nn.ReLU(True),
                _nn.Conv2d(branch_planes * 4, branch_planes * 4, 3, padding=1, groups=4, bias=False))
            self.compression = _nn.Sequential(
                _nn.BatchNorm2d(branch_planes * 5, momentum=bn_m), _nn.ReLU(True),
                _nn.Conv2d(branch_planes * 5, outplanes, 1, bias=False))
            self.shortcut = _nn.Sequential(
                _nn.BatchNorm2d(inplanes, momentum=bn_m), _nn.ReLU(True),
                _nn.Conv2d(inplanes, outplanes, 1, bias=False))

        def forward(self, x):
            h, w = x.shape[-2], x.shape[-1]
            x_ = self.scale0(x)
            scales = [
                F.interpolate(self.scale1(x), [h, w], mode='bilinear', align_corners=False) + x_,
                F.interpolate(self.scale2(x), [h, w], mode='bilinear', align_corners=False) + x_,
                F.interpolate(self.scale3(x), [h, w], mode='bilinear', align_corners=False) + x_,
                F.interpolate(self.scale4(x), [h, w], mode='bilinear', align_corners=False) + x_,
            ]
            scale_out = self.scale_process(torch.cat(scales, 1))
            return self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)

    return PAPPM()


def _build_lightbag(in_channels, out_channels):
    """Build Light_Bag inline (for S/M variants, not used in L training)."""
    import torch.nn as _nn

    class LightBag(_nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_p = _nn.Sequential(
                _nn.Conv2d(in_channels, out_channels, 1, bias=False),
                _nn.BatchNorm2d(out_channels))
            self.conv_i = _nn.Sequential(
                _nn.Conv2d(in_channels, out_channels, 1, bias=False),
                _nn.BatchNorm2d(out_channels))

        def forward(self, p, i, d):
            edge_att = torch.sigmoid(d)
            return self.conv_p((1 - edge_att) * i + p) + self.conv_i(i + edge_att * p)

    return LightBag()


def build_pidnet_l(num_classes=2):
    """Build PIDNet-L with official hyper-parameters from paper section 4.1."""
    return PIDNet(m=3, n=4, num_classes=num_classes, planes=64,
                  ppm_planes=112, head_planes=256, augment=True)


# ==============================================================================
# Boundary GT generation
# ==============================================================================

def generate_boundary_gt(label, kernel_size=7):
    """Generate binary boundary map from segmentation label.

    Uses max-pool dilation trick: a pixel is a boundary if its neighbourhood
    contains more than one class value.

    Args:
        label: LongTensor [B, H, W] with values in {0, 1, ...}
    Returns:
        FloatTensor [B, H, W] with 1.0 at class boundaries, 0.0 elsewhere.
        Mean over driving images should be roughly 0.02-0.15.
    """
    # [B, 1, H, W] float
    lbl_f = label.unsqueeze(1).float()
    pad = kernel_size // 2

    # Max-pool and min-pool in the neighbourhood
    lbl_max = F.max_pool2d(lbl_f, kernel_size=kernel_size, stride=1, padding=pad)
    lbl_min = -F.max_pool2d(-lbl_f, kernel_size=kernel_size, stride=1, padding=pad)

    # Where max != min -> boundary region
    boundary = (lbl_max != lbl_min).squeeze(1).float()   # [B, H, W]
    return boundary


# ==============================================================================
# Loss functions per PIDNet paper equations (5) and (6)
# ==============================================================================

def semantic_loss(logits, target):
    """Standard cross-entropy over full-resolution logits (upsampled to target)."""
    # logits: [B, C, h, w] (1/8 resolution), target: [B, H, W]
    target_h, target_w = target.shape[-2], target.shape[-1]
    logits_up = F.interpolate(logits, size=(target_h, target_w),
                              mode='bilinear', align_corners=algc)
    return F.cross_entropy(logits_up, target)


def boundary_loss(boundary_logits, boundary_gt, pos_weight_factor=10.0):
    """Boundary BCE loss with pos_weight to handle class imbalance.

    Boundary pixels are rare (~5-10% of all pixels), so we up-weight them.

    Args:
        boundary_logits: [B, 1, h, w] raw logits from D-branch seghead
        boundary_gt    : [B, H, W] float32 with 1 at boundaries
        pos_weight_factor: scalar weight for positive (boundary) class
    """
    target_h, target_w = boundary_gt.shape[-2], boundary_gt.shape[-1]
    logits_up = F.interpolate(boundary_logits, size=(target_h, target_w),
                              mode='bilinear', align_corners=algc)
    # [B, 1, H, W] -> [B, H, W]
    logits_up = logits_up.squeeze(1)
    pos_weight = torch.tensor([pos_weight_factor], device=boundary_logits.device)
    return F.binary_cross_entropy_with_logits(
        logits_up, boundary_gt, pos_weight=pos_weight)


def bas_loss(main_logits, target, boundary_gt, threshold=0.8):
    """Boundary-Aware Semantic loss (BAS, paper eq. 5).

    Selects hard semantic pixels near boundaries for extra CE supervision.
    A pixel is selected if its boundary probability > threshold.

    Args:
        main_logits : [B, C, h, w]   main output logits (1/8 res)
        target      : [B, H, W]      long labels (full res)
        boundary_gt : [B, H, W]      float boundary map (full res)
        threshold   : float          boundary confidence threshold
    """
    # Upsample logits to full resolution
    target_h, target_w = target.shape[-2], target.shape[-1]
    logits_up = F.interpolate(main_logits, size=(target_h, target_w),
                              mode='bilinear', align_corners=algc)  # [B, C, H, W]

    # Mask: pixels near boundaries whose boundary confidence exceeds threshold
    mask = (boundary_gt > threshold)   # [B, H, W]

    # Flatten
    _, C, _, _ = logits_up.shape
    logits_flat = logits_up.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    target_flat = target.reshape(-1)                              # [B*H*W]
    mask_flat = mask.reshape(-1)                                  # [B*H*W]

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=main_logits.device, requires_grad=True)

    # Apply CE only on boundary/hard pixels
    hard_logits = logits_flat[mask_flat]
    hard_target = target_flat[mask_flat]
    return F.cross_entropy(hard_logits, hard_target)


# ==============================================================================
# Poly LR with linear warmup
# ==============================================================================

def poly_lr_with_warmup(optimizer, cur_iter, max_iter, base_lr,
                        warmup_iters, warmup_start_lr=1e-4, power=0.9):
    """Update learning rate: linear warmup then poly decay."""
    if cur_iter < warmup_iters:
        # Linear warmup
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * cur_iter / warmup_iters
    else:
        lr = base_lr * (1.0 - (cur_iter - warmup_iters) /
                        (max_iter - warmup_iters + 1e-8)) ** power
    lr = max(lr, 1e-6)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ==============================================================================
# Training loop
# ==============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else 'CPU')
    print(f"Device: {device} - {device_name}")

    # Paths
    image_train_dir = os.path.join(
        PROJECT_ROOT, "data", "gt_4class",
        "leftImg8bit_trainvaltest", "leftImg8bit", "train")
    label_train_dir = os.path.join(PROJECT_ROOT, "data", "gt_4class", "train")
    image_val_dir = os.path.join(
        PROJECT_ROOT, "data", "gt_4class",
        "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    label_val_dir = os.path.join(PROJECT_ROOT, "data", "gt_4class", "val")

    model_dir   = os.path.join(PROJECT_ROOT, "models")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Parse image size
    h_str, w_str = args.image_size.split(',')
    image_size = (int(h_str), int(w_str))

    # Augmentation (stronger than baseline for from-scratch training)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        CompressionArtifactAugmentation(p=0.7, min_quality=35, max_quality=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CityscapesROI(
        image_train_dir, label_train_dir, train_transform, image_size=image_size)
    val_ds = CityscapesROI(
        image_val_dir, label_val_dir, val_transform, image_size=image_size)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Train dataset is empty. Run prepare_2class_labels.py to create 2-class labels.")
    if len(val_ds) == 0:
        raise RuntimeError(
            "Val dataset is empty. Check data at data/gt_4class/val.")

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # Build model
    model = build_pidnet_l(num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PIDNet-L parameters: {n_params / 1e6:.2f}M")

    # Quick forward shape check (use batch=2 + eval mode to avoid BN issues)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, image_size[0], image_size[1]).to(device)
        outs = model(dummy)
        print(f"Forward output shapes: "
              f"x_extra_p={list(outs[0].shape)}, "
              f"x_={list(outs[1].shape)}, "
              f"x_extra_d={list(outs[2].shape)}")
    model.train()

    # Optimizer: SGD as per paper section 4.2
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
    )

    # AMP scaler
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # LR schedule parameters
    warmup_epochs = 5
    total_iters = args.epochs * len(train_loader)
    warmup_iters = warmup_epochs * len(train_loader)

    # CSV logging
    metrics_csv = os.path.join(metrics_dir, "train_metrics_pidnet_l.csv")
    best_result_json = os.path.join(metrics_dir, "best_result_pidnet_l.json")
    checkpoint_path = os.path.join(model_dir, "best_pidnet_l.pth")

    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'mean_iou', 'best_miou_so_far', 'is_best',
            'loss_l0', 'loss_l1', 'loss_l2', 'loss_l3', 'total_loss',
        ])

    best_miou = 0.0
    best_epoch = -1
    global_iter = 0

    for epoch in range(args.epochs):
        model.train()

        epoch_l0 = epoch_l1 = epoch_l2 = epoch_l3 = epoch_total = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)           # [B, H, W] long

            # Generate boundary GT from label (on CPU-safe side, then move)
            boundary_gt = generate_boundary_gt(labels, kernel_size=7)  # [B, H, W] float

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Forward: returns [x_extra_p, x_, x_extra_d]
                outputs = model(images)
                x_extra_p, x_, x_extra_d = outputs

                # l0: CE on P-branch auxiliary output (weight lambda0=0.4)
                l0 = semantic_loss(x_extra_p, labels)

                # l1: Boundary BCE on D-branch output (weight lambda1=20)
                # pos_weight handles ~5-10% boundary pixel imbalance
                boundary_ratio = boundary_gt.mean().item()
                # dynamic pos_weight: more imbalanced -> higher weight
                pw = max(1.0, (1.0 - boundary_ratio) / (boundary_ratio + 1e-6))
                pw = min(pw, 20.0)
                l1 = boundary_loss(x_extra_d, boundary_gt, pos_weight_factor=pw)

                # l2: CE on main output (weight lambda2=1)
                l2 = semantic_loss(x_, labels)

                # l3: BAS-loss (boundary-aware CE) eq.(5), lambda3=1, t=0.8
                l3 = bas_loss(x_, labels, boundary_gt, threshold=0.8)

                # Combined loss per paper eq.(6)
                loss = 0.4 * l0 + 20.0 * l1 + 1.0 * l2 + 1.0 * l3

            if not torch.isfinite(loss):
                print(f"  [WARNING] Non-finite loss at iter {global_iter}, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                global_iter += 1
                continue

            scaler.scale(loss).backward()
            # Gradient clipping to prevent NaN from scratch training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            # Update LR
            cur_lr = poly_lr_with_warmup(
                optimizer, global_iter, total_iters, args.lr, warmup_iters)
            global_iter += 1

            l0_v = l0.item()
            l1_v = l1.item()
            l2_v = l2.item()
            l3_v = l3.item()
            tot_v = loss.item()

            epoch_l0 += l0_v
            epoch_l1 += l1_v
            epoch_l2 += l2_v
            epoch_l3 += l3_v
            epoch_total += tot_v
            n_batches += 1

            pbar.set_postfix(
                l0=f"{l0_v:.3f}",
                l1=f"{l1_v:.3f}",
                l2=f"{l2_v:.3f}",
                l3=f"{l3_v:.3f}",
                tot=f"{tot_v:.3f}",
                lr=f"{cur_lr:.5f}",
                bnd=f"{boundary_ratio:.3f}",
            )

        if n_batches == 0:
            print(f"  [WARNING] No valid batches in epoch {epoch + 1}.")
            continue

        avg_l0 = epoch_l0 / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_l2 = epoch_l2 / n_batches
        avg_l3 = epoch_l3 / n_batches
        avg_tot = epoch_total / n_batches

        print(f"\n--- Epoch {epoch + 1} train summary ---")
        print(f"  l0 (P-CE  x0.4) : {avg_l0:.4f}")
        print(f"  l1 (D-BCE x20 ) : {avg_l1:.4f}")
        print(f"  l2 (main CE x1) : {avg_l2:.4f}")
        print(f"  l3 (BAS    x1 ) : {avg_l3:.4f}")
        print(f"  total loss      : {avg_tot:.4f}")

        # ----------------------------------------------------------------
        # Validation: mIoU over 2 classes
        # ----------------------------------------------------------------
        model.eval()
        ious_all = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    # During eval use main branch output x_
                    x_extra_p, x_, x_extra_d = outputs
                    # Upsample to label resolution
                    x_up = F.interpolate(
                        x_, size=(labels.shape[-2], labels.shape[-1]),
                        mode='bilinear', align_corners=algc)
                pred = torch.argmax(x_up, dim=1)  # [B, H, W]
                for p, t in zip(pred, labels):
                    iou = []
                    for c in range(2):
                        inter = ((p == c) & (t == c)).sum().item()
                        union = ((p == c) | (t == c)).sum().item()
                        iou.append(inter / union if union > 0 else 0.0)
                    ious_all.append(float(np.mean(iou)))

        mean_iou = float(np.mean(ious_all)) if ious_all else 0.0
        is_best = mean_iou > best_miou

        print(f"  Val mIoU: {mean_iou:.4f}  |  Best so far: {max(best_miou, mean_iou):.4f}")

        # ----------------------------------------------------------------
        # Append CSV row
        # ----------------------------------------------------------------
        with open(metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(mean_iou),
                float(max(best_miou, mean_iou)),
                int(is_best),
                avg_l0, avg_l1, avg_l2, avg_l3, avg_tot,
            ])

        # ----------------------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------------------
        if is_best:
            best_miou = mean_iou
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': {
                    'model_name': 'pidnet_l',
                    'num_classes': 2,
                    'best_miou': float(best_miou),
                    'best_epoch': int(best_epoch),
                },
            }, checkpoint_path)

            with open(best_result_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name': 'pidnet_l',
                    'best_miou': float(best_miou),
                    'best_epoch': int(best_epoch),
                    'num_epochs': int(args.epochs),
                    'checkpoint_path': checkpoint_path,
                }, f, ensure_ascii=False, indent=2)

            print(f"  [SAVED] New best checkpoint: mIoU = {best_miou:.4f} "
                  f"at epoch {best_epoch} -> {checkpoint_path}")

    print(f"\nTraining complete. Best mIoU = {best_miou:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics CSV: {metrics_csv}")
    if best_epoch > 0:
        print(f"Best result JSON: {best_result_json}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train official PIDNet-L (CVPR 2023) for 2-class ROI segmentation")
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Base learning rate for SGD (default: 1e-2)')
    parser.add_argument('--image_size', type=str, default='512,1024',
                        help='Input image size HxW as "H,W" (default: 512,1024)')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train(args)
