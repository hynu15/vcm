"""BCE-Dice loss + DSN aux wrapper (paper §III-B Eq.9-12).

- Multi-class: BCE thay bằng CrossEntropy (mặc nhiên trên paper khi 4 lớp).
- Dice loss với smooth=1.
- Aux head: loss = L_main + 0.4 × L_aux (khớp CCNet criterion gốc).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,H,W) at full resolution; target: (B,H,W) long
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index).unsqueeze(1).float()
        tgt = target.clone()
        tgt[target == self.ignore_index] = 0
        oh = F.one_hot(tgt, self.num_classes).permute(0, 3, 1, 2).float() * valid
        inter = (probs * oh).sum(dim=(2, 3))
        union = (probs * valid).sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return ce_loss + dice.mean()


class BCEDiceWithAux(nn.Module):
    """Tính loss trên cả main và aux head; tự upsample logits về kích thước nhãn."""

    def __init__(self, num_classes: int = 4, aux_weight: float = 0.4, ignore_index: int = 255):
        super().__init__()
        self.main = BCEDiceLoss(num_classes, ignore_index=ignore_index)
        self.aux = BCEDiceLoss(num_classes, ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, preds, target: torch.Tensor) -> torch.Tensor:
        # preds = [main, aux], mỗi cái shape (B,C,h,w) với h,w < H,W (output stride 8)
        H, W = target.shape[1], target.shape[2]
        main_up = F.interpolate(preds[0], size=(H, W), mode="bilinear", align_corners=True)
        aux_up = F.interpolate(preds[1], size=(H, W), mode="bilinear", align_corners=True)
        return self.main(main_up, target) + self.aux_weight * self.aux(aux_up, target)
