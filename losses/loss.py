# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            loss = 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            loss = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                loss += 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
            loss /= self.num_classes
        return loss
