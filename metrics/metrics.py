# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot(a):
    plt.figure()
    plt.imshow(a)


class DiceScore(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            dice = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice += (2. * intersection + self.smooth) / (union + self.smooth)
            dice /= self.num_classes
        return dice

class IoU(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            den = union - intersection
            iou = (intersection + self.smooth) / (den + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            iou = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                den = union - intersection
                iou += (intersection + self.smooth) / (den + self.smooth)
            iou /= self.num_classes
        return iou
