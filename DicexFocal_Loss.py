# 定义损失函数
import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()

    def forward(self,output,target,smooth=1e-6):
        intersection = output * target
        dice = (2 * intersection.sum() + smooth) / (output.sum() + target.sum() + smooth)
        dice_loss = 1-dice
        # print(f'dice_loss:{dice_loss}')

        return dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        N = outputs.size()[0]
        p = outputs
        # 计算 p_t
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        # 计算 Focal Loss
        focal_loss = - alpha * ((1 - p_t) ** self.gamma) * torch.log(p_t + 1e-6)  # 添加小常数避免 log(0)

        # 根据需要进行 reduction
        if self.reduction == 'mean':
            loss = focal_loss.mean()
            # print(f'focal_loss:{loss}')
            return loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceFocal_Loss(nn.Module):
    def __init__(self):
        super(DiceFocal_Loss,self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self,output,target):
        loss = self.dice_loss(output,target) + self.focal_loss(output,target)
        return loss
