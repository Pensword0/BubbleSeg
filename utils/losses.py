import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    # 功能：将标签转换为 one-hot 编码格式。
    # 原理：对于每个像素点，标签转换为一个长度等于类别数的向量，其中标签对应的类别位置为1，其余位置为0。
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    # 功能：根据标签数据计算类别权重。
    # 原理：计算每个类别的出现次数，然后使用一定的规则（如中位数除以各类计数）来得出权重，旨在处理类别不平衡问题。
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    # 功能：实现二维交叉熵损失。
    # 原理：计算模型输出和真实标签之间的交叉熵，通常用于多分类问题。
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        #print('utils_losses_36')
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    # 功能：实现 Dice 损失。
    # 原理：基于 Dice 系数，这是一种衡量两个样本的重叠程度的指标。Dice 损失是 1 减去 Dice 系数，常用于图像分割任务，特别是在类别不平衡的情况下。
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    # 功能：实现 Focal 损失。
    # 原理：修改版的交叉熵损失，通过减少易分类样本的权重来解决类别不平衡问题。它通过增加一个调节因子来减少易分类样本对损失的贡献。
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    # 组合交叉熵损失和 Dice 损失。
    # 原理：这个组合损失函数同时考虑了像素级别的分类准确性（交叉熵损失）和预测区域与真实区域的重叠程度（Dice 损失）。
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class LovaszSoftmax(nn.Module):
    # 功能：实现 Lovasz-Softmax 损失。
    # 原理：这是一种基于梯度的损失函数，用于优化模型的排序错误。特别适用于不平衡数据集和图像分割任务，因为它直接针对 Jaccard 指数优化。
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
