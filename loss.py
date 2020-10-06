#!/usr/bin/env python3
# coding:utf8
from numpy.testing._private.utils import import_nose
import torch
import torch.nn as nn 
from utils.utils import *
import numpy as np
import ipdb

# 自定义均方误差损失函数
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, labels):
        # tan(85度) = 11 限制正切值过大
        outputs[:,2] = torch.tanh(outputs[:,2]) * 11
        labels[:, 2] = torch.tanh(labels[:, 2]) * 11
        outputs[:,2] = outputs[:,2] * 10**0.5     # 角度损失项乘以10，增大其数量级，也相当于加权
        labels[:, 2] = labels[:, 2] * 10**0.5

        return torch.sum(((outputs - labels))**2 , dim=(0,1)) / len(outputs) 

# 自定义矩形框评价指标 
def RectAngleMetric(outputs, labels, imgIds=None, root=None):
    # 这里不需要bp，可以直接clamp截断
    outputs[:,2] = torch.clamp(outputs[:,2], -11, 11)
    labels[:, 2] = torch.clamp(labels[:, 2], -11, 11)
    # 抓取框合格标准1: 抓取角度30度内
    res1 = (torch.abs(torch.atan(outputs[:, 2]) - torch.atan(labels[:, 2])) < 30).cpu().numpy()

    # 抓取框合格标准2: IoU > 25%
    p0_hat, p1_hat, p2_hat, p3_hat = grasps2bboxes(outputs)
    p0, p1, p2, p3 = grasps2bboxes(labels)  # p0_hat与p0并不全是一一对应的，因为代码输出保证p0在左上角，而数据集给的0号点不一定在左上角，但中心点是一致的,都是逆时针序
    # 这里矩形框是斜的，训练时角度也不可知，IoU不好算，用外接矩形近似得了
    batch_size = len(labels)
    x = [p0[:,0], p1[:,0], p2[:,0], p3[:,0]]
    y = [p0[:,1], p1[:,1], p2[:,1], p3[:,1]]
    right, left = np.max(x, axis=0), np.min(x, axis=0)
    bottom, top = np.max(y, axis=0), np.min(y, axis=0)
    x_hat = [p0_hat[:,0], p1_hat[:,0], p2_hat[:,0], p3_hat[:,0]]
    y_hat = [p0_hat[:,1], p1_hat[:,1], p2_hat[:,1], p3_hat[:,1]]
    right_hat, left_hat = np.max(x_hat, axis=0), np.min(x_hat, axis=0)
    bottom_hat, top_hat = np.max(y_hat, axis=0), np.min(y_hat, axis=0)
    # 求交集区域边界 
    left_cross = np.max([left, left_hat], axis=0)
    right_cross = np.min([right, right_hat], axis=0)
    top_cross = np.max([top, top_hat], axis=0)
    bottom_cross = np.min([bottom, bottom_hat], axis=0)

    # test: 可视化
    if root is not None:
        rect        = [left, top, right, bottom]
        rect_hat    = [left_hat, top_hat, right_hat, bottom_hat]
        rect_cross  = [left_cross, top_cross, right_cross, bottom_cross]
        drawRect(root, imgIds, [rect, rect_hat, rect_cross], "./out/rect", [(255,0,0), (0,255,0),(0,0,255)])
        # ipdb.set_trace()

    # 不存在交集区域的样本直接置False
    res2_mask = np.logical_not(np.logical_or(left_cross >= right_cross, top_cross >= bottom_cross))
    # 计算iou
    area = np.abs((right-left)*(bottom-top))
    area_hat = np.abs((right_hat-left_hat)*(bottom_hat-top_hat))
    area_cross = np.abs((right_cross-left_cross)*(bottom_cross-top_cross))
    iou = area_cross / (area + area_hat - area_cross)
    res2 = np.logical_and(iou > 0.25, res2_mask)

    return np.logical_and(res1, res2)

