#!/usr/bin/env python3
# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import cv2

# 4顶点描述的矩形框转描述抓取框的向量
def bboxes2grasps(x, y):
    # 0,2是一组对角线顶点
    xc = (x[0] + x[2]) / 2
    yc = (y[0] + y[2]) / 2
    # 0,3是一组边顶点
    detaX = x[3] - x[0]
    detaY = y[3] - y[0]
    tan_theta = detaY / (detaX if abs(detaX) > 0.01 else 0.01)
    # tan(85度) = 11 限制正切值过大
    # tan_theta = min(max(-11, tan_theta), 11)
    # 宽高 
    width = (detaX**2 + detaY**2)**0.5
    height = ((x[1] - x[0])**2 + (y[1] - y[0])**2)**0.5

    return [xc, yc, tan_theta, width, height]

# 抓取框向量转矩形框
def grasps2bboxes(labels):
    x, y, tan_theta, w, h = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4] 
    theta = torch.atan(tan_theta)
    x0, y0 = x-w/2*torch.cos(theta) + h/2*torch.sin(theta), y - w/2*torch.sin(theta) - h/2*torch.cos(theta) # 数据集是按逆时针顺序给出4个顶点的，这里也跟随逆时针序
    x1, y1 = x-w/2*torch.cos(theta) - h/2*torch.sin(theta), y - w/2*torch.sin(theta) + h/2*torch.cos(theta) 
    x2, y2 = x+w/2*torch.cos(theta) - h/2*torch.sin(theta), y + w/2*torch.sin(theta) + h/2*torch.cos(theta)
    x3, y3 = x+w/2*torch.cos(theta) + h/2*torch.sin(theta), y + w/2*torch.sin(theta) - h/2*torch.cos(theta)
    batch_size = len(labels)
    point0 = torch.cat([x0.reshape(batch_size, 1), y0.reshape(batch_size, 1)], dim=1).cpu().numpy().astype(np.int32)
    point1 = torch.cat([x1.reshape(batch_size, 1), y1.reshape(batch_size, 1)], dim=1).cpu().numpy().astype(np.int32)
    point2 = torch.cat([x2.reshape(batch_size, 1), y2.reshape(batch_size, 1)], dim=1).cpu().numpy().astype(np.int32)
    point3 = torch.cat([x3.reshape(batch_size, 1), y3.reshape(batch_size, 1)], dim=1).cpu().numpy().astype(np.int32)

    return point0, point1, point2, point3

# 绘制预测结果 root/out_dir不带文件分隔符
def drawBBox(root, imgIds, outputs, out_dir): 
    batch_size = len(outputs)
    point0, point1, point2, point3 = grasps2bboxes(outputs)
    for i in range(batch_size):
        img = cv2.imread(f"{root}/{imgIds[i]}r.png")
        img = cv2.resize(img, (224, 224))
        cv2.line(img, tuple(point0[i]), tuple(point1[i]), (0, 255, 0), 2)
        cv2.line(img, tuple(point0[i]), tuple(point3[i]), (0, 255, 0), 2)
        cv2.line(img, tuple(point2[i]), tuple(point1[i]), (0, 255, 0), 2)
        cv2.line(img, tuple(point2[i]), tuple(point3[i]), (0, 255, 0), 2)
        cv2.imwrite(f"{out_dir}/{imgIds[i]}_pred.png", img)

# 绘制水平矩形框 root/out_dir不带文件分隔符 edges 按 left top right bottom顺序存储
def drawRect(root, imgIds, edges, out_dir, color): 
    batch_size = len(imgIds)
    for i in range(batch_size):
        img = cv2.imread(f"{root}/{imgIds[i]}r.png")
        img = cv2.resize(img, (224, 224))
        for j in range(len(edges)): 
            # 第j个框的第0个点的第i个图片
            cv2.line(img, (edges[j][0][i], edges[j][1][i]), (edges[j][2][i], edges[j][1][i]), color[j], 2)
            cv2.line(img, (edges[j][0][i], edges[j][1][i]), (edges[j][0][i], edges[j][3][i]), color[j], 2)
            cv2.line(img, (edges[j][2][i], edges[j][3][i]), (edges[j][2][i], edges[j][1][i]), color[j], 2)
            cv2.line(img, (edges[j][2][i], edges[j][3][i]), (edges[j][0][i], edges[j][3][i]), color[j], 2)

        cv2.imwrite(f"{out_dir}/{imgIds[i]}_rect.png", img)

