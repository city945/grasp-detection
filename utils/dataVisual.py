#!/usr/bin/env python3
# coding:utf8

import numpy as np
import random
import cv2
import torch 
from utils import *     # 测试utils

# 含nan: 0132 
# 把框画到图上看看效果
img = cv2.imread('../data/train/pcd0132r.png')
img = cv2.resize(img, (224, 224))

with open('../data/train/pcd0132cpos.txt', 'r') as f:
    content = f.readlines()
    start = random.randrange(0, len(content), 4)
    x, y = [], []
    for point in content[start:start+4]:
        point = point.strip('\n').split(' ')
        x.append(float(point[0]) * 224 / 640)
        y.append(float(point[1]) * 224 / 480)

print(x, y)
label = bboxes2grasps(x, y)
print(label)
# 标签文件中描述的原始抓取框
x = np.array(x, np.int32)
y = np.array(y, np.int32)
cv2.line(img, (x[2], y[2]), (x[1], y[1]), (0, 0, 255), 2)
cv2.line(img, (x[2], y[2]), (x[3], y[3]), (0, 0, 255), 2)   
cv2.line(img, (x[0], y[0]), (x[1], y[1]), (0, 0, 255), 2)   
cv2.line(img, (x[0], y[0]), (x[3], y[3]), (0, 0, 255), 2)
cv2.circle(img, (x[0], y[0]), 2, (0, 0, 255))
cv2.circle(img, (x[1], y[1]), 2, (0, 255, 0))
cv2.circle(img, (x[2], y[2]), 2, (255, 0, 0))
cv2.circle(img, (x[3], y[3]), 2, (0, 0, 0))
# 0,2是一组对角线顶点
xc, yc = int(label[0]), int(label[1])
tan_theta = label[3]
cv2.circle(img, (xc, yc), 2, (0, 0, 255))
# 0,1是一组边顶点
cv2.line(img, (xc, yc), (xc + 5, yc + int(5*tan_theta)), (255, 0, 0), 2)

# 标签转抓取框 会有一定的误差
label = np.expand_dims(np.array(label), axis=0)
labels = np.concatenate((label, label), axis=0) # 升个Batch维 BCHW B=2
point0, point1, point2, point3 = grasps2bboxes(torch.tensor(labels))

cv2.line(img, tuple(point0[0]), tuple(point1[0]), (0, 255, 0), 2)
cv2.line(img, tuple(point0[0]), tuple(point3[0]), (0, 255, 0), 2)
cv2.line(img, tuple(point2[0]), tuple(point1[0]), (0, 255, 0), 2)
cv2.line(img, tuple(point2[0]), tuple(point3[0]), (0, 255, 0), 2)
# cv2.circle(img, tuple(point0[0]), 2, (0, 0, 255))
# cv2.circle(img, tuple(point1[0]), 2, (0, 255, 0))
# cv2.circle(img, tuple(point2[0]), 2, (255, 0, 0))
# cv2.circle(img, tuple(point3[0]), 2, (0, 0, 0))

cv2.imwrite('../out/bbox.png', img)
