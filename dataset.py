# coding=utf-8

from torch.utils import data
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import random 
import torch
from utils.utils import bboxes2grasps

class CornellDataset(data.Dataset): 
    """
    数据集名，数据集实现、预处理
    """
    def __init__(self, root, transform=None, train=True, test=False, input_size=(224, 224)): # 类似标准数据集中 train=True标识 train=T/F标识训练验证，test测试
        # 获取所有图片的地址，并根据训练，验证，测试划分数据
        self.root = root
        self.test = test
        imgIds = [os.path.basename(filename)[:-5] for filename in os.listdir(root) if 'png' in filename] # 数据集按train/cat.0.jpg编排

        if self.test:
            self.imgIds = imgIds # 测试集的所有图片测试
        elif train:
            self.imgIds = imgIds[:int(0.85 * len(imgIds))] # 训练集前%训练
        else:
            self.imgIds = imgIds[int(0.85 * len(imgIds)):] # 后%验证

        if transform is None: # 定义默认变形
            if self.test or not train: # 验证集x 或 测试集
                self.transform = transforms.Compose([
                    transforms.Resize(input_size),
                    # transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(input_size),
                    # transforms.CenterCrop(input_size),  # 将图像随机裁剪为不同大小(默认0.08~1.0和宽高比(默认3/4~4/3)，224是期望输出的图像大小
                    # transforms.RandomHorizontalFlip(),  # 以一定几率(默认为0.5)水平翻转图像
                    # transforms.RandomRotation(np.random.randint(0,360)),
                    transforms.ToTensor(),  # 将图像数据或数组数据转换为tensor数据类型
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index): # 实现__getitem实现索引、切片操作
        # 一次返回一张图片的数据
        imgId = self.imgIds[index]
        img_path = os.path.join(self.root, imgId + 'r.png')
        label_path = os.path.join(self.root, imgId + 'cpos.txt')
        data = Image.open(img_path)
        data = self.transform(data)

        # 标签处理
        with open(label_path, 'r') as f:
            content = f.readlines()
            # 数据集作妖，点坐标含nan
            flag = True # 标志位，是否继续查找可用数据
            while flag:
                flag = False
                # start = 0 # 仅取第一个框容易过拟合，效果更差
                start = random.randrange(0, len(content), 4) # 一个txt里有多个框，随机选取一个做标签
                x, y = [], []
                for point in content[start:start+4]:
                    point = point.strip('\n').split(' ')
                    if 'NaN' in point: 
                        flag = True
                        break
                    x.append(float(point[0]) * 224 / 640)  # 转成224*224
                    y.append(float(point[1]) * 224 / 480)
                

        label = torch.tensor(bboxes2grasps(x, y))
            
        return data, label, imgId # 一般只有两参数，实验发现DataLoader不管几个返回值，只要那边有相同个数的变量接收

    def __len__(self): # Dataset子类必须，len用于DataLoader
        return len(self.imgIds)
