#!/usr/bin/env python3
# coding:utf8

import torch
from torch import nn
from torchvision import models
import numpy as np

class MyVgg16(nn.Module):
    def __init__(self):
        super(MyVgg16, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7*7*512, 1024),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(1024, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        
        return out

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = MyVgg16()
    outputs = model(inputs)
    print(outputs)
    