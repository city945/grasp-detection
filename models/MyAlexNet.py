#!/usr/bin/env python3
# coding:utf8

import torch
from torch import nn
from torchvision import models
import numpy as np

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.features = models.alexnet(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 512),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        
        return out

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = MyAlexNet()
    outputs = model(inputs)
    print(outputs)
    