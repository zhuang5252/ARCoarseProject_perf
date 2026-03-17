# -*- coding: utf-8 -*-
# @Author    : 
# @File      : binary_resnet.py
# @Created   : 2025/11/28 上午10:56
# @Desc      : 二分类ResNet网络定义(binary_resnet网络定义)

"""
二分类ResNet网络定义(binary_resnet网络定义)
"""

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BinaryResNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super().__init__()

        # 模型第一层卷积输出通道数
        self.inplanes = 32  # 减少初始通道数

        # 首层卷积调整（stride=1保留更多空间信息）
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 仅保留一个基础块层
        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2, stride=2)

        # 自适应池化+二分类输出
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Linear(64 * BasicBlock.expansion, 1),
        #     nn.Sigmoid()  # 输出0-1的概率值
        # )
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def binary_resnet(cfg, pretrained=False):
    model_name = cfg['base_model_name']  # 假设cfg是一个字典，并且包含'model_name'键
    num_classes = cfg['nb_classes']
    input_channel = cfg['input_channel']
    if model_name == 'BinaryResNet':
        return BinaryResNet(input_channel, num_classes)
    else:
        assert f'Error: This model_name {model_name} is currently not supported!'


# 使用示例
if __name__ == "__main__":
    model = BinaryResNet(input_channel=3)
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    print(output.shape)  # 应输出 torch.Size([4])，即4个样本的二分类概率
