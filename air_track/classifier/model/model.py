# -*- coding: utf-8 -*-
# @Author    : 
# @File      : model.py
# @Created   : 2025/11/28 上午10:57
# @Desc      : 二级分类器模型框架，支持多种backbone和其他部分的组合

"""
二级分类器模型框架，支持多种backbone和其他部分的组合
"""

import torch
from torch import nn
from typing import Dict
from air_track.utils import auto_import_module


class Model(nn.Module):
    """可配置的深度学习模型框架，支持多种backbone和head组合

    Args:
        cfg (Dict): 配置字典，包含模型结构参数
        pretrained (bool): 是否加载预训练权重
    """
    def __init__(self, cfg: Dict, pretrained: bool = False):
        super().__init__()
        self.cfg = cfg
        self._validate_config()
        self.pretrained = pretrained
        self.model_mode = self.cfg['model_mode']

        # 模型组件初始化
        self.base_model = self._build_backbone()

    def _validate_config(self) -> None:
        """验证配置参数有效性"""
        required_keys = ['model_mode', 'model_cls', 'base_model_name', 'input_channel', 'nb_classes']
        for key in required_keys:
            if key not in self.cfg:
                raise ValueError(f"Missing required config key: {key}")

    def _build_backbone(self) -> nn.Module:
        """构建模型backbone部分

        Returns:
            nn.Module: 初始化好的backbone模型
        """
        try:
            backbone_module = auto_import_module(
                f'air_track.classifier.model.backbones.{self.cfg["model_cls"]}',
                self.cfg["model_cls"])
            base_model = backbone_module(cfg=self.cfg, pretrained=self.pretrained)
            # TODO 后续视情况改为注册器形式

            return base_model
        except Exception as e:
            raise RuntimeError(f"Failed to build backbone: {str(e)}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            inputs (torch.Tensor): 输入张量

        Returns:
            返回output
        """
        output = self.base_model(inputs)

        # 如果是预测模式，则使用softmax输出概率分布
        if self.model_mode == 'predict':
            output = torch.softmax(output, dim=1)

        return output
