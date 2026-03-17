import os
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from air_track.detector.model import heads
from typing import Dict, Optional, Tuple, Union, List
from air_track.utils.registry import BACKBONES, HEADS
from air_track.utils import auto_import_module, combine_load_cfg_yaml


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
        self.current_unfreeze_stage = 0  # 当前解冻阶段(渐进式解冻用)

        # 模型组件初始化
        self.base_model = self._build_backbone()
        self.fc_comb = self._build_combine()
        self.head = self._build_head()

        # 微调模式处理
        if cfg.get('finetune', False):
            self.freeze_config = self.cfg.get('freeze_config', {})
            self.constant_unfreeze_layers = self.freeze_config.get('constant_unfreeze_layers', 'head')
            self._init_finetune_mode()

            # 初始冻结设置
            self._init_freeze_strategy()

    def _validate_config(self) -> None:
        """验证配置参数有效性"""
        required_keys = ['model_mode', 'backbone_type', 'backbone_name', 'base_model_name',
                         'head_name', 'nb_classes', 'return_feature_map']
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
                f'air_track.detector.model.backbones.{self.cfg["backbone_type"]}',
                self.cfg["backbone_name"])
            base_model = backbone_module(cfg=self.cfg, pretrained=self.pretrained)
            # TODO 后续视情况改为注册器形式

            return base_model
        except Exception as e:
            raise RuntimeError(f"Failed to build backbone: {str(e)}")

    def _build_combine(self) -> Optional[nn.Module]:
        """构建特征融合层

        Returns:
            Optional[nn.Module]: 特征融合层，如果不需要则为None
        """
        base_model_output_channels = self.base_model.output_channels
        self.combine_outputs_dim = self.cfg.get('combine_outputs_dim', -1)

        if self.combine_outputs_dim > 0:
            self.output_channels = self.combine_outputs_dim
            combine_outputs_kernel = 1
            fc_comb = nn.Conv2d(base_model_output_channels, self.combine_outputs_dim,
                                kernel_size=combine_outputs_kernel)
            return fc_comb
        else:
            self.output_channels = base_model_output_channels
            return None

    def _build_head(self) -> nn.Module:
        """构建模型输出头

        Returns:
            nn.Module: 初始化好的head模块
        """
        try:
            head = heads.__dict__[self.cfg["head_name"]](
                in_ch=self.output_channels,
                cls_num=self.cfg["nb_classes"],
                mode=self.cfg["model_mode"]
            )
            # 注册器形式 TODO 后续视情况改为注册器形式
            # head = HEADS[self.cfg["head_name"]](
            #     in_ch=self.output_channels,
            #     cls_num=self.cfg["nb_classes"],
            #     mode=self.cfg["model_mode"]
            # )

            return head
        except KeyError:
            raise ValueError(f"Invalid head name: {self.cfg['head_name']}")

    def _init_finetune_mode(self) -> None:
        """初始化微调模式，加载预训练权重并冻结指定层"""
        pretrained_path = self.freeze_config.get('finetune_model_path')
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Finetune model path not found: {pretrained_path}")
        else:
            print(f"Finetune model path is: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = self._filter_state_dict(checkpoint['model_state_dict'])

        # 非严格模式加载，允许缺失部分权重
        self.load_state_dict(state_dict, strict=False)

    # TODO 是否要过滤掉head的权重
    def _filter_state_dict(self, state_dict: Dict) -> OrderedDict:
        """过滤预训练权重，跳过不匹配的head层"""
        filtered = OrderedDict()
        for k, v in state_dict.items():
            if 'head' not in k:
                filtered[k] = v

        return filtered

    def _get_all_layers(self):
        """获取所有可训练层的名称，不包含constant_unfreeze_layers涉及层"""
        all_layers = []
        for name, param in self.named_parameters():
            skip_flag = False
            for unfreeze_layer in self.constant_unfreeze_layers:
                if unfreeze_layer in name:  # unfreeze_layer层始终不冻结
                    skip_flag = True
                    break

            if not skip_flag:
                all_layers.append(name)

        return all_layers

    def _get_layer_groups(self) -> Dict[int, List[str]]:
        """自适应层分组方法

        策略:
        1. 通过智能模块识别确保相关层被分到同一大组
        2. 保持对任意网络结构的适应性
        3. 正确处理conv和bn等关联层
        4. 最终输出按从底层到顶层的顺序（Bottom-Up）
        """
        all_layers = self._get_all_layers()

        # 1. 改进的模块识别逻辑
        major_groups = OrderedDict()
        current_major = None
        current_module_key = None

        for layer in all_layers:
            parts = layer.split('.')
            # TODO 目前是根据parts的长度来进行组合，现在official_resnet上进行了验证，其他网络还需进一步验证并调整
            # 智能识别模块关键部分
            if len(parts) >= 5:
                # 对于conv/bn等层，取到父模块级别
                # for i, item in enumerate(parts):
                #     if item in ['conv', 'bn']:
                #         module_key = '.'.join(parts[:2])
                # module_key = '.'.join(parts[:4 if parts[3] in ['conv', 'bn'] else 3])
                module_key = '.'.join(parts[:3])
            elif len(parts) <= 2:
                module_key = '.'.join(parts[:1])  # 处理最后层模块
            else:
                module_key = '.'.join(parts[:2])  # 处理顶层模块

            # 新模块开始
            if module_key != current_module_key:
                current_module_key = module_key
                current_major = len(major_groups)
                major_groups[current_major] = []

            major_groups[current_major].append(layer)

        # 2. 合并过小的组(小于3层的合并到前一组)
        merged_groups = OrderedDict()
        current_group_idx = 0
        temp_group = []

        for group in major_groups.values():
            if len(temp_group) + len(group) < 3 and merged_groups:
                # 合并到前一组
                merged_groups[current_group_idx - 1].extend(group)
            else:
                merged_groups[current_group_idx] = group
                temp_group = group.copy()
                current_group_idx += 1

        # 3. 根据max_unfreeze_stages进行二次分组
        if not hasattr(self, 'max_unfreeze_stages'):
            return merged_groups

        num_layers = len(merged_groups)
        if self.max_unfreeze_stages >= len(merged_groups):
            return merged_groups

        # 需要细分的情况
        group_size = int(num_layers / self.max_unfreeze_stages + 0.5)

        # 4.1 自上而下分组（Top-Bottom顺序）
        layer_groups = {}
        for i in range(self.max_unfreeze_stages):
            start = i * group_size
            end = (i + 1) * group_size if i < self.max_unfreeze_stages - 1 else num_layers
            for j, k in enumerate(merged_groups.keys()):
                if start <= j < end:
                    if i not in layer_groups:
                        layer_groups[i] = merged_groups[k]
                    else:
                        layer_groups[i] += merged_groups[k]

        # 4.2 反转键顺序为自下而上分组（Top-Bottom → Bottom-Up）
        reversed_keys = sorted(layer_groups.keys(), reverse=True)
        bottom_up_groups = {
            new_idx: layer_groups[original_key]
            for new_idx, original_key in enumerate(reversed_keys)
        }

        return bottom_up_groups

    def _init_freeze_strategy(self) -> None:
        """初始化冻结策略"""
        freeze_strategy = self.freeze_config['freeze_strategy']

        # 情况1: 渐进式解冻 [0, 25, 50] 在指定epoch解冻下一阶段
        if freeze_strategy == 'auto':
            unfreeze_epochs = self.freeze_config['unfreeze_epochs']
            self.max_unfreeze_stages = len(unfreeze_epochs)  # 按照unfreeze_epochs分为几个阶段
            self._freeze_layers_progressive()

        # 情况2: 按层索引冻结 [0, 2]
        elif freeze_strategy == 'index':
            freeze_layers_indices = self.freeze_config['freeze_layers_indices']
            self._freeze_layers_by_index(freeze_layers_indices)

        # 情况3: 按层名称冻结 ['conv1', 'bn1']
        elif freeze_strategy == 'name':
            freeze_layers_names = self.freeze_config['freeze_layers_names']
            self._freeze_layers_by_name(freeze_layers_names)
        else:
            raise ValueError(f"Invalid freeze_strategy: {freeze_strategy}")

        # 记录当前解冻的层  TODO 后续优化其他网络/修改监测方式
        temp = ['base_model.base_model.conv1.weight',
                'base_model.base_model.layer1.0.conv1.weight',
                'base_model.base_model.layer2.0.conv1.weight',
                'base_model.base_model.layer3.0.conv1.weight',
                'base_model.base_model.layer4.0.conv1.weight',
                'fc_comb.weight',
                'head.fc_mask.weight']
        for name, param in self.named_parameters():
            if name in temp:
                print(f'{name}.grad: {param.requires_grad}')

    def _freeze_layers_progressive(self, logger=None) -> None:
        """渐进式解冻策略"""
        layer_groups = self._get_layer_groups()

        # 冻结所有层(除了constant_unfreeze_layers涉及层)
        for name, param in self.named_parameters():
            skip_flag = False
            for unfreeze_layer in self.constant_unfreeze_layers:
                if unfreeze_layer in name:  # unfreeze_layer层始终不冻结
                    skip_flag = True
                    break

            if not skip_flag:
                param.requires_grad_(False)

        # 解冻当前阶段及之前的所有层
        for stage in range(self.current_unfreeze_stage):
            for layer_name in layer_groups.get(stage, []):
                for name, param in self.named_parameters():
                    if name == layer_name:
                        param.requires_grad_(True)
                        break

        # 记录当前解冻的层  TODO 后续优化其他网络/修改监测方式
        if logger:
            temp = ['base_model.base_model.conv1.weight',
                    'base_model.base_model.layer1.0.conv1.weight',
                    'base_model.base_model.layer2.0.conv1.weight',
                    'base_model.base_model.layer3.0.conv1.weight',
                    'base_model.base_model.layer4.0.conv1.weight',
                    'fc_comb.weight',
                    'head.fc_mask.weight']
            for name, param in self.named_parameters():
                if name in temp:
                    logger.info(f'{name}.grad: {param.requires_grad}')

    def _freeze_layers_by_index(self, layer_indices: List[int]) -> None:
        """按层索引冻结策略

        Args:
            layer_indices: 要冻结的层索引范围，如[0,2]表示冻结0-2层
        """
        if len(layer_indices) != 2:
            raise ValueError("Layer indices should be a list of 2 elements [start, end]")

        # 获取所有可训练层的名称
        all_layers = self._get_all_layers()
        start, end = layer_indices[0], min(layer_indices[1], len(all_layers ) -1)

        for i, name in enumerate(all_layers):
            for param_name, param in self.named_parameters():
                if param_name == name:
                    param.requires_grad_(i < start or i > end)
                    break

    def _freeze_layers_by_name(self, layer_names: List[str]) -> None:
        """按层名称冻结策略

        Args:
            layer_names: 要冻结的层名称列表
        """
        for name, param in self.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad_(False)

    def unfreeze_next_stage(self, logger=None) -> bool:
        """解冻下一阶段层(渐进式解冻用)"""
        self.current_unfreeze_stage += 1
        if hasattr(self, 'max_unfreeze_stages') and self.current_unfreeze_stage <= self.max_unfreeze_stages:
            print(f"Unfreezing stage: {self.current_unfreeze_stage}")
            self._freeze_layers_progressive(logger)
            return True
        return False

    def forward(self, inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播

        Args:
            inputs (torch.Tensor): 输入张量

        Returns:
            如果是相似学习模式，返回(output, features)
            否则只返回output
        """
        features = self.base_model(inputs)

        if self.fc_comb is not None:
            features = F.relu(self.fc_comb(features))

        output = self.head(features)

        if self.cfg['return_feature_map']:
            return output, features
        return output


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_yaml = os.path.join(script_dir, 'config/template_detect_train.yaml')

    # 读取yaml文件
    yaml_list = [train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 创建模型
    model = Model(cfg_data['model_params'])
    res = model(torch.zeros(2, 1, 512, 640))
    print()
