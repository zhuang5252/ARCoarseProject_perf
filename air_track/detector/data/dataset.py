# -*- coding: utf-8 -*-
import os
import math
import torch
import random
from typing import Dict, Any
# import pytorch_lightning as pl
from torch.utils.data import DataLoader
from air_track.detector.data.base_dataset.dataset_aot import DetectDatasetAot
from air_track.detector.data.base_dataset.dataset_AR_csv import DetectDatasetARCsv
from air_track.detector.data.base_dataset.dataset_yolo_txt import DetectDatasetYoloTxt
from air_track.utils import common_utils


class DetectDatasetFactory(torch.utils.data.Dataset):
    _DATASET_MAPPING = {
        'yolo_txt': DetectDatasetYoloTxt,
        'ar_csv': DetectDatasetARCsv,
        'aot': DetectDatasetAot,
    }

    def __init__(
            self,
            stage: str,
            cfg_data: Dict[str, Any],
    ) -> None:
        # 参数校验
        if not isinstance(cfg_data, dict):
            raise ValueError("cfg_data must be a dictionary")
        if 'dataset_params' not in cfg_data:
            raise KeyError("cfg_data must contain 'dataset_params'")

        self.stage = stage
        self.cfg_data = cfg_data
        self.datasets = []
        self.sample_weights = []

        self.dataset_params = self.cfg_data['dataset_params']
        if 'dataset_type' not in self.dataset_params:
            raise KeyError("cfg_data must contain 'dataset_type'")

        self.dataset_type = self.dataset_params['dataset_type'].lower()

        # 动态加载数据集类
        if self.dataset_type not in self._DATASET_MAPPING:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # 初始化子数据集
        for config in cfg_data.get('admix_datasets', []):
            sub_cfg = cfg_data.copy()
            sub_cfg['data_dir'] = config['data_dir']
            sub_cfg[f'part_{stage}'] = config.get('parts', [])
            # sub_cfg['classes'] = config['classes']
            # sub_cfg['nb_classes'] = config['nb_classes']

            dataset = self._DATASET_MAPPING[self.dataset_type](stage, sub_cfg)
            self.datasets.append(dataset)
            self.sample_weights.append(config.get('weight', 1.0))

        # 如果没有配置融合，则使用单一数据集
        if not self.datasets:
            self.datasets = [self._DATASET_MAPPING[self.dataset_type](stage, cfg_data)]
            self.sample_weights = [1.0]

        # 归一化权重
        total_weight = sum(self.sample_weights)
        self.sample_weights = [w / total_weight for w in self.sample_weights]

        # 计算各数据集应贡献的样本数
        min_len = min(len(d) for d in self.datasets)  # 新旧数据最小样本数
        # 按照最小样本数找到相应的索引
        self.base_dataset_idx = [i for i, d in enumerate(self.datasets) if len(d) == min_len][0]
        self.base_dataset_len = min_len

        # 按照最小长度和权重计算总样本数
        self.total_len = int(self.base_dataset_len / self.sample_weights[self.base_dataset_idx] + 0.5)

        # 每个数据集需要贡献的样本数
        self.required_counts = None
        # 记录每个数据集已使用的样本索引（用于超出最小长度的数据集的不放回采样）
        self.used_indices = None

        # 初始化采样状态
        self.reset_epoch()

    def reset_epoch(self):
        """每个epoch开始时调用"""
        # 记录每个数据集已使用的样本索引
        self.used_indices = [set() for _ in self.datasets]

        # 基础数据集使用全部样本
        base_indices = list(range(len(self.datasets[self.base_dataset_idx])))
        # random.shuffle(base_indices)  # 在dataset里已经打乱过一次了
        self.used_indices[self.base_dataset_idx] = set(base_indices)

        # 计算其他数据集需要的样本数
        self.required_counts = [
            math.ceil(self.total_len * w) if i != self.base_dataset_idx
            else len(self.datasets[self.base_dataset_idx])
            for i, w in enumerate(self.sample_weights)
        ]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """默认基础数据集优先，其他数据集按权重轮询采样"""
        # TODO: 目前为同一个batch随机按照权重分配，同一个batch内可能相同数据集，后续按需求优化
        # 确定应该从哪个数据集获取样本
        if idx < len(self.datasets[self.base_dataset_idx]):
            dataset_idx = self.base_dataset_idx
            sample_idx = list(self.used_indices[dataset_idx])[idx]
        else:
            # 对其他数据集进行轮询采样
            dataset_idx, sample_idx = None, None
            for i, (d, req) in enumerate(zip(self.datasets, self.required_counts)):
                if i == self.base_dataset_idx:  # 判断是否为最小长度的基础数据集
                    continue

                if len(self.used_indices[i]) < req:  # 检查是否还需要更多样本
                    # 不放回的采样
                    available = set(range(len(d))) - self.used_indices[i]  # 计算可用样本
                    if available:
                        if self.stage == 'train':  # 训练阶段随机选择
                            # TODO 每一轮的每个batch都是随机的，新旧数据组合也就是随机的
                            sample_idx = random.choice(list(available))  # 随机选择是为了尽可能多的使用旧数据
                        else:
                            sample_idx = idx % self.base_dataset_len  # 验证阶段不随机
                        self.used_indices[i].add(sample_idx)  # 记录已使用
                        dataset_idx = i  # 设置数据集索引
                        break  # 找到后退出循环

            # 边界情况处理，非基础数据集耗尽
            if dataset_idx is None:
                # 回退到基础数据集，重复采样
                dataset_idx = self.base_dataset_idx
                sample_idx = list(self.used_indices[dataset_idx])[idx % len(self.used_indices[dataset_idx])]
                # 可以添加日志记录这种情况
                import warnings
                warnings.warn("Falling back to base dataset due to exhausted non-base samples")

        return self.datasets[dataset_idx][sample_idx]


class DetectDataset(torch.utils.data.Dataset):
    def __init__(self, stage, cfg_data):
        self.stage = stage
        self.cfg_data = cfg_data
        self.inner_dataset = DetectDatasetFactory(stage, cfg_data)  # 内部实际数据集

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        """必须实现该方法才能被DataLoader使用"""
        return self.inner_dataset[idx]


# class PlDetectDataset(pl.LightningDataModule):
#     def __init__(self, stage, cfg_data):
#         # 参数校验
#         super().__init__()
#         if not isinstance(cfg_data, dict):
#             raise ValueError("cfg_data must be a dictionary")
#         if 'dataset_params' not in cfg_data:
#             raise KeyError("cfg_data must contain 'dataset_params'")
#
#         self.stage = stage
#         self.cfg_data = cfg_data
#
#         self.dataset_params = self.cfg_data['dataset_params']
#         if 'dataset_type' not in self.dataset_params:
#             raise KeyError("cfg_data must contain 'dataset_type'")
#
#         # TODO 后续视情况改为注册器形式
#         self.dataset_type = self.dataset_params['dataset_type'].lower()
#         if self.dataset_type == 'yolo_txt':
#             self.dataset = DetectDatasetYoloTxt(stage=self.stage, cfg_data=self.cfg_data)
#         elif self.dataset_type == 'ar_csv':
#             self.dataset = DetectDatasetARCsv(stage=self.stage, cfg_data=self.cfg_data)
#         elif self.dataset_type == 'aot':
#             self.dataset = DetectDatasetAot(stage=self.stage, cfg_data=self.cfg_data)
#         else:
#             raise ValueError(f"Unknown dataset type: {self.dataset_type}")
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def set_dataloader(self):
#         """构建数据"""
#         if self.stage == 'train':
#             dataloader = DataLoader(
#                 self.dataset,
#                 num_workers=self.cfg_data['train_data_loader']['num_workers'],
#                 batch_size=self.cfg_data['train_data_loader']['batch_size'],
#                 pin_memory=self.cfg_data['train_data_loader']['pin_memory'],
#                 shuffle=self.cfg_data['train_data_loader']['shuffle']
#             )
#         elif self.stage == 'val':
#             dataloader = DataLoader(
#                 self.dataset,
#                 num_workers=self.cfg_data['val_data_loader']['num_workers'],
#                 batch_size=self.cfg_data['val_data_loader']['batch_size'],
#                 pin_memory=self.cfg_data['val_data_loader']['pin_memory'],
#                 shuffle=self.cfg_data['val_data_loader']['shuffle']
#             )
#         elif self.stage == 'test':
#             dataloader = DataLoader(
#                 self.dataset,
#                 num_workers=self.cfg_data['test_data_loader']['num_workers'],
#                 batch_size=self.cfg_data['test_data_loader']['batch_size'],
#                 pin_memory=self.cfg_data['test_data_loader']['pin_memory'],
#                 shuffle=self.cfg_data['test_data_loader']['shuffle']
#             )
#         else:
#             raise ValueError(f"Unknown stage: {self.stage}")
#
#         return dataloader


if __name__ == '__main__':
    # 测试代码
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_yaml = os.path.join(script_dir, 'config/dataset_RGBT_Tiny_scene_finetune_admix.yaml')
    train_yaml = os.path.join(script_dir, 'config/detect_train.yaml')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]

    # 合并读取若干个yaml的配置文件内容
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    common_utils.reprod_init(seed=cfg_data['seed'])

    # 初始化数据集
    dataset = DetectDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)
    print(f"Total samples: {len(dataset)}")

    # 验证DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 调试时建议设为0避免多进程问题
        collate_fn=lambda x: x  # 简单合并样本
    )

    # pl_dataset = PlDetectDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)
    # dataset = pl_dataset.dataset
    # dataloader = pl_dataset.set_dataloader()

    for e in range(5):
        print(f"\nEpoch {e}:")
        dataset.inner_dataset.reset_epoch()
        # 测试数据加载
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i}:")
            for sample in batch:
                print(f"Image shape: {sample['image'].shape if 'image' in sample else 'N/A'}")
                print(f"From: {sample.get('img_path', 'Unknown')}")
