import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from air_track.utils import common_utils, extract_number, normalize, check_and_change_img_size, build_augmenter, \
    check_data_is_normalized, load_json, process_data_parts_config


class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, stage, cfg):
        self.stage = stage
        self.cfg = cfg
        self.dataset_params = cfg['dataset_params']

        # 优先使用cfg_data中的part配置
        if self.cfg.get('part_train') and self.cfg.get('part_val'):
            self.json_parts = self.cfg
        else:
            # 数据配置文件处理
            self.parts_config_path = process_data_parts_config(self.cfg['parts_config_path'])
            # 读取数据配置文件
            self.json_parts = load_json(self.parts_config_path)

        if stage == self.cfg['stage_train']:
            self.is_training = True
            self.parts = self.json_parts['part_train']
        elif stage == self.cfg['stage_valid']:
            self.is_training = False
            self.parts = self.json_parts['part_val']
        else:
            self.is_training = False
            self.parts = self.json_parts['part_test']

        self.data_dir = cfg['data_dir']
        self.max_pixel = self.cfg['max_pixel']
        self.input_channel = self.cfg['model_params']['input_channel']
        self.img_size_w, self.img_size_h = self.dataset_params['img_size']
        self.normalize = self.dataset_params['normalize']
        self.normalize_to_range = list(self.dataset_params['normalize_to_range'])
        self.return_torch_tensors = self.dataset_params['return_torch_tensors']

        # 数据增强配置
        self.aug_cfg = self.dataset_params.get('augmentation', {})
        # 仅在训练阶段且8bit图像上启用增强
        self.enable_aug = self.is_training and self.dataset_params['enable_augment']
        # 数据增强初始化
        self.augmenter = build_augmenter(self.aug_cfg)

        self.npy_paths = []
        self.labels = []
        for part in self.parts:
            # self.parts支持两种格式（list和dict）
            if isinstance(self.parts, dict):
                # 来自于json文件配置
                part_value = self.parts[part]
                shuffle = part_value['shuffle']
                frame_nums = int(part_value['nums'])
            else:
                shuffle = False
                frame_nums = np.iinfo(np.int64).max

            npy_folder = os.path.join(self.cfg['data_dir'], part, self.cfg['npy_folder'])
            label_file = os.path.join(self.data_dir, part, cfg['label_file'])

            npy_paths = [f for f in os.listdir(npy_folder) if f.endswith(cfg['npy_format'])]
            npy_paths.sort(key=extract_number)
            npy_paths = [os.path.join(npy_folder, f) for f in npy_paths]
            labels = pd.read_csv(label_file).values

            if shuffle:
                random.seed(self.cfg['seed'])
                random.shuffle(npy_paths)

            # 限制帧数
            npy_paths = npy_paths[:frame_nums]

            labels_new = []
            for npy_path in npy_paths:
                num = int(os.path.basename(npy_path).split('.')[0])
                item = labels[num - 1]
                if int(item[0]) == num:
                    labels_new.append(item)
                else:
                    assert ValueError('npy数据与label标签不匹配')

            self.npy_paths.extend(npy_paths)
            self.labels.extend(labels_new)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = str(self.npy_paths[idx])
        num = int(os.path.basename(npy_path).split('.')[0])
        data = np.load(npy_path)
        item = self.labels[idx]
        label = item[2:].astype(np.float32)

        if int(item[0]) != num:
            print('跳过npy数据与标签不匹配的帧')
            return self.__getitem__(idx + 1)

        # 校对并改变图像尺寸
        if len(data.shape) == 3 and data.shape[0] == self.input_channel:
            data = np.transpose(data, (1, 2, 0))
        if self.enable_aug and check_data_is_normalized(data):
            data = data * 255
        data = data.astype(np.uint8)
        data = check_and_change_img_size(data, img_size_w=self.img_size_w, img_size_h=self.img_size_h)

        if len(data.shape) == 2:  # 单通道图像
            # 在第-1维增加维度，与三通道图像相匹配
            data = np.expand_dims(data, axis=-1)

        # 应用增强
        if self.enable_aug:
            seq = self.augmenter.to_deterministic()
            data = seq.augment_image(data)

        if len(data.shape) == 3 and data.shape[-1] == self.input_channel:
            data = np.transpose(data, (2, 0, 1))

        if self.normalize:
            data = data / self.cfg['max_pixel']

        if self.return_torch_tensors:
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()

        res = {'input': data, 'label': label}

        return res


if __name__ == '__main__':
    """用于测试阶段使用，现作为展示代码，若需单独使用，将下段代码拿出去到单独脚本中使用，莫要修改此文件"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_yaml = os.path.join(script_dir, 'config/dataset.yaml')
    train_yaml = os.path.join(script_dir, 'config/classify_train.yaml')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]

    # 合并读取若干个yaml的配置文件内容
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    common_utils.reprod_init(seed=cfg_data['seed'])

    dataset = ClassifierDataset(stage='train', cfg=cfg_data)

    dataloader = DataLoader(
        dataset,
        num_workers=0,
        shuffle=True,
        batch_size=1,
    )

    for i, data in enumerate(dataloader):
        print(i)
        img_tensor = data['input'][0]
        max_pixel = cfg_data['max_pixel']  # 从配置读取最大值（通常为255）
        img_np = img_tensor.numpy().transpose(1, 2, 0) * max_pixel
        img_np = img_np.astype(np.uint8)

        cv2.imwrite(f'/media/linana/2C5821EF5821B88A1/yanqing_data/temp/{i}.png', img_np)

        # 显示结果
        # cv2.imshow('Augmentation Demo', img_np)
        # # cv2.imshow(f'{batch["img_path"]}', img_np)
        # # cv2.waitKey()
        # if cv2.waitKey(2000) == 27:
        #     break
