# -*- coding: utf-8 -*-
# @Author    : 
# @File      : base_dataset.py
# @Created   : 2025/6/30 下午2:23
# @Desc      : 数据集dataset基类

import os
import cv2
import copy
import torch
import random
import numpy as np
import pandas as pd
from typing import List, Any, Dict
from air_track.aligner.utils.transform_utils import gen_transform
from air_track.detector.utils.detect_utils import augment_single
from air_track.detector.utils.transform_utils import apply_transform
from air_track.utils import build_augmenter, check_and_change_img_size, common_utils, check_image_bit_depth, \
    load_json, process_data_parts_config


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        self.stage = stage
        self.cfg_data = cfg_data
        self.dataset_params = cfg_data['dataset_params']
        self.classes = self.cfg_data['classes']
        self.max_pixel = self.cfg_data['max_pixel']
        self.img_read_method = self.dataset_params['img_read_method'].lower()
        self.frame_step = self.dataset_params['frame_step']  # 1 frame_step图像帧之间的步长
        self.input_frames = self.dataset_params['input_frames']  # 2 input_frames决定连续取几帧数据，为2时，取前后相邻两帧

        # 优先使用cfg_data中的part配置
        if self.cfg_data.get('part_train') and self.cfg_data.get('part_val'):
            self.json_parts = self.cfg_data
        else:
            # 数据配置文件处理
            self.parts_config_path = process_data_parts_config(self.cfg_data['parts_config_path'])
            # 读取数据配置文件
            self.json_parts = load_json(self.parts_config_path)

        # 初始化数据集阶段
        if self.stage == self.cfg_data['stage_train']:
            self.is_training = True
            self.parts = self.json_parts['part_train']
        elif self.stage == self.cfg_data['stage_valid']:
            self.is_training = False
            self.parts = self.json_parts['part_val']
        else:
            self.is_training = False
            self.parts = self.json_parts['part_test']

        self.img_size_w, self.img_size_h = self.dataset_params['img_size']  # 指定图像尺寸
        self.train_val_test_scale = self.dataset_params['train_val_test_scale']  # 训练集、验证集、测试集比例
        self.down_scale = self.dataset_params['down_scale']  # 1 模型及数据下采样的比例
        self.target_min_size = self.dataset_params['target_min_size']  # 5 目标最小尺寸

        self.frame_align_flag = self.dataset_params.get('frame_align_flag', False)
        self.return_torch_tensors = self.dataset_params.get('return_torch_tensors', True)
        self.gamma_flag = self.dataset_params.get('gamma_flag', False)

        # 数据增强配置
        self.aug_cfg = self.dataset_params.get('augmentation', {})
        # 仅在训练阶段且8bit图像上启用增强
        self.enable_aug = self.is_training and self.dataset_params['enable_augment']
        # 数据增强初始化
        self.augmenter = build_augmenter(self.aug_cfg)

        if self.train_val_test_scale != 'None':
            if len(self.train_val_test_scale) == 3:
                self.train_scale = float(self.train_val_test_scale[0])
                self.val_scale = float(self.train_val_test_scale[1])
                self.test_scale = float(self.train_val_test_scale[2])
            else:
                raise Exception("train_val_test_scale is not 'None', train_val_test_scale must be a list with 3 elements")

    def split_data(self, frames, shuffle=True):
        """拆分数据集"""
        # 单帧模式，打乱数据集顺序
        if self.input_frames == 1 and shuffle:
            random.seed(self.cfg_data['seed'])
            random.shuffle(frames)

        if self.train_val_test_scale != 'None':
            l = len(frames)
            if self.stage == self.cfg_data['stage_train']:
                frames = frames[: int(l * self.train_scale)]
            elif self.stage == self.cfg_data['stage_valid']:
                frames = frames[int(l * self.train_scale): int(l * (self.train_scale + self.val_scale))]
            else:
                frames = frames[
                            int(l * (self.train_scale + self.val_scale)):
                            int(l * (self.train_scale + self.val_scale + self.test_scale))]
        else:
            frames = frames

        return frames

    def load_align_transforms_pkl(self, parts):
        """读取帧对齐所得的pkl文件（存储相邻两帧的帧对齐参数）"""
        transforms = {}
        # 下部分代码所需文件在帧对齐的预测offset中生成
        for part in parts:
            with common_utils.timeit_context('load transforms ' + str(part)):
                transforms[part] = {}
                transforms_dir = os.path.join(self.cfg_data['transforms_dir'], part)
                for fn in os.listdir(transforms_dir):
                    if fn.endswith('.pkl'):
                        flight_id = fn[:-4]
                        # transforms[part] = pd.read_pickle(f'{transforms_dir}/{fn}')
                        transforms[part][flight_id] = pd.read_pickle(f'{transforms_dir}/{flight_id}.pkl')
        return transforms

    def get_prev_step_transforms(self, cur_frame_num, flight_transforms):
        """
        从帧对齐的self.transforms中取出prev的帧对齐参数

        存储顺序为: frames_transform = [frame_step、frame_step*2 ......]
        代表的意思为: 当前帧与前一帧的帧对齐参数、前一帧与前两帧的帧对齐参数，以此类推
        """
        shape = (self.img_size_w, self.img_size_h)
        frames_transform = {}
        for frame_step in range(self.frame_step, self.input_frames, self.frame_step):
            for step in range(0, frame_step):
                """根据飞行变换信息构建几何变换矩阵，并逐步累积这些变换"""
                frame_num = cur_frame_num - step  # 当前帧位置存储的是前一帧到当前帧的帧对齐参数，因此step从0开始
                # 取出这一帧的transform
                flight_transforms_row = flight_transforms[flight_transforms.frame == frame_num]

                # pkl文件中存在frame_num这一帧的变换矩阵
                if len(flight_transforms_row):
                    dx = flight_transforms_row.iloc[0]["dx"]
                    dy = flight_transforms_row.iloc[0]["dy"]
                    angle = flight_transforms_row.iloc[0]["angle"]

                    # 通过pkl文件的offset计算变换矩阵
                    transform = gen_transform(shape, dx, dy, angle)

                    # 变换矩阵
                    if frame_step not in frames_transform:
                        frames_transform[frame_step] = transform
                    else:
                        frames_transform[frame_step] = np.matmul(transform, frames_transform[frame_step])  # 矩阵乘得到累积变换矩阵

                # pkl文件中不存在frame_num这一帧的变换矩阵
                else:
                    transform = np.eye(2, 3)  # 不进行变换
                    frames_transform[frame_step] = transform

        return frames_transform

    def apply_transform_in_prev_frames(self, prev_frames_orig, prev_transforms):
        """对prev_frames中的每一个detections进行变换"""
        prev_frames = copy.deepcopy(prev_frames_orig)
        for prev_step_frame, key in zip(prev_frames, prev_transforms):
            transform = prev_transforms[key]
            for detection in prev_step_frame['detections']:
                # 应用变换矩阵修改detection的值
                apply_transform(detection, transform, self.img_size_w, self.img_size_h)

        return prev_frames

    def get_prev_step_frame(self, frames, cur_frame, index, frame_step, input_frames):
        """
        从dataset的frames中取出prev的数据

        存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
        """
        prev_frames, prev_img_paths = [], []
        cur_img_path = cur_frame['img_path']
        cur_img_base_path = os.path.dirname(cur_img_path)

        # 超参数设置为小于2帧或者步长不大于0，则返回空列表
        if input_frames < 2 or frame_step <= 0:
            assert ValueError('input_frames < 2 or frame_step <= 0. ')

        for i in range(frame_step, input_frames, frame_step):
            interval_index = i * frame_step

            prev_frame = dict(frames[index - interval_index])
            prev_flight_id = prev_frame['flight_id']
            cur_flight_id = cur_frame['flight_id']
            prev_frame_num = prev_frame['frame_num']
            cur_frame_num = cur_frame['frame_num']
            prev_img_path = prev_frame['img_path']
            prev_img_base_path = os.path.dirname(prev_img_path)

            if prev_img_base_path == cur_img_base_path and \
                    prev_flight_id == cur_flight_id and prev_frame_num + interval_index == cur_frame_num:
                prev_frames.append(prev_frame)
                prev_img_paths.append(prev_img_path)

        return prev_frames, prev_img_paths

    def read_prev_images_labels(self, cur_img, cur_label, prev_frames):
        """读取prev的图像和标签数据"""
        prev_images, prev_labels = [], []

        # 迭代读取
        for prev_frame in prev_frames:
            prev_img_path = prev_frame['img_path']
            # 读取标签
            prev_label = prev_frame['detections']
            error = prev_frame['error']
            # 读取图像
            if self.img_read_method == 'gray':
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
            elif self.img_read_method == 'unchanged':
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_UNCHANGED)
            else:
                prev_img = cv2.imread(prev_img_path)

            # 若没有前一帧，则两帧相同
            if error or prev_img is None:
                prev_img = copy.deepcopy(cur_img)
                prev_label = copy.deepcopy(cur_label)
            else:
                # 校验图像深度与配置中的max_pixel是否一致
                check_image_bit_depth(prev_img, prev_img_path, self.max_pixel)
                # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
                prev_img = check_and_change_img_size(prev_img, self.img_size_w, self.img_size_h)

            prev_images.append(prev_img)
            prev_labels.append(prev_label)

        return prev_images, prev_labels

    def apply_augment(self, seq, prev_imgs, cur_img, prev_labels, cur_labels):
        """应用数据增强操作到前序帧和当前帧

        Args:
            seq: 数据增强序列
            prev_imgs: 前序帧图像列表
            cur_img: 当前帧图像
            prev_labels: 前序帧标注列表
            cur_labels: 当前帧标注

        Returns:
            增强后的(prev_imgs, cur_img, prev_labels, cur_labels)
        """
        # 深拷贝输入数据以避免修改原始数据
        prev_imgs = copy.deepcopy(prev_imgs)
        cur_img = copy.deepcopy(cur_img)
        prev_labels = copy.deepcopy(prev_labels)
        cur_labels = copy.deepcopy(cur_labels)

        # 处理前序帧
        for i in range(len(prev_imgs)):
            img_aug, labels_aug = augment_single(seq, prev_imgs[i], prev_labels[i])
            prev_imgs[i] = img_aug
            # 更新标注坐标
            for j, (label, label_aug) in enumerate(zip(prev_labels[i], labels_aug)):
                prev_labels[i][j].update({
                    'cx': float(label_aug[0]),
                    'cy': float(label_aug[1]),
                    'w': float(label_aug[2]),
                    'h': float(label_aug[3])
                })

        # 处理当前帧
        cur_img, cur_labels_aug = augment_single(seq, cur_img, cur_labels)
        for label, label_aug in zip(cur_labels, cur_labels_aug):
            label.update({
                'cx': float(label_aug[0]),
                'cy': float(label_aug[1]),
                'w': float(label_aug[2]),
                'h': float(label_aug[3])
            })

        return prev_imgs, cur_img, prev_labels, cur_labels

    def add_cur_frame_rgb_gray(self, cur_image: np.ndarray,
                               cur_img_path: str,
                               seq: Any,
                               res: Dict[str, Any]
                               ) -> Dict[str, Any]:
        """将当前帧的RGB和灰度图像添加到数据集字典中"""
        input_img_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
        input_img_gray = np.stack([input_img_gray] * 3, axis=-1)

        res['cur_image_rgb'] = cur_image
        res['cur_image_gray'] = input_img_gray

        img_path_filter = cur_img_path.replace('images', 'images_filtered')
        if os.path.exists(img_path_filter):
            input_img_filter = cv2.imread(img_path_filter)
            if self.enable_aug:
                # 应用增强
                input_img_filter = seq.augment_image(input_img_filter)
            input_img_filter = check_and_change_img_size(input_img_filter, self.img_size_w, self.img_size_h)
            res['cur_image_filter'] = input_img_filter

        return res

    def add_prev_frames_rgb_gray(self, prev_images: List[Any],
                                 prev_img_paths: List[str],
                                 seq: Any,
                                 res: Dict[str, Any]
                                 ) -> Dict[str, Any]:
        """将前序帧的RGB和灰度图像添加到数据集字典中"""
        for i, prev_img in enumerate(prev_images):
            input_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            input_img_gray = np.stack([input_img_gray] * 3, axis=-1)

            res[f'prev_image_rgb{i}'] = prev_img
            res[f'prev_image_gray{i}'] = input_img_gray

        for i, prev_img_path in enumerate(prev_img_paths):
            img_path_filter = prev_img_path.replace('images', 'images_filtered')
            if os.path.exists(img_path_filter):
                input_img_filter = cv2.imread(img_path_filter)
                if self.enable_aug:
                    # 应用增强
                    input_img_filter = seq.augment_image(input_img_filter)
                input_img_filter = check_and_change_img_size(input_img_filter, self.img_size_w, self.img_size_h)
                res[f'prev_image_filter{i}'] = input_img_filter

        return res

    def gamma_and_normalize(self, res, cfg, gamma_flag=False):
        """gamma变换和归一化操作"""
        gamma_aug = 2 ** np.random.normal(cfg['gamma_loc'], cfg['gamma_scale'])

        for k in list(res.keys()):
            if 'image' in k:
                if len(res[k].shape) == 2:
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / self.max_pixel).float().unsqueeze(
                        -1)
                else:
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / self.max_pixel).float()

                if self.is_training and gamma_flag:
                    res[k] = torch.pow(res[k], gamma_aug)
                res[k] = res[k].permute(2, 0, 1)

            elif isinstance(res[k], np.ndarray):
                res[k] = torch.from_numpy(res[k].astype(np.float32)).float()

        return res


if __name__ == '__main__':
    pass
