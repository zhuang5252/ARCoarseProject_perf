import copy
import os
import cv2
import numpy as np
import pandas as pd
import torch.utils.data
from typing import List
from torch.utils.data import DataLoader
from air_track.aligner.utils.transform_utils import build_geom_transform
from air_track.detector.data.data_load import load_datasets
from air_track.detector.data.base_dataset.gaussian_render import render_y
from air_track.detector.utils.detect_utils import argmin2d
from air_track.detector.utils.transform_utils import apply_transform
from air_track.detector.visualization.visualize_and_save import draw_feature_img_orig_data
from air_track.utils import common_utils, check_and_change_img_size, transform_imgs


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        self.stage = stage
        self.cfg_data = cfg_data
        self.dataset_params = cfg_data['dataset_params']

        if stage == cfg_data['stage_train']:
            self.is_training = True
            parts = cfg_data['part_train']
        else:  # stage == cfg_data.stage_valid
            self.is_training = False
            parts = cfg_data['part_val']

        self.frames = []
        self.frame_nums_with_objs: List[int] = []
        self.frame_nums_with_distance_objs: List[int] = []

        self.parts = parts

    def load_dataset(self):
        """读取对应 self.stage 的数据"""
        # 读取数据
        frames, frame_nums_with_objs, frame_nums_with_distance_objs = load_datasets(cfg=self.cfg_data,
                                                                                    parts=self.parts,
                                                                                    stage=self.stage)

        self.frames = frames
        self.frame_nums_with_objs = frame_nums_with_objs
        self.frame_nums_with_distance_objs = frame_nums_with_distance_objs

        print(self.stage, len(self.frames), len(self.frame_nums_with_objs), len(self.frame_nums_with_distance_objs))

    def __len__(self) -> int:
        return len(self.frames)


class DetectDataset(BaseDataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        super().__init__(stage, cfg_data)
        self.load_dataset()
        self.frame_step = self.dataset_params['frame_step']  # 1 frame_step图像帧之间的步长
        self.input_frames = self.dataset_params['input_frames']  # 2 input_frames决定连续取几帧数据，为2时，取前后相邻两帧

        self.img_size_w, self.img_size_h = self.dataset_params['img_size']  # 指定图像尺寸
        self.down_scale = self.dataset_params['down_scale']  # 2 模型及数据下采样的比例
        self.target_min_size = self.dataset_params['target_min_size']  # 2 目标最小尺寸

        self.frame_align_flag = self.dataset_params.get('frame_align_flag', False)
        self.return_torch_tensors = self.dataset_params.get('return_torch_tensors', True)
        self.gamma_flag = self.dataset_params.get('gamma_flag', False)

        # 使用帧对齐标志为真
        if self.frame_align_flag:
            self.transforms = self.load_align_transforms_pkl(self.parts)

    def load_align_transforms_pkl(self, parts):
        """读取帧对齐所得的pkl文件（存储相邻两帧的帧对齐参数）"""
        transforms = {}
        # 下部分代码所需文件在帧对齐的预测offset中生成
        for part in parts:
            with common_utils.timeit_context('load transforms ' + str(part)):
                transforms[part] = {}
                transforms_dir = os.path.join(self.cfg_data['transforms_dir'], f'{part}')
                for fn in os.listdir(transforms_dir):
                    if fn.endswith('.pkl'):
                        flight_id = fn[:-4]
                        transforms[part][flight_id] = pd.read_pickle(f'{transforms_dir}/{flight_id}.pkl')
        return transforms

    def get_prev_step_transforms(self, cur_frame_num, flight_transforms):
        """
        从帧对齐的self.transforms中取出prev的帧对齐参数

        存储顺序为: frames_transform = [frame_step、frame_step*2 ......]
        代表的意思为: 当前帧与前一帧的帧对齐参数、前一帧与前两帧的帧对齐参数，以此类推
        """
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
                    transform = build_geom_transform(translation_x=dx, translation_y=dy,
                                                     scale_x=1.0, scale_y=1.0,
                                                     angle=angle,
                                                     return_params=True)

                    # 变换矩阵
                    if frame_step not in frames_transform:
                        frames_transform[frame_step] = transform
                    else:
                        frames_transform[frame_step] = np.matmul(transform, frames_transform[frame_step])  # 矩阵乘得到累积变换矩阵

                # pkl文件中不存在frame_num这一帧的变换矩阵
                else:
                    transform = np.eye(3, 3)  # 不进行变换
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

    def get_prev_step_frame(self, cur_frame, index, frame_step, input_frames):
        """
        从dataset的frames中取出prev的数据

        存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
        """
        prev_frames = []

        # 超参数设置为小于2帧或者步长不大于0，则返回空列表
        if input_frames < 2 or frame_step <= 0:
            assert ValueError('input_frames < 2 or frame_step <= 0. ')

        for i in range(frame_step, input_frames, frame_step):
            interval_index = i * frame_step

            prev_frame = self.frames[index - interval_index]
            prev_flight_id = prev_frame['flight_id']
            cur_flight_id = cur_frame['flight_id']
            prev_frame_num = prev_frame['frame_num']
            cur_frame_num = cur_frame['frame_num']

            if prev_flight_id == cur_flight_id and prev_frame_num == cur_frame_num - interval_index:
                prev_frames.append(prev_frame)

        return prev_frames

    def read_prev_images(self, cur_img, prev_frames):
        """读取prev的图像数据"""
        prev_images = []

        # 迭代读取
        for prev_frame in prev_frames:
            prev_img = cv2.imread(prev_frame['img_path'], cv2.IMREAD_UNCHANGED)
            prev_img = check_and_change_img_size(prev_img, self.img_size_w, self.img_size_h)
            # 若没有前一帧，则两帧相同
            if prev_img is None:
                prev_img = cur_img.copy()
            prev_images.append(prev_img)

        return prev_images

    def gamma_and_normalize(self, res, cfg, gamma_flag=False):
        """gamma变换和归一化操作"""
        gamma_aug = 2 ** np.random.normal(cfg['gamma_loc'], cfg['gamma_scale'])

        for k in list(res.keys()):
            if k == 'image' or k.startswith('prev_image_aligned'):
                res[k] = torch.from_numpy(res[k].astype(np.float32) / self.cfg_data['max_pixel']).float()

                if self.is_training and gamma_flag:
                    res[k] = torch.pow(res[k], gamma_aug)

            elif isinstance(res[k], np.ndarray):
                res[k] = torch.from_numpy(res[k].astype(np.float32)).float()

        return res

    def __getitem__(self, index):
        index = 28520 - 262
        # 当前帧
        same_frame_flag = False
        cur_frame = self.frames[index]

        # 判断index是否满足向前找input_frames帧
        if index - self.frame_step * (self.input_frames - 1) < 0:
            return self.__getitem__(index + 1)

        # 获取前 input_frames - 1 帧
        prev_frames = self.get_prev_step_frame(cur_frame, index, self.frame_step, self.input_frames)

        # 没找到指定的前 input_frames - 1 帧
        if len(prev_frames) != self.input_frames - 1:
            if index + 1 < len(self.frames):
                return self.__getitem__(index + 1)
            else:
                prev_frames = [cur_frame]
                same_frame_flag = True

        # 路径需要满足所需层级
        cur_img = cv2.imread(cur_frame['img_path'], cv2.IMREAD_UNCHANGED)
        cur_img = check_and_change_img_size(cur_img, self.img_size_w, self.img_size_h)

        # 如果图像不存在，index + 1
        if cur_img is None:
            return self.__getitem__(index + 1)

        # 使用cv2读取前几帧图像，返回列表
        prev_images = self.read_prev_images(cur_img, prev_frames)

        '''标签及图像应用变换'''
        # 使用transform对prev_frames所有帧的detections进行转换
        cur_step_detections = cur_frame['detections']  # 当前帧的detections

        # 应用帧对齐（变换标签和图像）
        if self.frame_align_flag and not same_frame_flag:
            # 从pkl文件中读取帧对齐参数
            flight_transforms = self.transforms[cur_frame['part']][cur_frame['flight_id']]
            # 获取指定帧的变换矩阵
            frames_transforms = self.get_prev_step_transforms(cur_frame['frame_num'], flight_transforms)
            # 对标签(detections)进行转换
            prev_step_detections = self.apply_transform_in_prev_frames(prev_frames, frames_transforms)[0]['detections']
            # 对图像进行转换
            prev_images = transform_imgs(prev_images, frames_transforms)
        else:
            # 前一帧的detections
            prev_step_detections = prev_frames[0]['detections']

        # 对两帧图像结合
        input_imgs = [cur_img] + prev_images

        '''高斯、下采样，制作标签'''
        # TODO 标签制作现只支持连续两帧（后续若想多个帧，需考虑下方代码render_y中prev_item['cx'] - item['cx']的设计，多帧图像取均值或者只用前一帧来计算，或者其他）
        res = render_y(self.cfg_data, cur_step_detections, prev_step_detections, img_w=self.img_size_w,
                       img_h=self.img_size_h, down_scale=self.down_scale, target_min_size=self.target_min_size)

        save_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/visual_feature_orig/'
        max_iloc_y, max_iloc_x = argmin2d(res['cls'][0])
        other_format = '_no_star'
        for key in res.keys():
            draw_feature_img_orig_data(res['cls'], save_path + f'label_{key}{other_format}.png',
                                       is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)

        res['part'] = cur_frame['part']
        res['flight_id'] = cur_frame['flight_id']
        res['image_name'] = cur_frame['img_name']
        res['cur_step_detections'] = cur_step_detections

        res['idx'] = index
        res['image'] = input_imgs[0]

        # 取出前几帧的图像
        for i, prev_img in enumerate(input_imgs[1:]):
            res[f'prev_image_aligned{i}'] = prev_img

        # 是否返回torch_tensors
        if self.return_torch_tensors:
            # gamma转换和归一化
            res = self.gamma_and_normalize(res, cfg=self.dataset_params, gamma_flag=self.gamma_flag)

        return res


if __name__ == '__main__':
    dataset_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/detector/config/dataset.yaml'
    train_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/detector/config/hrnet_train.yaml'

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]

    # 合并读取若干个yaml的配置文件内容
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    common_utils.reprod_init(seed=cfg_data['seed'])

    # base_dataset = BaseDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)

    dataset_train = DetectDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)

    train_dl = DataLoader(
        dataset_train,
        num_workers=0,
        shuffle=True,
        batch_size=1,
    )

    for i, item in enumerate(train_dl):
        print(i)
