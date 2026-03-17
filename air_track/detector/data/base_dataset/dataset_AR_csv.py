import os
import cv2
import copy
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from air_track.detector.data.base_dataset.base_dataset import BaseDataset
from air_track.detector.data.base_dataset.data_load_AR import load_datasets
from air_track.detector.data.base_dataset.gaussian_render import render_y
from air_track.utils import common_utils, transform_imgs, check_and_change_img_size, check_image_bit_depth, \
    copy_split_data
from air_track.utils.registry import DATASETS


@DATASETS.register(name='DetectDatasetARCsv')
class DetectDatasetARCsv(BaseDataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        super().__init__(stage, cfg_data)
        self.frames = []
        self.frame_nums_with_objs: List[int] = []
        self.frame_nums_with_distance_objs: List[int] = []

        # 加载数据集
        self.load_dataset()
        # 拆分训练集、验证集、测试集
        if self.train_val_test_scale != 'None':
            self.frames = self.split_data(self.frames)
            if self.dataset_params['copy_split_data']:
                # 拷贝拆分后的数据到指定目录
                save_dir = os.path.join(self.cfg_data['data_dir'], 'split_dataset')
                # 提取所有 img_path 的值
                img_paths = [item['img_path'] for item in self.frames]
                copy_split_data(stage, img_paths,
                                self.cfg_data['img_folder'], self.cfg_data['label_folder'],
                                self.cfg_data['img_format'], self.cfg_data['gt_format'])

        # 使用帧对齐标志为真
        if self.frame_align_flag:
            self.transforms = self.load_align_transforms_pkl(self.parts)

    def __len__(self):
        return len(self.frames)

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

    def __getitem__(self, idx):
        # 多帧情况判断index是否满足向前找input_frames帧
        if self.input_frames > 1:
            if idx - self.frame_step * (self.input_frames - 1) < 0:
                return self.__getitem__(idx + 1)

        same_frame_flag = False
        cur_frame = dict(self.frames[idx])
        # 读取图像和标签路径
        cur_img_path = cur_frame['img_path']
        cur_img_name = cur_frame['img_name']
        cur_step_detections = cur_frame['detections']

        # 校验文件存在
        if not os.path.exists(cur_img_path):
            return self.__getitem__(idx + 1)
        # 校验标签是否正常
        if cur_frame['error']:
            return self.__getitem__(idx + 1)

        '''获取前序帧'''
        if self.input_frames > 1:
            # 获取前 input_frames - 1 帧
            prev_frames, prev_img_paths = self.get_prev_step_frame(self.frames, cur_frame, idx, self.frame_step, self.input_frames)

            # 没找到指定的前 input_frames - 1 帧
            if len(prev_frames) != self.input_frames - 1:
                if idx + 1 < len(self.frames):
                    return self.__getitem__(idx + 1)
                else:
                    prev_frames = [cur_frame]
                    same_frame_flag = True
        else:
            prev_frames, prev_img_paths = [], []

        # 读取当前帧图片
        if self.img_read_method == 'gray':
            cur_image_orig = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)
        elif self.img_read_method == 'unchanged':
            cur_image_orig = cv2.imread(cur_img_path, cv2.IMREAD_UNCHANGED)
        else:
            cur_image_orig = cv2.imread(cur_img_path)

        # 如果图像读取失败，index + 1
        if cur_image_orig is None:
            return self.__getitem__(idx + 1)

        # 校验图像深度与配置中的max_pixel是否一致
        check_image_bit_depth(cur_image_orig, cur_img_path, self.max_pixel)
        # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
        cur_image = check_and_change_img_size(cur_image_orig, self.img_size_w, self.img_size_h)

        '''前序帧读取并应用帧对齐'''
        # 使用cv2读取前几帧图像，返回列表
        if self.input_frames > 1:
            if not same_frame_flag:
                prev_images, prev_step_detections = self.read_prev_images_labels(
                    cur_image, cur_step_detections, prev_frames)
            else:
                prev_images, prev_step_detections = [copy.deepcopy(cur_image)], [copy.deepcopy(cur_step_detections)]
            '''应用帧对齐（变换标签和图像）'''
            if self.frame_align_flag:
                # 从pkl文件中读取帧对齐参数
                if cur_frame['flight_id']:
                    flight_transforms = self.transforms[cur_frame['part']][cur_frame['flight_id']]
                else:
                    flight_transforms = self.transforms[cur_frame['part']]
                # 获取指定帧的变换矩阵
                frames_transforms = self.get_prev_step_transforms(cur_frame['frame_num'], flight_transforms)
                # 对标签(detections)进行转换
                prev_frames = self.apply_transform_in_prev_frames(prev_frames, frames_transforms)
                prev_step_detections = [item['detections'] for item in prev_frames]
                # 对图像进行转换
                prev_images = transform_imgs(prev_images, frames_transforms, self.img_size_w, self.img_size_h)
        else:
            prev_images, prev_step_detections = [], [[]]

        '''数据增强'''
        seq = self.augmenter.to_deterministic()
        if self.enable_aug:
            prev_images, cur_image, prev_step_detections, cur_step_detections = self.apply_augment(
                seq, prev_imgs=prev_images, cur_img=cur_image,
                prev_labels=prev_step_detections, cur_labels=cur_step_detections)

        '''高斯、下采样，制作标签'''
        res = render_y(self.cfg_data, prev_step_detections, cur_step_detections,
                       img_w=self.img_size_w, img_h=self.img_size_h,
                       down_scale=self.down_scale, target_min_size=self.target_min_size)

        # 添加图像数据
        res['part'] = cur_frame['part']
        res['flight_id'] = cur_frame['flight_id']
        res['cur_image'] = cur_image
        res['cur_img_orig'] = cur_image_orig
        res['cur_img_path'] = cur_img_path
        res['cur_img_name'] = cur_img_name
        # res['cur_labels'] = [cur_step_detections]

        if self.input_frames > 1:
            res['prev_img_paths'] = prev_img_paths
            # res['prev_labels'] = [prev_step_detections]
            # 取出前几帧的图像
            for i, prev_img in enumerate(prev_images):
                res[f'prev_image_aligned{i}'] = prev_img

        # admix特殊处理，同时要满足单帧模式
        if self.img_read_method == 'admix' and self.input_frames == 1:
            # 在res中增加cur_image_rgb、cur_image_gray、可选cur_image_filter
            res = self.add_cur_frame_rgb_gray(cur_image, cur_img_path, seq, res)
            # 在res中增加prev_image_rgb{i}、prev_image_gray{i}、可选prev_image_filter{i}
            res = self.add_prev_frames_rgb_gray(prev_images, prev_img_paths, seq, res)

        # 是否返回torch_tensors
        if self.return_torch_tensors:
            # gamma转换和归一化
            res = self.gamma_and_normalize(res, cfg=self.dataset_params, gamma_flag=self.gamma_flag)

        return res


if __name__ == '__main__':
    dataset_yaml = '../../config/dataset_AR.yaml'
    train_yaml = '../../config/detect_train_AR.yaml'

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]

    # 合并读取若干个yaml的配置文件内容
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    common_utils.reprod_init(seed=cfg_data['seed'])

    dataset = DetectDatasetARCsv(stage=cfg_data['stage_train'], cfg_data=cfg_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    '''可视化帧对齐及数据增强后的数据'''
    for i, batch in enumerate(dataloader):
        print(f"Processing {i + 1}/{len(dataset)}")
        # 获取数据和原数据
        cur_img_name = batch['cur_img_name'][0]
        cur_img_tensor = batch['cur_image'][0]
        prev_img_tensor = batch['prev_image_aligned0'][0]

        # 转换为numpy并反归一化
        max_pixel = cfg_data['max_pixel']  # 从配置读取最大值（通常为255）
        cur_img_np = cur_img_tensor.numpy().transpose(1, 2, 0) * max_pixel
        cur_img_np = cur_img_np.astype(np.uint8)
        prev_img_np = prev_img_tensor.numpy().transpose(1, 2, 0) * max_pixel
        prev_img_np = prev_img_np.astype(np.uint8)

        merge_result = np.concatenate([cur_img_np, prev_img_np, cur_img_np], axis=2)
        cv2.imwrite(os.path.join('/home/csz_changsha/tmp', f'{cur_img_name}_merge_result.png'), merge_result)

    '''可视化数据增强后的数据'''
    # for i, batch in enumerate(dataloader):
    #     print(f"Processing {i + 1}/{len(dataset)}")
    #     # if i < 3:  # 查看前3个样本
    #     # 获取数据和原数据
    #     # img_tensor = batch['cur_image'][0]
    #     # labels = batch['cur_labels'][0]
    #
    #     img_tensor = batch['prev_image_aligned0'][0]
    #     labels = batch['prev_labels'][0][0]
    #
    #     # 转换为numpy并反归一化
    #     max_pixel = cfg_data['max_pixel']  # 从配置读取最大值（通常为255）
    #     img_np = img_tensor.numpy().transpose(1, 2, 0) * max_pixel
    #     img_np = img_np.astype(np.uint8)
    #
    #     # 转换颜色空间（根据实际存储格式调整）
    #     if dataset.img_read_method == 'gray':
    #         img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    #     else:
    #         img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #
    #     img_np = cv2.resize(img_np, (1920, 1080))
    #
    #     # 绘制边界框和标签
    #     h, w = img_np.shape[:2]
    #     for label in labels:
    #         # 解析归一化坐标
    #         cx = label['cx'] * w
    #         cy = label['cy'] * h
    #         box_w = label['w'] * w
    #         box_h = label['h'] * h
    #
    #         # 计算边界坐标
    #         x1 = int(cx - box_w / 2)
    #         y1 = int(cy - box_h / 2)
    #         x2 = int(cx + box_w / 2)
    #         y2 = int(cy + box_h / 2)
    #
    #         # 绘制矩形
    #         cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #         # 添加类别标签
    #         text = f"{label['cls_name']}"
    #         # cv2.putText(img_np, text, (x1 + 2, y1 + 20),
    #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #
    #     # 显示结果
    #     cv2.imshow('Augmentation Demo', img_np)
    #     # cv2.imshow(f'{batch["img_path"]}', img_np)
    #     # cv2.waitKey()
    #     if cv2.waitKey(2000) == 27:
    #         break

    '''可视化原图和标签'''
    # for i, batch in enumerate(dataloader):
    #     print(f"Processing {i + 1}/{len(dataset)}")
    #     if 80 <= i <= 90:
    #         images = batch['image']
    #         img_paths = batch['img_path']
    #         label_paths = batch['label_path']
    #         for img_tensor, label_path in zip(images, label_paths):
    #             # 读取标签
    #             with open(label_path, 'r') as file:
    #                 labels = file.readlines()
    #             labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的
    #
    #             # 转换为numpy并反归一化
    #             max_pixel = cfg_data['max_pixel']  # 从配置读取最大值（通常为255）
    #             img_np = img_tensor.numpy().transpose(1, 2, 0) * max_pixel
    #             img_np = img_np.astype(np.uint8)
    #
    #             # 转换颜色空间（根据实际存储格式调整）
    #             if dataset.img_read_method == 'gray':
    #                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    #             else:
    #                 img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #
    #             img_np = cv2.resize(img_np, (1920, 1080))
    #
    #             # 绘制边界框和标签
    #             h, w = img_np.shape[:2]
    #             for _label in labels:
    #                 label = [literal_eval(x) for x in _label]
    #                 # 解析归一化坐标
    #                 cx = label[1] * w
    #                 cy = label[2] * h
    #                 box_w = label[3] * w
    #                 box_h = label[4] * h
    #
    #                 # 计算边界坐标
    #                 x1 = int(cx - box_w / 2)
    #                 y1 = int(cy - box_h / 2)
    #                 x2 = int(cx + box_w / 2)
    #                 y2 = int(cy + box_h / 2)
    #
    #                 # 绘制矩形
    #                 cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #                 # 添加类别标签
    #                 text = f"123"
    #                 # cv2.putText(img_np, text, (x1 + 2, y1 + 20),
    #                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #
    #             # 显示结果
    #             cv2.imshow('Augmentation Demo', img_np)
    #             # cv2.imshow(f'{batch["img_path"]}', img_np)
    #             # cv2.waitKey()
    #             if cv2.waitKey(2000) == 27:
    #                 break
