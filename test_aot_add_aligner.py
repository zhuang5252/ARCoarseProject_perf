import os
import cv2
import copy
import pickle
import random
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from air_track.engine.predictor import Predictor
from air_track.detector.data.base_dataset.data_load_aot import load_datasets
from air_track.detector.data.base_dataset.base_dataset import BaseDataset
from air_track.detector.data.base_dataset.gaussian_render import render_y
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import cxcywhn2xyxy
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_distance, show_confidence, \
    show_confidence_sorted
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size, \
    transform_imgs, check_image_bit_depth, copy_split_data


class DetectDatasetAot(BaseDataset):
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
        # 随机补充数据集
        # if self.is_training:
        #     self.add_other_dataset(max_len=18000)
        # else:
        #     self.add_other_dataset(max_len=3000)

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
        return len(self.frame_nums_with_objs)

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

    def add_other_dataset(self, max_len=15000):
        # 获取所有可能的候选索引（排除已存在的索引）
        candidate_indexes = set(range(len(self.frames))) - set(self.frame_nums_with_objs)

        # 计算实际需要补充的数量（不超过剩余候选数量和max_len限制）
        num_to_add = min(max_len, len(candidate_indexes))

        # 随机选取不重复的索引
        random_indexes = random.sample(candidate_indexes, num_to_add)

        # 添加到现有列表
        self.frame_nums_with_objs.extend(random_indexes)

    def __getitem__(self, idx):
        # 多帧情况判断index是否满足向前找input_frames帧
        if self.input_frames > 1:
            if idx - self.frame_step * (self.input_frames - 1) < 0:
                return self.__getitem__(idx + 1)

        # 获取当前帧
        index = self.frame_nums_with_objs[idx]
        same_frame_flag = False
        cur_frame = dict(self.frames[index])

        cur_frame['img_path'] = cur_frame['img_path'].replace('/mnt/zxc/AOT_Data', '/home/csz_changsha/data/Aot')

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

        prev_frames[0]['img_path'] = prev_frames[0]['img_path'].replace('/mnt/zxc/AOT_Data', '/home/csz_changsha/data/Aot')

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
        res['cur_image_orig'] = cur_image_orig
        res['cur_img_path'] = cur_img_path
        res['cur_img_name'] = cur_img_name
        res['cur_labels'] = [cur_step_detections]

        if self.input_frames > 1:
            res['prev_img_paths'] = prev_img_paths
            # res['prev_labels'] = [prev_step_detections]
            # 取出前几帧的图像
            for i, prev_img in enumerate(prev_images):
                res[f'prev_image_aligned{i}'] = prev_img

        # admix特殊处理，同时要求必须是三通道图像
        if self.img_read_method == 'admix' and cur_image.shape[2] == 3:
            # 在res中增加cur_image_rgb、cur_image_gray、可选cur_image_filter
            res = self.add_cur_frame_rgb_gray(cur_image, cur_img_path, seq, res)
            # 在res中增加prev_image_rgb{i}、prev_image_gray{i}、可选prev_image_filter{i}
            res = self.add_prev_frames_rgb_gray(prev_images, prev_img_paths, seq, res)

        # 是否返回torch_tensors
        if self.return_torch_tensors:
            # gamma转换和归一化
            res = self.gamma_and_normalize(res, cfg=self.dataset_params, gamma_flag=self.gamma_flag)

        return res


def test(inference_yaml):
    """全流程测试"""
    # 创建系统预测器
    if isinstance(inference_yaml, list):
        predictor = Predictor(inference_yaml)
    elif isinstance(inference_yaml, str):
        predictor = Predictor([inference_yaml])
    else:
        raise ValueError('inference_yaml must be list or str.')

    '''创建子模块的预测器'''
    predictor.set_predictor(predictor.align_model_path, predictor.detect_model_path, predictor.classify_model_path)
    # 获取一级检测器的参数配置
    detect_cfg = predictor.detect_cfg

    '''必要参数设置'''
    max_pixel = predictor.max_pixel
    img_read_method = predictor.img_read_method
    frame_step = predictor.detect_predictor.dataset_params['frame_step']
    input_frames = predictor.detect_predictor.dataset_params['input_frames']
    img_size_w, img_size_h = predictor.detect_predictor.dataset_params['img_size']

    # 超参数设置为小于1帧或者步长不大于0，则抛出异常
    if input_frames < 1 or frame_step < 0:
        assert ValueError('input_frames < 1 or frame_step < 0. ')

    dataset_valid = DetectDatasetAot(stage=detect_cfg['stage_valid'], cfg_data=detect_cfg)
    dataset_valid.frame_align_flag = False
    valid_dl = DataLoader(
        dataset_valid,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    detector_counts = 0
    final_counts = 0
    inference_times = []
    img_paths = []
    bbox_xyxy = []
    bbox_cxcywhn = []
    frame_nums = []
    candidate_targets = []
    distance_gts = []
    distance_label_x, distance_label_y = [], []
    temp = 0  # 用来跳过标签存在这一帧，但不存在图像数据的帧
    scale_w, scale_h = 0, 0
    for i, data in enumerate(valid_dl):
        print(f"当前处理第{i + 1}/{len(valid_dl)}帧")

        part = data['part'][0]
        flight_id = data['flight_id'][0]
        cur_file = data['cur_frame']['img_path'][0]

        if img_read_method == 'gray':
            cur_img_orig = cv2.imread(cur_file, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            cur_img_orig = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        else:
            cur_img_orig = cv2.imread(cur_file)

        # 图像存在进行后续操作
        orig_h, orig_w = cur_img_orig.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h

        # 校验图像深度与配置中的max_pixel是否一致
        check_image_bit_depth(cur_img_orig, cur_file, max_pixel)
        # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
        cur_img = check_and_change_img_size(cur_img_orig, img_size_w, img_size_h)

        if input_frames > 1:
            prev_file = data['prev_img_paths'][0][0]
            if img_read_method == 'gray':
                prev_img_orig = cv2.imread(prev_file, cv2.IMREAD_GRAYSCALE)
            elif img_read_method == 'unchanged':
                prev_img_orig = cv2.imread(prev_file, cv2.IMREAD_UNCHANGED)
            else:
                prev_img_orig = cv2.imread(prev_file)

            # 校验图像深度与配置中的max_pixel是否一致
            check_image_bit_depth(prev_img_orig, prev_file, max_pixel)
            # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
            prev_img = check_and_change_img_size(prev_img_orig, img_size_w, img_size_h)
            prev_images = [prev_img]
        else:
            prev_images = []

        '''执行帧对齐'''
        align_inference_time = 0
        if predictor.cfg_inference['use_frame_align']:
            prev_images, item_dx_dy_angle, transform, align_inference_time = predictor.aligner_predict(
                prev_images, cur_img, img_size_w, img_size_h)

        '''执行一级检测'''
        prediction, detected_objects, detect_inference_time = predictor.detector_predict(prev_images, cur_img)
        detector_counts += len(detected_objects)

        '''执行二级分类'''
        if predictor.cfg_inference['use_second_classify']:
            final_detected_objects, classify_inference_time = predictor.classifier_predict(
                detected_objects, cur_img_orig, scale_w, scale_h)
        else:
            # 使用inference.yaml中的conf_threshold进行最终筛选
            final_detected_objects = []
            for detected_object in detected_objects:
                if detected_object['conf'] > predictor.cfg_inference['conf_threshold']:
                    final_detected_objects.append(detected_object)
            # final_detected_objects = detected_objects
            classify_inference_time = 0
        final_counts += len(final_detected_objects)

        inference_times.append(align_inference_time + detect_inference_time + classify_inference_time)

        '''计算指标并可视化'''
        cur_step_detections = data['cur_step_detections']
        if cur_step_detections and detect_cfg['calculate_metrics_flag']:
            if len(cur_step_detections) != 1:
                print('len(cur_step_detections) != 1')
            detection = cur_step_detections[0]
            distance = detection['distance'].item()
            bbox_cxcywhn = [1, detection['cx'].item(), detection['cy'].item(), detection['w'].item(),
                            detection['h'].item()]
            frame_bbox_xyxy = cxcywhn2xyxy(bbox_cxcywhn, img_w=orig_w, img_h=orig_h)
            frame_bbox_xyxy = [1] + frame_bbox_xyxy
            distance_label_x.append(i)
            distance_label_y.append(distance)
        else:
            distance = np.nan
            frame_bbox_xyxy = [1, np.nan, np.nan, np.nan, np.nan]

        frame_nums.append(i)
        img_paths.append(cur_file)
        distance_gts.append(distance)
        bbox_xyxy.append(frame_bbox_xyxy)
        candidate_targets.append(final_detected_objects)

        # 检测结果可视化
        if detect_cfg['visualize_flag']:
            # 可视化结果保存
            visualize_and_save(cur_file, final_detected_objects,
                               os.path.join((detect_cfg['visualize_save_dir']), part),
                               scale_w, scale_h, frame_bbox_xyxy, distance)

    if detect_cfg['calculate_metrics_flag']:
        save_file_dir = detect_cfg['metrics_cache_save_file']
        cache_file = f'{save_file_dir}/cache_aot.pkl'
        with open(cache_file, 'wb') as file:
            pickle.dump({
                'img_paths': img_paths,
                'frame_nums': frame_nums,
                'bbox_xyxy': bbox_xyxy,
                'distance_gts': distance_gts,
                'candidate_targets': candidate_targets,
                'scale_w': scale_w,
                'scale_h': scale_h
            }, file)

        detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
        pos_candidate_targets, neg_candidate_targets = calculate_metrics(
            predictor.classes, bbox_xyxy, candidate_targets, frame_nums, scale_w, scale_h,
            threshold=detect_cfg['threshold'],
            target_min_size=detect_cfg['target_min_size'],
            iou=detect_cfg['iou_flag'], cls_flag=detect_cfg['cls_flag'])
        if detect_cfg['analysis_show']:
            show_confidence(pos_conf_list, neg_conf_list, save_path=detect_cfg['visualize_save_dir'],
                            file_name=f'conf_aot.png')
            show_confidence_sorted(pos_conf_list, neg_conf_list,
                                   save_path=detect_cfg['visualize_save_dir'],
                                   file_name=f'conf_aot_sorted.png')
            if len(distance_label_y) > 0:
                show_distance(distance_label_x, distance_label_y, pos_pred_x, distance_pred_y,
                              save_path=detect_cfg['visualize_save_dir'], file_name=f'distance_aot.png')

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')
    print('一级检测器总检出数量: ', detector_counts)
    print('最终总检出数量: ', final_counts)


if __name__ == "__main__":
    """全流程测试-帧对齐模型实时推理"""

    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_yaml = os.path.join(script_dir, 'air_track/config/inference.yaml')
    cfg_inference = combine_load_cfg_yaml(yaml_paths_list=[inference_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_inference['seed'])

    # 进行测试
    # dataset_params = cfg_inference['dataset_params']
    # data_dir = dataset_params['data_dir']
    test(inference_yaml)
