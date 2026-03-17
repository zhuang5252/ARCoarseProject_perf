import os
import cv2
import copy
import torch
import pickle
import random
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import DataLoader
from air_track.detector.data.data_load import load_datasets
from air_track.detector.data.base_dataset.gaussian_render import render_y
from air_track.aligner.utils.transform_utils import build_geom_transform
from air_track.detector.utils.detect_utils import xywh2xyxy, check_boundary, cxcywhn2xyxy, argmax2d
from air_track.detector.utils.transform_utils import apply_transform
from air_track.detector.visualization.visualize_and_save import show_single_scatter_plot
from air_track.utils import reprod_init, combine_load_cfg_yaml, common_utils, check_and_change_img_size, \
    transform_imgs


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
        elif stage == cfg_data['stage_valid']:
            self.is_training = False
            parts = cfg_data['part_val']
        else:
            self.is_training = False
            parts = cfg_data['part_test']

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

    def add_other_dataset(self, max_len=15000):
        random_indexes = random.sample(range(len(self.frames)),
                                       min(max_len, len(self.frames)) - len(self.frame_nums_with_objs))
        self.frame_nums_with_objs += random_indexes

    def __len__(self) -> int:
        return len(self.frame_nums_with_objs)


class DetectDataset(BaseDataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        super().__init__(stage, cfg_data)
        self.load_dataset()
        # if self.is_training:
        #    self.add_other_dataset(max_len=18000)
        # else:
        #    self.add_other_dataset(max_len=3000)

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

    def __getitem__(self, idx):
        # 当前帧
        same_frame_flag = False
        index = self.frame_nums_with_objs[idx]
        cur_frame = self.frames[index]

        # 判断index是否满足向前找input_frames帧
        if index - self.frame_step * (self.input_frames - 1) < 0:
            return self.__getitem__(idx + 1)

        # 获取前 input_frames - 1 帧
        prev_frames = self.get_prev_step_frame(cur_frame, index, self.frame_step, self.input_frames)

        # 没找到指定的前 input_frames - 1 帧
        if len(prev_frames) != self.input_frames - 1:
            if idx + 1 < len(self.frame_nums_with_objs):
                return self.__getitem__(idx + 1)
            else:
                prev_frames = [cur_frame]
                same_frame_flag = True

        cur_frame['img_path'] = cur_frame['img_path'].replace('/mnt/zxc/AOT_Data', '/home/csz_changsha/data/Aot')
        prev_frames[0]['img_path'] = prev_frames[0]['img_path'].replace('/mnt/zxc/AOT_Data',
                                                                        '/home/csz_changsha/data/Aot')

        # 路径需要满足所需层级
        cur_img = cv2.imread(cur_frame['img_path'], cv2.IMREAD_UNCHANGED)
        cur_img = check_and_change_img_size(cur_img, self.img_size_w, self.img_size_h)

        # 如果图像不存在，index + 1
        if cur_img is None:
            return self.__getitem__(idx + 1)

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

        res['part'] = cur_frame['part']
        res['flight_id'] = cur_frame['flight_id']
        res['image_name'] = cur_frame['img_name']
        res['cur_step_detections'] = cur_step_detections
        res['cur_frame'] = cur_frame
        res['prev_frames'] = prev_frames

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


def detector_out_2_classifier_input(cfg, detections, orig_img):
    """检测器的输出转换为分类器的输入"""
    dataset_params = cfg['dataset_params']

    img_size_w, img_size_h = dataset_params['img_size']
    orig_h, orig_w = orig_img.shape[-2:]

    img_list, label_list = [], []
    for detection in detections:
        # 以候选目标的中心为基准，裁剪出指定尺寸的特征图
        bbox_cxcywh = [detection['cx'].item(), detection['cy'].item(), img_size_w, img_size_h]
        bbox_xyxy = xywh2xyxy(bbox_cxcywh)

        # 检查边界条件并调整
        bbox_xyxy = check_boundary(bbox_xyxy, img_size_w, img_size_h, orig_w, orig_h)

        x1, y1, x2, y2 = bbox_xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = orig_img[0, :, y1:y2, x1:x2]

        img_list.append(img)

    return img_list


def create_align_predictor(predict_yaml, device='cuda:0'):
    """创建检测器的预测器"""
    from air_track.aligner.engine import Predictor

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = device
    predictor.set_model()

    return predictor


def create_detect_predictor(predict_yaml, device='cuda:0'):
    """创建检测器的预测器"""
    from air_track.detector.engine import Predictor

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = device
    predictor.set_model()

    return predictor


def create_classify_predictor(predict_yaml, device='cuda:0'):
    """创建二级分类器的预测器"""
    from air_track.classifier.engine import Predictor

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = device
    predictor.set_model()

    return predictor


def test(cfg_inference):
    """全流程测试"""
    device = cfg_inference['device']
    max_pixel = cfg_inference['dataset_params']['max_pixel']

    '''创建帧对齐预测器'''
    align_predictor = None
    if cfg_inference['use_frame_align']:
        align_predictor = create_align_predictor(cfg_inference['align_predictor_yaml'], device)
        align_predictor.max_pixel = max_pixel

    '''创建检测器'''
    detect_predictor = create_detect_predictor(cfg_inference['detect_predictor_yaml'], device)
    detect_predictor.max_pixel = max_pixel
    detect_cfg = detect_predictor.cfg

    '''创建分类器'''
    classify_cfg = None
    classify_predictor = None
    if cfg_inference['use_second_classify']:
        classify_predictor = create_classify_predictor(cfg_inference['classify_predictor_yaml'], device)
        classify_predictor.max_pixel = max_pixel
        classify_cfg = classify_predictor.cfg

    # 检测器参数
    frame_step = detect_predictor.dataset_params['frame_step']
    input_frames = detect_predictor.dataset_params['input_frames']
    model_params = detect_predictor.model_params
    img_size_w, img_size_h = detect_predictor.dataset_params['img_size']

    # 超参数设置为小于2帧或者步长不大于0，则返回空列表
    if input_frames < 2 or frame_step <= 0:
        assert ValueError('input_frames < 2 or frame_step <= 0. ')

    dataset_valid = DetectDataset(stage=detect_cfg['stage_valid'], cfg_data=detect_cfg)
    dataset_valid.frame_align_flag = False
    valid_dl = DataLoader(
        dataset_valid,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    inference_times = []
    bbox_gts = []
    frame_nums = []
    cur_files = []
    candidate_targets = []
    distance_gts = []
    distance_label_x, distance_label_y = [], []
    scale_w, scale_h = 0, 0
    pos_conf_list = []
    for i, data in enumerate(valid_dl):
        print(f"当前处理第{i + 1}/{len(valid_dl)}帧")

        cur_file = data['cur_frame']['img_path'][0]
        prev_file = data['prev_frames'][0]['img_path'][0]

        cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        prev_img = cv2.imread(prev_file, cv2.IMREAD_UNCHANGED)
        # 图像存在进行后续操作
        orig_h, orig_w = cur_img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h

        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        if cfg_inference['use_frame_align']:
            transform, inference_time = align_predictor.predict(prev_img, cur_img)
            prev_img_aligned = cv2.warpAffine(prev_img,
                                              transform[:2, :],
                                              dsize=(img_size_w, img_size_h),
                                              flags=cv2.INTER_LINEAR)
            prev_images = [prev_img_aligned]
        else:
            prev_images = [prev_img]

        # 数据前处理
        input_data = detect_predictor.process_input(cur_img, prev_images)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, detect_inference_time = detect_predictor.predict(input_data)

        '''二级分类器'''
        classify_inference_time = 0
        if cfg_inference['use_second_classify']:
            classify_predictor.normalize = False  # 检测模块已经做过归一化
            # 检出目标
            if detected_objects:
                classify_input_data = detector_out_2_classifier_input(classify_cfg, detected_objects, input_data)
                classify_input_data = classify_predictor.process_input(torch.stack(classify_input_data, dim=0))
                classify_prediction, classify_inference_time = classify_predictor.predict(classify_input_data)

                # 找出列表中元素是'pos'的索引
                pos_indices = [index for index, value in enumerate(classify_prediction)
                               if value == classify_cfg['classes'][1]]
                # 保留检测框中二级分类器为正样本的检测框
                detected_objects = [detected_objects[i] for i in pos_indices]
        '''二级分类器结束'''

        inference_times.append(detect_inference_time + classify_inference_time)

        '''计算指标并可视化'''
        cur_step_detections = data['cur_step_detections']
        if cur_step_detections:
            if len(cur_step_detections) != 1:
                print('len(cur_step_detections) != 1')
            prediction[-1] = torch.sigmoid(prediction[-1][0].float())
            mask_pred_feature = prediction[-1][0].float().detach().cpu()
            feature_h, feature_w = mask_pred_feature.shape[-2:]
            detection = cur_step_detections[0]
            bbox_cxcywhn = [detection['cx'].item(), detection['cy'].item(), detection['w'].item(),
                            detection['h'].item()]
            bbox_xyxy = cxcywhn2xyxy(bbox_cxcywhn, img_w=feature_w, img_h=feature_h)

            x1, y1, x2, y2 = bbox_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2 + 0.5), int(y2 + 0.5)

            bbox_feature = mask_pred_feature[y1:y2, x1:x2]
            y, x = argmax2d(bbox_feature)
            conf = bbox_feature[y, x]
            pos_conf_list.append(conf.item())
        else:
            pos_conf_list.append(np.nan)

        frame_nums.append(i)

    save_file_dir = detect_cfg['metrics_cache_save_file']
    cache_file = f'{save_file_dir}/cache_pos_conf.pkl'
    with open(cache_file, 'wb') as file:
        pickle.dump({'frame_nums': frame_nums, 'scale_w': scale_w, 'scale_h': scale_h, 'pos_conf_list': pos_conf_list},
                    file)

    show_single_scatter_plot(frame_nums, pos_conf_list, save_path=detect_cfg['visualize_save_dir'],
                             file_name='conf.png', title='Positive Confidence',
                             xlabel='Frame ID', ylabel='conf', label='Confidence',
                             color='green', s=3, sort=False)

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == "__main__":
    """测试每一帧的标签位置对应的模型输出特征图的最大置信度，并可视化概率分布图"""

    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_yaml = os.path.join(script_dir, 'air_track/config/inference.yaml')
    cfg_inference = combine_load_cfg_yaml(yaml_paths_list=[inference_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_inference['seed'])

    # 进行测试
    dataset_params = cfg_inference['dataset_params']
    data_dir = dataset_params['data_dir']
    test(cfg_inference)
