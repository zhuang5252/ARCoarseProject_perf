import os
import cv2
import copy
import torch
import pickle
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import DataLoader
from air_track.detector.data.base_dataset.data_load_aot import load_datasets
from air_track.detector.engine.predictor import Predictor
from air_track.detector.data.base_dataset.gaussian_render import render_y
from air_track.aligner.utils.transform_utils import build_geom_transform
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import combine_images, cxcywhn2xyxy
from air_track.detector.utils.transform_utils import apply_transform
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_distance, show_confidence, \
    show_confidence_sorted
from air_track.utils import common_utils, reprod_init, img_fn, transform_imgs, check_and_change_img_size


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


class DetectDatasetAot(BaseDataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        super().__init__(stage, cfg_data)
        self.load_dataset()
        # if self.is_training:
        #     self.add_other_dataset(max_len=18000)
        # else:
        #     self.add_other_dataset(max_len=3000)

        # self.frame_step = self.dataset_params['frame_step']  # 1 frame_step图像帧之间的步长
        # self.input_frames = self.dataset_params['input_frames']  # 2 input_frames决定连续取几帧数据，为2时，取前后相邻两帧

        self.img_size_w, self.img_size_h = self.dataset_params['img_size']  # 指定图像尺寸
        self.img_read_method = self.dataset_params['img_read_method'].lower()
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
            if self.img_read_method == 'gray':
                prev_img = cv2.imread(prev_frame['img_path'], cv2.IMREAD_GRAYSCALE)
            elif self.img_read_method == 'unchanged':
                prev_img = cv2.imread(prev_frame['img_path'], cv2.IMREAD_UNCHANGED)
            else:
                prev_img = cv2.imread(prev_frame['img_path'])
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
            if 'image' in k:
                if len(res[k].shape) == 2:
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / self.cfg_data['max_pixel']).float().unsqueeze(
                        -1)
                else:
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / self.cfg_data['max_pixel']).float()

                if self.is_training and gamma_flag:
                    res[k] = torch.pow(res[k], gamma_aug)
                res[k] = res[k].permute(2, 0, 1)

            elif isinstance(res[k], np.ndarray):
                res[k] = torch.from_numpy(res[k].astype(np.float32)).float()

        return res

    def __getitem__(self, idx):
        index = self.frame_nums_with_objs[idx]
        cur_frame = self.frames[index]

        cur_frame['img_path'] = cur_frame['img_path'].replace('/mnt/zxc/AOT_Data', '/home/csz_changsha/data/Aot')

        # 路径需要满足所需层级
        if self.img_read_method == 'gray':
            cur_img = cv2.imread(cur_frame['img_path'], cv2.IMREAD_GRAYSCALE)
        elif self.img_read_method == 'unchanged':
            cur_img = cv2.imread(cur_frame['img_path'], cv2.IMREAD_UNCHANGED)
        else:
            cur_img = cv2.imread(cur_frame['img_path'])
        cur_img = check_and_change_img_size(cur_img, self.img_size_w, self.img_size_h)

        # 如果图像不存在，index + 1
        if cur_img is None:
            return self.__getitem__(idx + 1)

        '''标签及图像应用变换'''
        # 使用transform对prev_frames所有帧的detections进行转换
        cur_step_detections = cur_frame['detections']  # 当前帧的detections

        '''高斯、下采样，制作标签'''
        res = render_y(self.cfg_data, cur_step_detections, img_w=self.img_size_w,
                       img_h=self.img_size_h, down_scale=self.down_scale, target_min_size=self.target_min_size)

        res['part'] = cur_frame['part']
        res['flight_id'] = cur_frame['flight_id']
        res['img_name'] = cur_frame['img_name']
        res['cur_step_detections'] = cur_step_detections

        # res['idx'] = index
        res['image'] = cur_img

        # 是否返回torch_tensors
        if self.return_torch_tensors:
            # gamma转换和归一化
            res = self.gamma_and_normalize(res, cfg=self.dataset_params, gamma_flag=self.gamma_flag)

        return res


def process_input(cur_image, prev_images, device):
    """输入数据前处理"""
    # 将cur数据与prev数据合并
    input_data = combine_images(prev_images, cur_image)

    input_data = input_data.float().to(device)

    return input_data


def get_prev_step_frame(data, model_params):
    """
    从dataset中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    prev_images = []

    for i in range(model_params['input_frames'] - 1):
        prev_images.append(data[f'prev_image_aligned{i}'])

    return prev_images


def predict_with_gt(predictor):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg
    classes = cfg['classes']
    model_params = cfg['model_params']
    img_size_w, img_size_h = cfg['dataset_params']['img_size']
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_valid = DetectDatasetAot(stage=cfg['stage_valid'], cfg_data=cfg)
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
    candidate_targets = []
    distance_gts = []
    distance_label_x, distance_label_y = [], []
    scale_w, scale_h = 0, 0
    for i, data in enumerate(valid_dl):
        print(f"当前处理第{i + 1}/{len(valid_dl)}帧")

        cur_file = img_fn(cfg, data['part'][0], data['flight_id'][0], data['img_name'][0])
        if os.path.exists(cur_file) is False:
            continue

        cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = cur_img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        cur_img = data['image']

        # 数据前处理
        input_data = cur_img.to(device)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, inference_time = predictor.predict(input_data)
        inference_times.append(inference_time)

        cur_step_detections = data['cur_step_detections']
        if cur_step_detections:
            if len(cur_step_detections) != 1:
                print('len(cur_step_detections) != 1')
            detection = cur_step_detections[0]
            distance = detection['distance'].item()
            bbox_cxcywhn = [detection['cx'].item(), detection['cy'].item(), detection['w'].item(),
                            detection['h'].item()]
            bbox_xyxy = cxcywhn2xyxy(bbox_cxcywhn, img_w=orig_w, img_h=orig_h)
            distance_label_x.append(i)
            distance_label_y.append(distance)
        else:
            distance = np.nan
            bbox_xyxy = [np.nan, np.nan, np.nan, np.nan]

        bbox_xyxy = [1] + bbox_xyxy

        frame_nums.append(i)
        distance_gts.append(distance)
        bbox_gts.append([bbox_xyxy])
        candidate_targets.append(detected_objects)

        # 检测结果可视化
        if cfg['visualize_flag']:
            # 可视化结果保存
            visualize_and_save(cur_file, detected_objects, os.path.join((cfg['visualize_save_dir']), data['part'][0]),
                               scale_w=scale_w, scale_h=scale_h,
                               bbox_gts=[bbox_xyxy], distance=distance)

    if cfg['calculate_metrics_flag']:
        save_file_dir = cfg['metrics_cache_save_file']
        cache_file = f'{save_file_dir}/cache_metrics.pkl'
        with open(cache_file, 'wb') as file:
            pickle.dump({'bbox_gts': bbox_gts, 'distance_gts': distance_gts, 'candidate_targets': candidate_targets},
                        file)

        # 计算指标
        detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
        pos_candidate_targets, neg_candidate_targets = calculate_metrics(
            classes, bbox_gts, candidate_targets,
            frame_nums, scale_w, scale_h,
            threshold=cfg['threshold'],
            target_min_size=cfg['target_min_size'],
            iou=cfg['iou_flag'], cls_flag=cfg['cls_flag'])

        if cfg['analysis_show']:
            show_confidence(pos_conf_list, neg_conf_list, save_path=cfg['visualize_save_dir'],
                            file_name=f'conf.png')
            show_confidence_sorted(pos_conf_list, neg_conf_list, save_path=cfg['visualize_save_dir'],
                                   file_name=f'conf_sorted.png')


    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == "__main__":
    """有标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict_aot.yaml')
    cfg_predict = common_utils.combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    # 创建预测器
    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    predict_with_gt(predictor)
