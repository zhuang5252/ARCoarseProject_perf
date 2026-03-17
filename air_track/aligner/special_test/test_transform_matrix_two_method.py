import os
import cv2
import torch
import timeit
import numpy as np
from torch import nn
from air_track.aligner.model import resnet34_orig as resnet34
from air_track.aligner.utils.transform_utils import create_points, build_geom_transform_predict
from air_track.aligner.utils.offset_grid_to_transform import offset_grid_to_transform, offset_grid_to_transform_params
from air_track.utils import combine_load_cfg_yaml, numpy_2_tensor, load_model, check_and_change_img_size, reprod_init


class Predictor:
    def __init__(self, yaml_list: list):
        # 合并若干个yaml的配置文件内容
        self.cfg = combine_load_cfg_yaml(yaml_paths_list=yaml_list)
        self.model_path = self.cfg['model_path']
        self.dataset_params = self.cfg['dataset_params']
        self.img_size_w, self.img_size_h = self.dataset_params['img_size']
        self.max_pixel = self.cfg['max_pixel']
        self.device = self.cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.prev_points, self.prev_points_1d = create_points(eval(self.cfg['points_shape']),
                                                              crop_w=self.img_size_w, crop_h=self.img_size_h)
        self.prev_points = np.expand_dims(self.prev_points[..., 2:-2, 2:-2], axis=0)

    def set_model(self):
        """定义并load模型"""
        print('The model path is: ', self.model_path)
        model_params = self.cfg['model_params']
        self.model: nn.Module = resnet34.__dict__[model_params['model_cls']](cfg=model_params, pretrained=False)
        self.model = load_model(self.model, self.model_path, device=self.device)

        return self.model

    def process_input(self, prev_img, cur_img):
        """输入数据前处理"""
        cur_img = check_and_change_img_size(cur_img, self.img_size_w, self.img_size_h)
        prev_img = check_and_change_img_size(prev_img, self.img_size_w, self.img_size_h)

        if len(cur_img.shape) == 2 and len(prev_img.shape) == 2:
            cur_tensor_img = numpy_2_tensor(cur_img, self.max_pixel).unsqueeze(0).unsqueeze(0).to(self.device)
            prev_tensor_img = numpy_2_tensor(prev_img, self.max_pixel).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            cur_tensor_img = numpy_2_tensor(cur_img, self.max_pixel).permute(2, 0, 1).unsqueeze(0).to(self.device)
            prev_tensor_img = numpy_2_tensor(prev_img, self.max_pixel).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return prev_tensor_img, cur_tensor_img

    def process_output(self, offsets, heatmap):
        """输出数据后处理"""
        cur_points_pred = self.prev_points + offsets.detach().cpu().numpy()
        transform_1, _ = offset_grid_to_transform(
            prev_frame_points=self.prev_points.reshape(2, -1),
            cur_frame_points=cur_points_pred.reshape(2, -1),
            points_weight=heatmap[0].detach().cpu().numpy().reshape(-1) ** 2
        )
        dx, dy, angle, err = offset_grid_to_transform_params(
            prev_frame_points=self.prev_points.reshape(2, -1),
            cur_frame_points=cur_points_pred.reshape(2, -1),
            points_weight=heatmap[0].detach().cpu().numpy().reshape(-1) ** 2
        )
        transform_2 = build_geom_transform_predict(translation_x=dx, translation_y=dy,
                                                   scale_x=1.0, scale_y=1.0,
                                                   angle=angle,
                                                   return_params=True)

        return transform_1, transform_2

    def predict(self, prev_img, cur_img):
        """模型预测"""
        self.model.eval()
        # 输入数据前处理
        prev_frame, cur_frame = self.process_input(prev_img, cur_img)

        start_time = timeit.default_timer()
        with torch.no_grad():
            heatmap, offsets = self.model(prev_frame, cur_frame)

        end_time = timeit.default_timer()
        # 推理耗时
        inference_time = (end_time - start_time) * 1000

        # 输出数据后处理
        transform_1, transform_2 = self.process_output(offsets, heatmap)

        return transform_1, transform_2, inference_time


def predict(predictor, data_dir):
    sigma_scale = 0.0025  # 图像缩放参数
    sigma_angle = 0.2  # 图像旋转参数
    sigma_offset = 12.0  # 图像位移参数
    cfg = predictor.cfg
    img_read_method = cfg['dataset_params']['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']
    save_path = cfg['visualize_save_dir']

    img_paths = [f for f in os.listdir(data_dir) if f.endswith(cfg['img_format'])]
    img_paths.sort()

    frame_num = len(img_paths)
    inference_times = []
    loss_1_list, loss_2_list = [], []
    loss_1_img_means, loss_2_img_means = [], []
    loss_1_img_maxes, loss_2_img_maxes = [], []
    loss_1_img_mins, loss_2_img_mins = [], []
    for idx, file_name in enumerate(img_paths):
        # if idx > 99:
        #     break
        print(f'当前处理第{idx + 1}/{frame_num}帧')

        # 生成随机变化参数
        scale = np.exp(np.random.normal(0, sigma_scale * 2))
        angle = np.random.normal(0, sigma_angle * 2)
        dx = np.random.normal(0, sigma_offset * 2)
        dy = np.random.normal(0, sigma_offset * 2)
        # scale = np.exp(sigma_scale * 2)
        # angle = sigma_angle * 2
        # dx = sigma_offset * 2
        # dy = sigma_offset * 2
        tr = build_geom_transform_predict(translation_x=dx, translation_y=dy,
                                          scale_x=scale, scale_y=scale,
                                          angle=angle,
                                          return_params=True)

        prev_img_file = os.path.join(data_dir, file_name)

        if img_read_method == 'gray':
            prev_img = cv2.imread(prev_img_file, cv2.IMREAD_GRAYSCALE)
            predictor.max_pixel = 255
        elif img_read_method == 'unchanged':
            prev_img = cv2.imread(prev_img_file, cv2.IMREAD_UNCHANGED)
        else:
            prev_img = cv2.imread(prev_img_file)

        orig_h, orig_w = prev_img.shape[:2]
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        # if (orig_h, orig_w) != (512, 640):
        #     prev_img = transform_img_size(prev_img, w=img_size_w, h=img_size_h)

        img_h, img_w = prev_img.shape[:2]
        cur_img = cv2.warpAffine(prev_img,
                                 tr[:2, :],
                                 dsize=(img_w, img_h),
                                 flags=cv2.INTER_LINEAR)

        transform_1, transform_2, inference_time = predictor.predict(prev_img, cur_img)
        inference_times.append(inference_time)

        cur_img_1 = cv2.warpAffine(prev_img,
                                   transform_1[:2, :],
                                   dsize=(img_w, img_h),
                                   flags=cv2.INTER_LINEAR)
        cur_img_2 = cv2.warpAffine(prev_img,
                                   transform_2[:2, :],
                                   dsize=(img_w, img_h),
                                   flags=cv2.INTER_LINEAR)

        if cfg['visualize_flag']:
            save_path_1 = os.path.join(save_path, 'cur_img_1')
            save_path_2 = os.path.join(save_path, 'cur_img_2')
            os.makedirs(save_path_1, exist_ok=True)
            os.makedirs(save_path_2, exist_ok=True)
            merge_result = np.stack([cur_img, cur_img_1, cur_img], axis=2)
            cv2.imwrite(os.path.join(save_path_1, f'{idx}_merge_result.png'), merge_result)
            merge_result = np.stack([cur_img, cur_img_2, cur_img], axis=2)
            cv2.imwrite(os.path.join(save_path_2, f'{idx}_merge_result.png'), merge_result)

        loss_1 = np.abs(tr[:2] - transform_1[:2])
        loss_2 = np.abs(tr[:2] - transform_2[:2])
        loss_1_list.append(loss_1)
        loss_2_list.append(loss_2)

        cur_img_1 = cur_img_1.astype(np.int16)
        cur_img_2 = cur_img_2.astype(np.int16)
        prev_img = prev_img.astype(np.int16)

        temp = 50
        loss_1_img = np.abs(
            cur_img_1[temp:img_h - temp, temp:img_w - temp] - prev_img[temp:img_h - temp, temp:img_w - temp])
        loss_2_img = np.abs(
            cur_img_2[temp:img_h - temp, temp:img_w - temp] - prev_img[temp:img_h - temp, temp:img_w - temp])
        loss_1_img_mean = np.mean(loss_1_img)
        loss_2_img_mean = np.mean(loss_2_img)
        loss_1_img_max = np.max(loss_1_img)
        loss_2_img_max = np.max(loss_2_img)
        loss_1_img_min = np.min(loss_1_img)
        loss_2_img_min = np.min(loss_2_img)
        loss_1_img_means.append(loss_1_img_mean)
        loss_2_img_means.append(loss_2_img_mean)
        loss_1_img_maxes.append(loss_1_img_max)
        loss_2_img_maxes.append(loss_2_img_max)
        loss_1_img_mins.append(loss_1_img_min)
        loss_2_img_mins.append(loss_2_img_min)

    loss_1_mean = np.mean(np.array(loss_1_list))
    loss_1_max = np.max(np.array(loss_1_list))
    loss_1_min = np.min(np.array(loss_1_list))
    loss_2_mean = np.mean(np.array(loss_2_list))
    loss_2_max = np.max(np.array(loss_2_list))
    loss_2_min = np.min(np.array(loss_2_list))
    print(f'loss_1: loss_1_mean: {loss_1_mean}, loss_1_max: {loss_1_max}, loss_1_min: {loss_1_min}')
    print(f'loss_2: loss_2_mean: {loss_2_mean}, loss_2_max: {loss_2_max}, loss_2_min: {loss_2_min}')

    loss_1_img_mean = np.mean(np.array(loss_1_img_means))
    loss_1_img_max = np.max(np.array(loss_1_img_maxes))
    loss_1_img_min = np.min(np.array(loss_1_img_mins))
    loss_2_img_mean = np.mean(np.array(loss_2_img_means))
    loss_2_img_max = np.max(np.array(loss_2_img_maxes))
    loss_2_img_min = np.min(np.array(loss_2_img_mins))
    print(f'loss_1_img_mean: {loss_1_img_mean}, loss_1_img_max: {loss_1_img_max}, loss_1_img_min: {loss_1_img_min}')
    print(f'loss_2_img_mean: {loss_2_img_mean}, loss_2_img_max: {loss_2_img_max}, loss_2_img_min: {loss_2_img_min}')


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    predict(predictor, cfg_predict['test_data_dir'])
