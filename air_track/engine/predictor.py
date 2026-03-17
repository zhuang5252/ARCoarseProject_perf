# -*- coding: utf-8 -*-
# @Author    : 
# @File      : predictor.py
# @Created   : 2025/7/1 下午5:04
# @Desc      : 整个AirTrack系统的预测器
import os
import cv2
import copy
import numpy as np
from air_track.detector.utils.detect_utils import xywh2xyxy, check_boundary
from air_track.utils import combine_load_cfg_yaml, check_image_bit_depth, check_and_change_img_size, \
    round_up_to_nearest_power_of_two


def create_align_predictor(predict_yaml, model_path=None, device='cuda:0'):
    """创建检测器的预测器"""
    from air_track.aligner.engine import Predictor

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = device
    predictor.set_model(model_path)

    return predictor


def create_detect_predictor(predict_yaml, model_path=None, device='cuda:0'):
    """创建检测器的预测器"""
    from air_track.detector.engine import Predictor

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = device
    predictor.set_model(model_path)

    return predictor


def create_classify_predictor(predict_yaml, model_path=None, device='cuda:0'):
    """创建二级分类器的预测器"""
    from air_track.classifier.engine import Predictor

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = device
    predictor.set_model(model_path)

    return predictor


def detector_out_2_classifier_input(cfg, detections, orig_img, scale_w, scale_h):
    """检测器的输出转换为分类器的输入"""
    orig_img = copy.deepcopy(orig_img)
    dataset_params = cfg['dataset_params']
    input_size_w, input_size_h = dataset_params['img_size']

    # 异常图像
    if len(orig_img.shape) != 2 and len(orig_img.shape) != 3:
        raise ValueError('Image shape is not correct!')

    # 将图像从HWC转为CHW
    if len(orig_img.shape) == 3:
        orig_img = np.transpose(orig_img, (2, 0, 1))
    orig_h, orig_w = orig_img.shape[-2:]

    img_list, label_list = [], []
    for detection in detections:
        # 以候选目标的中心为基准，裁剪出指定尺寸的特征图
        # 确定长边
        max_len = max(detection['w'] / scale_w, detection['h'] / scale_h)
        # 这里一级检测器的输出目标框已经规定了最小输出尺寸
        img_size_w = int(max_len * 2)
        img_size_h = int(max_len * 2)
        # 向上取整到最近的2的倍数
        img_size_w = round_up_to_nearest_power_of_two(img_size_w)
        img_size_h = round_up_to_nearest_power_of_two(img_size_h)

        bbox_cxcywh = [detection['cx'] / scale_w, detection['cy'] / scale_h, img_size_w, img_size_h]
        bbox_xyxy = xywh2xyxy(bbox_cxcywh)

        # 检查边界条件并调整
        bbox_xyxy = check_boundary(bbox_xyxy, img_size_w, img_size_h, orig_w, orig_h)

        x1, y1, x2, y2 = bbox_xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 灰度图像
        if len(orig_img.shape) == 2:
            img = orig_img[y1:y2, x1:x2]
            # img = np.expand_dims(img, axis=0)
        # 非灰度图像
        else:
            img = orig_img[y1:y2, x1:x2, :]

        # 校验并调整图像尺寸
        img = check_and_change_img_size(img, img_size_w=input_size_w, img_size_h=input_size_h)
        # 灰度图像同样需要为3维
        if len(orig_img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        img_list.append(img)

    return np.stack(img_list, axis=0)


class Predictor:
    def __init__(self, yaml_list, args: dict = None):
        # 合并若干个yaml的配置文件内容
        self.args = args
        self.cfg_inference = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

        # 提供补充和可覆盖配置项 TODO 若嵌套的配置项较多，后续按需修改
        if self.args is not None:
            for k, v in self.args.items():
                self.cfg_inference[k] = v

        self.device = self.cfg_inference['device']
        self.classes = self.cfg_inference['dataset_params']['classes']
        self.img_read_method = self.cfg_inference['img_read_method'].lower()
        self.max_pixel = self.cfg_inference['dataset_params']['max_pixel']

        self.align_model_path = self.cfg_inference.get('align_model_path', None)
        self.detect_model_path = self.cfg_inference.get('detect_model_path', None)
        self.classify_model_path = self.cfg_inference.get('classify_model_path', None)

        self.align_predictor = None
        self.detect_predictor = None
        self.classify_predictor = None

        self.detect_cfg = None
        self.classify_cfg = None

    def set_predictor(self, align_model_path=None, detect_model_path=None, classify_model_path=None):
        """创建子模块的预测器"""

        # 创建帧对齐预测器
        if self.cfg_inference['use_frame_align']:
            self.align_predictor = create_align_predictor(self.cfg_inference['align_predictor_yaml'],
                                                          align_model_path, self.device)
            self.align_predictor.max_pixel = self.max_pixel

        # 创建检测器
        self.detect_predictor = create_detect_predictor(self.cfg_inference['detect_predictor_yaml'],
                                                        detect_model_path, self.device)
        self.detect_predictor.max_pixel = self.max_pixel
        self.detect_cfg = self.detect_predictor.cfg

        # 创建分类器
        if self.cfg_inference['use_second_classify']:
            self.classify_predictor = create_classify_predictor(self.cfg_inference['classify_predictor_yaml'],
                                                                classify_model_path, self.device)
            self.classify_predictor.max_pixel = self.max_pixel
            self.classify_cfg = self.classify_predictor.cfg

    def get_prev_step_images(self,
                             path, file_names, index,
                             img_size_w=None, img_size_h=None,
                             frame_step=None, input_frames=None,
                             max_pixel=None
                             ):
        """
        从file_names中取出prev的数据

        存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
        """
        if not max_pixel:
            max_pixel = self.max_pixel
        if not frame_step:
            frame_step = self.detect_predictor.dataset_params['frame_step']
        if not input_frames:
            input_frames = self.detect_predictor.dataset_params['input_frames']
        if not img_size_w and not img_size_h:
            img_size_w, img_size_h = self.detect_predictor.dataset_params['img_size']

        prev_images = []
        for i in range(frame_step, input_frames, frame_step):
            interval_index = i * frame_step

            file_name = file_names[index - interval_index]
            prev_img_path = os.path.join(path, file_name)

            if self.img_read_method == 'gray':
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
            elif self.img_read_method == 'unchanged':
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_UNCHANGED)
            else:
                prev_img = cv2.imread(prev_img_path)

            # 校验图像深度与配置中的max_pixel是否一致
            check_image_bit_depth(prev_img, prev_img_path, max_pixel)
            # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
            prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

            prev_images.append(prev_img)

        return prev_images

    def aligner_predict(self, prev_images, cur_img, img_size_w, img_size_h):
        """帧对齐预测"""
        if len(prev_images) > 1:
            raise ValueError('目前尚不支持多个前一帧图像对齐. ')
        prev_img = prev_images[0]
        item_dx_dy_angle, transform, align_inference_time = self.align_predictor.predict(prev_img, cur_img)
        prev_img_aligned = cv2.warpAffine(prev_img,
                                          transform[:2, :],
                                          dsize=(img_size_w, img_size_h),
                                          flags=cv2.INTER_LINEAR)
        prev_images = [prev_img_aligned]

        return prev_images, item_dx_dy_angle, transform, align_inference_time

    def detector_predict(self, prev_images, cur_img):
        """一级检测器预测"""
        # 数据前处理
        input_data = self.detect_predictor.process_multi_frame_input(prev_images, cur_img)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, detect_inference_time = self.detect_predictor.predict(input_data)

        return prediction, detected_objects, detect_inference_time

    def classifier_predict(self, detected_objects, cur_img_orig, scale_w, scale_h):
        """二级分类器预测"""
        final_detected_objects = []
        # 检出目标
        if detected_objects:
            classify_input_data = detector_out_2_classifier_input(
                self.classify_cfg, detected_objects, cur_img_orig, scale_w, scale_h
            )
            classify_input_data = self.classify_predictor.process_input(classify_input_data)
            classify_prediction, cls_name_batch, classify_inference_time = self.classify_predictor.predict(
                classify_input_data)

            # for idx, item in enumerate(classify_input_data):
            #     temp = item.permute(1, 2, 0).detach().cpu().numpy() * 255
            #     cls = cls_name_batch[idx]
            #     cv2.imwrite(f'/home/csz_changsha/tmp/second/{idx}_{cls}.jpg', temp.astype(np.uint8))

            if self.cfg_inference['second_classify_method'] == 'filter_conf':
                '''使用置信度相乘方式筛选正样本'''
                detector_conf_list = [value['conf'] for value in detected_objects]
                classify_conf_list = classify_prediction[:, 1].tolist()
                for i, (detector_conf, classify_conf) in enumerate(zip(detector_conf_list, classify_conf_list)):
                    final_conf = detector_conf * classify_conf
                    if final_conf > self.cfg_inference['conf_threshold']:
                        detected_objects[i]['conf'] = final_conf
                        final_detected_objects.append(detected_objects[i])
            else:
                '''单纯使用二级分类器输出筛选正样本'''
                # 找出列表中元素是'pos'的索引
                pos_indices = [index for index, value in enumerate(cls_name_batch)
                               if value == self.classify_cfg['classes'][1]]

                # 保留检测框中二级分类器为正样本的检测框
                final_detected_objects = [detected_objects[i] for i in pos_indices]
        else:
            final_detected_objects = detected_objects
            classify_inference_time = 0

        return final_detected_objects, classify_inference_time


if __name__ == '__main__':
    pass
