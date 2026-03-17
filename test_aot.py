import os
import cv2
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from air_track.detector.data.dataset import DetectDataset
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import xywh2xyxy, check_boundary, cxcywhn2xyxy, combine_images
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_distance
from air_track.utils import reprod_init, combine_load_cfg_yaml, img_fn


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


def get_prev_step_frame(data, model_params):
    """
    从dataset中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    prev_images = []

    for i in range(model_params['input_frames'] - 1):
        prev_images.append(data[f'prev_image_aligned{i}'])

    return prev_images


def process_input(cur_image, prev_images, device):
    """输入数据前处理"""
    # 将cur数据与prev数据合并
    input_data = combine_images(prev_images, cur_image)

    input_data = input_data.float().to(device)

    return input_data


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
    temp = 0  # 用来跳过标签存在这一帧，但不存在图像数据的帧
    scale_w, scale_h = 0, 0
    for i, data in enumerate(valid_dl):
        print(f"当前处理第{i + 1}/{len(valid_dl)}帧")

        cur_file = img_fn(detect_cfg, data['part'][0], data['flight_id'][0], data['image_name'][0])

        cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        # 图像存在进行后续操作
        orig_h, orig_w = cur_img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        cur_img = data['image']

        # 读取prev帧图像数据(已经在dataset中进行了帧对齐)
        prev_images = get_prev_step_frame(data, model_params)

        # 数据前处理
        input_data = process_input(cur_img, prev_images, device)

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
        frame_nums.append(i)
        distance_gts.append(distance)
        bbox_gts.append(bbox_xyxy)
        candidate_targets.append(detected_objects)

        # 检测结果可视化
        if detect_cfg['visualize_flag']:
            # 可视化结果保存
            visualize_and_save(cur_file, detected_objects, os.path.join((detect_cfg['visualize_save_dir']), data['part'][0]),
                               scale_w=scale_w, scale_h=scale_h,
                               bbox_gt=bbox_xyxy, distance=distance)

    if detect_cfg['calculate_metrics_flag']:
        save_file_dir = detect_cfg['metrics_cache_save_file']
        cache_file = f'{save_file_dir}/cache_metrics.pkl'
        with open(cache_file, 'wb') as file:
            pickle.dump({'bbox_gts': bbox_gts, 'distance_gts': distance_gts, 'candidate_targets': candidate_targets},
                        file)

        detection_rate, distance_pred_x, distance_pred_y = \
            calculate_metrics_with_distance(bbox_gts, candidate_targets, frame_nums, scale_w, scale_h,
                                            threshold=detect_cfg['threshold'], target_min_size=detect_cfg['target_min_size'],
                                            iou=detect_cfg['iou_flag'])
        if detect_cfg['show_distance']:
            show_distance(distance_label_x, distance_label_y, distance_pred_x, distance_pred_y,
                          save_path=detect_cfg['visualize_save_dir'], file_name=f'distance.png')

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == "__main__":
    """全流程测试-帧对齐使用pkl缓存文件"""

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
