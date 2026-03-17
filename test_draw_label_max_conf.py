import os
import cv2
import torch
import pickle
import numpy as np
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import xywh2xyxy, check_boundary, cxcywhn2xyxy, argmax2d
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_confidence, \
    show_single_scatter_plot
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size


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


def get_prev_step_frame(dataset_params, path, file_names, index):
    """
    从file_names中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    frame_step = dataset_params['frame_step']
    input_frames = dataset_params['input_frames']
    img_size_w, img_size_h = dataset_params['img_size']
    prev_images = []

    for i in range(frame_step, input_frames, frame_step):
        interval_index = i * frame_step

        file_name = file_names[index - interval_index]
        file = os.path.join(path, file_name)

        prev_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        prev_images.append(prev_img)

    return prev_images


def test(cfg_inference, img_folder, label_folder):
    """全流程测试"""
    device = cfg_inference['device']
    use_gray = cfg_inference['use_gray']
    max_pixel = cfg_inference['dataset_params']['max_pixel']
    dataset_params = cfg_inference['dataset_params']

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

    img_size_w, img_size_h = detect_predictor.dataset_params['img_size']

    img_names = os.listdir(img_folder)
    img_names.sort()

    inference_times = []
    frame_nums = []
    pos_conf_list = []
    scale_w, scale_h = 0, 0
    for i, img_name in enumerate(img_names):
        print(f"当前处理第{i}/{len(img_names)}帧")

        img_path = os.path.join(img_folder, img_name)
        label_path = os.path.join(label_folder,
                                  img_name.replace(dataset_params['img_format'], dataset_params['gt_format']))

        if use_gray:
            cur_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            cur_img = cv2.imread(img_path)

        # 若当前图片不存在，则使用temp使得cur_img读取 i + frame_step * input_frames 帧
        if cur_img is None:
            continue

        # 图像存在才进行后续操作
        orig_h, orig_w = cur_img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

        if cfg_inference['use_frame_align']:
            prev_images = [cur_img]  # TODO 目前检测模块不具备输入多帧图像
            if len(prev_images) > 1:
                raise ValueError('目前尚不支持多个前一帧图像对齐. ')
            prev_img = prev_images[0]
            transform, inference_time = align_predictor.predict(prev_img, cur_img)
            prev_img_aligned = cv2.warpAffine(prev_img,
                                              transform[:2, :],
                                              dsize=(img_size_w, img_size_h),
                                              flags=cv2.INTER_LINEAR)
            prev_images = [prev_img_aligned]

        # 数据前处理
        input_data = detect_predictor.process_input(cur_img)

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
                classify_prediction, cls_name_batch, classify_inference_time = classify_predictor.predict(classify_input_data)

                # 找出列表中元素是'pos'的索引
                pos_indices = [index for index, value in enumerate(cls_name_batch)
                               if value == classify_cfg['classes'][1]]
                # 保留检测框中二级分类器为正样本的检测框
                detected_objects = [detected_objects[i] for i in pos_indices]
        '''二级分类器结束'''

        inference_times.append(detect_inference_time + classify_inference_time)

        '''计算指标并可视化'''
        # 读取标签
        with open(label_path, 'r') as file:
            labels = file.readlines()
        labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的
        # 将标签由字符串转换为浮点数
        labels = [[float(item) for item in label] for label in labels]

        mask_pred_feature = prediction[0].float().detach().cpu()
        mask_pred_feature = mask_pred_feature[-1][0]
        feature_h, feature_w = mask_pred_feature.shape[-2:]

        # 将标签从cxcywhn转换为xyxy
        frame_bbox_xyxy, frame_bbox_cxcywhn = [], []
        if labels:
            for label in labels:
                bbox_xyxy = cxcywhn2xyxy(label[1:], img_w=feature_w, img_h=feature_h)
                frame_bbox_xyxy.append([int(label[0])] + bbox_xyxy)
                frame_bbox_cxcywhn.append(label)

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
    """全流程测试"""

    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_yaml = os.path.join(script_dir, 'air_track/config/inference.yaml')
    cfg_inference = combine_load_cfg_yaml(yaml_paths_list=[inference_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_inference['seed'])

    # 进行测试
    dataset_params = cfg_inference['dataset_params']
    data_dir = dataset_params['data_dir']
    # for part in dataset_params['part_val']:
    for part in dataset_params['part_test']:
        print('当前测试数据集 part：', part)
        data_path = os.path.join(dataset_params['data_dir'], part, dataset_params['img_folder'])
        label_path = os.path.join(dataset_params['data_dir'], part, dataset_params['label_folder'])
        test(cfg_inference, data_path, label_path)
