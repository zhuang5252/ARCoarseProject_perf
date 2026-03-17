import os
import cv2
import pickle
import shutil
import numpy as np
from air_track.engine.predictor import Predictor
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import cxcywhn2xyxy, argmax2d, convert_detections
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_distance, show_confidence, \
    show_confidence_sorted, draw_feature_img_orig_data
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size, check_image_bit_depth, \
    natural_sort_key


def test(inference_yaml, part, img_path, label_folder, show_feature=False):
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

    '''读取文件夹内文件循环测试'''
    file_names = os.listdir(img_path)
    file_names.sort()
    # img_names.sort(key=natural_sort_key)

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
    scale_w_list, scale_h_list = [], []
    for i, img_name in enumerate(file_names):
        print(f"当前处理第{i + 1}/{len(file_names)}帧")

        if i < frame_step * (input_frames - 1):
            continue

        if i < temp:
            continue

        cur_img_path = os.path.join(img_path, img_name)
        cur_label_path = os.path.join(label_folder,
                                      img_name.replace(dataset_params['img_format'], dataset_params['gt_format']))

        # 可以允许标签不存在，即为无标签预测
        if not os.path.exists(cur_img_path):
            continue

        if img_read_method == 'gray':
            cur_img_orig = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            cur_img_orig = cv2.imread(cur_img_path, cv2.IMREAD_UNCHANGED)
        else:
            cur_img_orig = cv2.imread(cur_img_path)

        # 若当前图片不存在，则使用temp使得cur_img读取 i + frame_step * input_frames 帧
        if cur_img_orig is None:
            temp = i + frame_step * input_frames
            continue

        # 图像存在才进行后续操作
        orig_h, orig_w = cur_img_orig.shape[:2]
        scale_w = img_size_w / orig_w  # TODO 应该统一一下scale的计算方式（gen_classifier_data.py与此不一致）
        scale_h = img_size_h / orig_h

        # 校验图像深度与配置中的max_pixel是否一致
        check_image_bit_depth(cur_img_orig, cur_img_path, max_pixel)
        # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
        cur_img = check_and_change_img_size(cur_img_orig, img_size_w, img_size_h)

        # 读取prev帧图像数据
        if input_frames > 1:
            prev_images = predictor.get_prev_step_images(
                img_path, file_names, i, img_size_w, img_size_h, frame_step, input_frames)
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
            # # 转换检测结果到原图尺寸基准下
            # final_detected_objects = convert_detections(detected_objects, scale_w, scale_h)
            classify_inference_time = 0
        final_counts += len(final_detected_objects)

        inference_times.append(align_inference_time + detect_inference_time + classify_inference_time)

        # 特征可视化
        if show_feature:
            if isinstance(prediction[0], np.ndarray):
                show_temp = prediction[0].astype(np.float32)
                show_size = prediction[1].astype(np.float32)
            else:
                show_temp = prediction[0].float().cpu().detach().numpy()
                show_size = prediction[1].float().cpu().detach().numpy()
            home_dir = os.path.expanduser('~')
            save_path = os.path.join(home_dir, 'csz/features')
            os.makedirs(save_path, exist_ok=True)
            max_iloc_y, max_iloc_x = argmax2d(show_temp[0][0])
            name = os.path.basename(cur_img_path).replace('.', '_')
            draw_feature_img_orig_data(
                show_temp[0], f'{save_path}/{name}_mask.png',
                is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)

            draw_feature_img_orig_data(
                show_size[0], f'{save_path}/{name}_size.png',
                is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)

        '''计算指标所需数据'''
        frame_bbox_xyxy, frame_bbox_cxcywhn = [], []
        if os.path.exists(cur_label_path) and detect_cfg['calculate_metrics_flag']:
            # 读取标签
            with open(cur_label_path, 'r') as file:
                labels = file.readlines()
            labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的
            # 将标签由字符串转换为浮点数
            labels = [[float(item) for item in label] for label in labels]

            # 将标签从cxcywhn转换为xyxy
            for label in labels:
                bbox_gt = cxcywhn2xyxy(label[1:], img_w=orig_w, img_h=orig_h)
                frame_bbox_xyxy.append([int(label[0])] + bbox_gt)
                frame_bbox_cxcywhn.append(label)

            bbox_xyxy.append(frame_bbox_xyxy)
            bbox_cxcywhn.append(frame_bbox_cxcywhn)
        else:
            bbox_xyxy.append([])
            bbox_cxcywhn.append([])

        frame_nums.append(i)
        img_paths.append(cur_img_path)
        scale_w_list.append(scale_w)
        scale_h_list.append(scale_h)
        candidate_targets.append(final_detected_objects)

        # 检测结果可视化
        if detect_cfg['visualize_flag']:
            # 可视化结果保存
            visualize_and_save(cur_img_path, final_detected_objects, os.path.join((detect_cfg['visualize_save_dir']), part),
                               scale_w, scale_h, frame_bbox_xyxy, distance=None)

        # 获取当前用户的主目录
        # home_dir = os.path.expanduser('~')
        # # 用于筛选有检测框的数据
        # if final_detected_objects:
        #     save_detected_data_path = os.path.join(home_dir, 'csz/detected_data')
        #     os.makedirs(save_detected_data_path, exist_ok=True)
        #     shutil.move(cur_img_path, os.path.join(save_detected_data_path, img_name))
        # else:
        #     continue

    # part为多级目录时，只保留第一级目录名
    if '/' in part:
        part = part.split('/')[0]

    if detect_cfg['calculate_metrics_flag']:
        save_file_dir = detect_cfg['metrics_cache_save_file']
        os.makedirs(save_file_dir, exist_ok=True)
        cache_file = f'{save_file_dir}/cache_{part}.pkl'
        with open(cache_file, 'wb') as file:
            pickle.dump({
                'img_paths': img_paths,
                'frame_nums': frame_nums,
                'bbox_xyxy': bbox_xyxy,
                'bbox_cxcywhn': bbox_cxcywhn,
                'distance_gts': distance_gts,
                'candidate_targets': candidate_targets,
                'scale_w_list': scale_w_list,
                'scale_h_list': scale_h_list
            }, file)

        detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
        pos_candidate_targets, neg_candidate_targets = calculate_metrics(
            predictor.classes, bbox_xyxy, candidate_targets, frame_nums, scale_w_list, scale_h_list,
            threshold=detect_cfg['threshold'],
            target_min_size=detect_cfg['target_min_size'],
            iou=detect_cfg['iou_flag'], cls_flag=detect_cfg['cls_flag'])
        if detect_cfg['analysis_show']:
            show_confidence(pos_conf_list, neg_conf_list, save_path=detect_cfg['visualize_save_dir'],
                            file_name=f'conf_{part}.png')
            show_confidence_sorted(pos_conf_list, neg_conf_list,
                                   save_path=detect_cfg['visualize_save_dir'],
                                   file_name=f'conf_{part}_sorted.png')
            if len(distance_label_y) > 0:
                show_distance(distance_label_x, distance_label_y, pos_pred_x, distance_pred_y,
                              save_path=detect_cfg['visualize_save_dir'], file_name=f'distance_{part}.png')

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')
    print('一级检测器总检出数量: ', detector_counts)
    print('最终总检出数量: ', final_counts)


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
        test(inference_yaml, part, data_path, label_path, show_feature=False)
