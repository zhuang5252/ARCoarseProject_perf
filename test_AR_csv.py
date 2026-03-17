import os
import cv2
import pickle
import numpy as np
from air_track.engine.predictor import Predictor
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import read_flight_id_csv
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_distance, show_confidence, \
    show_confidence_sorted
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size, check_image_bit_depth


def test(inference_yaml, data_dir, part, flight_id):
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

    '''读取文件循环测试'''
    img_folder = predictor.cfg_inference['dataset_params']['img_folder']
    img_path = f'{data_dir}/{part}/{img_folder}/{flight_id}'

    # 从 groundtruth.csv 文件中提取所需信息
    df_flight, frame_numbers, file_names = read_flight_id_csv(predictor.cfg_inference['dataset_params'],
                                                              data_dir, flight_id, part)

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
    for i, (frame_num, file_name) in enumerate(zip(frame_numbers, file_names)):
        print(f"当前处理第{i + 1}/{len(frame_numbers)}帧")

        if i < frame_step * (input_frames - 1):
            continue

        if i < temp:
            continue

        cur_img_path = os.path.join(img_path, file_name)

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
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h

        # 校验图像深度与配置中的max_pixel是否一致
        check_image_bit_depth(cur_img_orig, cur_img_path, max_pixel)
        # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
        cur_img = check_and_change_img_size(cur_img_orig, img_size_w, img_size_h)

        # 读取prev帧图像数据
        if input_frames > 1:
            try:
                prev_images = predictor.get_prev_step_images(
                    img_path, file_names, i, img_size_w, img_size_h, frame_step, input_frames)
            except:
                continue
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

        '''计算指标所需数据'''
        frame_bbox_xyxy, frame_bbox_cxcywhn = [], []
        if detect_cfg['calculate_metrics_flag'] and df_flight is not None:
            # 取出列表中标题的index行的数据 TODO 现在这里是单帧单目标，后续按需扩展到多帧多目标
            gt_left = df_flight['gt_left'].iloc[i]
            gt_right = df_flight['gt_right'].iloc[i]
            gt_top = df_flight['gt_top'].iloc[i]
            gt_bottom = df_flight['gt_bottom'].iloc[i]
            distance = df_flight['range_distance_m'].iloc[i]
            bbox_gt = [1, gt_left, gt_top, gt_right, gt_bottom]

            if not np.isnan(distance):
                distance_label_x.append(i)
                distance_label_y.append(distance)

            distance_gts.append(distance)
            frame_bbox_xyxy.append(bbox_gt)
            bbox_xyxy.append(frame_bbox_xyxy)
            # bbox_cxcywhn
        else:
            distance = None

        frame_nums.append(i)
        img_paths.append(cur_img_path)
        scale_w_list.append(scale_w)
        scale_h_list.append(scale_h)
        candidate_targets.append(final_detected_objects)

        # 检测结果可视化
        if detect_cfg['visualize_flag']:
            # 可视化结果保存
            visualize_and_save(cur_img_path, final_detected_objects, os.path.join((detect_cfg['visualize_save_dir']), part),
                               scale_w, scale_h, frame_bbox_xyxy, distance)

    if detect_cfg['calculate_metrics_flag']:
        save_file_dir = detect_cfg['metrics_cache_save_file']
        os.makedirs(save_file_dir, exist_ok=True)
        cache_file = f'{save_file_dir}/cache_{part}_{flight_id}.pkl'
        with open(cache_file, 'wb') as file:
            pickle.dump({
                'img_paths': img_paths,
                'frame_nums': frame_nums,
                'bbox_xyxy': bbox_xyxy,
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
    inference_yaml = os.path.join(script_dir, 'air_track/config/inference_AR_csv.yaml')
    cfg_inference = combine_load_cfg_yaml(yaml_paths_list=[inference_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_inference['seed'])

    # 进行测试
    dataset_params = cfg_inference['dataset_params']
    data_dir = dataset_params['data_dir']
    # for part in dataset_params['part_val']:
    for part in dataset_params['part_test']:
        print('当前测试数据集 part：', part)
        for flight_id in os.listdir(os.path.join(data_dir, part, dataset_params['img_folder'])):
            if os.path.isdir(os.path.join(data_dir, part, dataset_params['img_folder'], flight_id)):
                print('当前测试数据集 flight_id：', flight_id)
                test(inference_yaml, data_dir, part=part, flight_id=flight_id)
