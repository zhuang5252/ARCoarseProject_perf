import os
import cv2
import pickle
import numpy as np
from air_track.detector.engine.predictor import Predictor
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import read_flight_id_csv
from air_track.detector.visualization.visualize_and_save import show_distance, show_confidence, \
    visualize_and_save, show_confidence_sorted
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size


# TODO 应对6月节点临时代码，后续需要更新完善


# TODO 代码中无帧对齐
def predict_with_gt(predictor, data_dir, part, flight_id):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg
    classes = cfg['classes']
    img_read_method = predictor.img_read_method
    img_size_w, img_size_h = predictor.dataset_params['img_size']

    img_folder = cfg['img_folder']
    img_path = f'{data_dir}/{part}/{img_folder}/{flight_id}'

    # 从 groundtruth.csv 文件中提取所需信息
    df_flight, frame_numbers, file_names = read_flight_id_csv(cfg, data_dir, flight_id, part)

    inference_times = []
    bbox_gts = []
    frame_nums = []
    candidate_targets = []
    distance_gts = []
    distance_label_x, distance_label_y = [], []
    temp = 0  # 用来跳过标签存在这一帧，但不存在图像数据的帧
    scale_w, scale_h = 0, 0
    for i, (frame_num, file_name) in enumerate(zip(frame_numbers, file_names)):
        print(f"当前处理第{i + 1}/{len(frame_numbers)}帧")

        cur_file = os.path.join(img_path, file_name)
        if img_read_method == 'gray':
            cur_img = cv2.imread(cur_file, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        else:
            cur_img = cv2.imread(cur_file)

        # 若当前图片不存在，则使用temp使得cur_img读取 i + frame_step * input_frames 帧
        if cur_img is None:
            continue

        # 图像存在才进行后续操作
        orig_h, orig_w = cur_img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

        # 数据前处理
        input_data = predictor.process_input(cur_img)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, inference_time = predictor.predict(input_data)
        inference_times.append(inference_time)

        # 取出列表中标题的index行的数据
        gt_left = df_flight['gt_left'].iloc[i]
        gt_right = df_flight['gt_right'].iloc[i]
        gt_top = df_flight['gt_top'].iloc[i]
        gt_bottom = df_flight['gt_bottom'].iloc[i]
        distance = df_flight['range_distance_m'].iloc[i]

        bbox_gt = [1, gt_left, gt_top, gt_right, gt_bottom]

        if not np.isnan(distance):
            distance_label_x.append(i)
            distance_label_y.append(distance)

        frame_nums.append(i)
        distance_gts.append(distance)
        bbox_gts.append([bbox_gt])
        candidate_targets.append(detected_objects)

        # 检测结果可视化
        if cfg['visualize_flag']:
            visualize_and_save(
                cur_file, detected_objects,
                os.path.join(cfg['visualize_save_dir'], part),
                scale_w, scale_h, [bbox_gt])

    if cfg['calculate_metrics_flag']:
        save_file_dir = cfg['metrics_cache_save_file']
        os.makedirs(save_file_dir, exist_ok=True)
        cache_file = f'{save_file_dir}/cache_{flight_id}.pkl'
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
                            file_name=f'conf_{part}.png')
            show_confidence_sorted(pos_conf_list, neg_conf_list, save_path=cfg['visualize_save_dir'],
                                   file_name=f'conf_{part}_sorted.png')

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == "__main__":
    """有标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict_AR.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    # 创建预测器
    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    # parts = cfg_data['part_train'] + cfg_data['part_val'] + cfg_data['part_test']
    parts = cfg_predict['part_test']
    for part in parts:
        print('当前测试数据集 part：', part)
        for flight_id in os.listdir(os.path.join(cfg_predict['data_dir'], part, cfg_predict['img_folder'])):
            print('当前测试数据集 flight_id：', flight_id)
            predict_with_gt(predictor, cfg_predict['data_dir'], part=part, flight_id=flight_id)
