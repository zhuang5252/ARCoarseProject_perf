# -*- coding: utf-8 -*-
# @Author    :
# @File      : predict_img.py
# @Created   : 2025/7/19 下午4:23
# @Desc      : 二级分类器无标签推理测试

import os
import cv2
from air_track.classifier.engine import Predictor
from air_track.utils import combine_load_cfg_yaml, extract_number, reprod_init, col_data_save_csv


def predict_without_gt(predictor, data_dir, save_predictions=False):
    cfg = predictor.cfg

    img_folder = os.path.join(data_dir, cfg['img_folder'])
    img_read_method = predictor.img_read_method

    img_paths = [f for f in os.listdir(img_folder) if f.endswith(cfg['img_format'])]
    img_paths.sort(key=extract_number)

    labels = [[]] * len(img_paths)

    inference_times = []
    idx_list, neg_classify_predictions, pos_classify_predictions = [], [], []
    frame_num = len(img_paths)
    for i, (img_file_name, label) in enumerate(zip(img_paths, labels)):
        print(f'当前处理第{i + 1}/{frame_num}帧')
        idx_list.append(img_file_name)
        img_file = os.path.join(img_folder, img_file_name)

        if not os.path.exists(img_file):
            continue

        if img_read_method == 'gray':
            data = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            data = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        else:
            data = cv2.imread(img_file)

        data = predictor.process_input(data)
        classify_prediction, cls_name_batch, inference_time = predictor.predict(data)
        inference_times.append(inference_time)

        pos_classify_predictions.append(classify_prediction[0][1])
        neg_classify_predictions.append(classify_prediction[0][0])

    if save_predictions:
        save_classify_predictions = {'img_name': idx_list, 'neg_conf': neg_classify_predictions, 'pos_conf': pos_classify_predictions}
        col_data_save_csv(f'{data_dir}/classify_predictions.csv', save_classify_predictions)


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    predict_without_gt(predictor, cfg_predict['test_data_dir'], save_predictions=False)
