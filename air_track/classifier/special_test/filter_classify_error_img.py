# -*- coding: utf-8 -*-
# @Author    :
# @File      : filter_classify_error_img.py
# @Created   : 2025/7/19 下午2:23
# @Desc      : 筛选并分析二级分类器分类错误样本，并保存到指定路径下

import copy
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from air_track.classifier.engine import Predictor
from air_track.detector.visualization.visualize_and_save import show_bar_chart
from air_track.utils import combine_load_cfg_yaml, extract_number, reprod_init, col_data_save_csv


def write_label(label_lists, save_path):
    """将标签写入文件"""
    os.makedirs(save_path, exist_ok=True)

    orig_idx_list, neg_conf, pos_conf = [], [], []
    idx_list, cls_name_list, neg_list, pos_list = [], [], [], []
    for i, label in enumerate(label_lists):
        print(f"当前标签保存第{i + 1}/{len(label_lists)}帧")
        label = label_lists[i]

        idx_list.append(label[0])
        cls_name_list.append(label[1])
        neg_list.append(label[2])
        pos_list.append(label[3])
        orig_idx_list.append(label[4])
        neg_conf.append(label[5])
        pos_conf.append(label[6])

    save_labels = {'index': idx_list, 'cls_name': cls_name_list, 'neg': neg_list, 'pos': pos_list,
                   'orig_index': orig_idx_list, 'neg_conf': neg_conf, 'pos_conf': pos_conf}
    col_data_save_csv(f'{save_path}/label.csv', save_labels)


def predict_with_gt(predictor, data_dir, save_folder='error_samples', mv=False):
    cfg = predictor.cfg

    img_folder = os.path.join(data_dir, cfg['img_folder'])
    label_file = os.path.join(data_dir, cfg['label_file'])
    img_read_method = predictor.img_read_method

    img_paths = [f for f in os.listdir(img_folder) if f.endswith(cfg['img_format'])]
    img_paths.sort(key=extract_number)
    labels = pd.read_csv(label_file).values

    if len(img_paths) != len(labels):
        assert 'frame_num and label_num not equal !'

    output_dir = os.path.join(data_dir, save_folder)
    output_img_dir = os.path.join(data_dir, save_folder, 'img')
    output_npy_dir = os.path.join(data_dir, save_folder, 'npy')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)

    idx = 1
    num_correct = 0
    pos_cnt = 0
    pos_classify_neg_cnt = 0
    neg_classify_pos_cnt = 0
    inference_times = []
    save_labels = []
    pos_conf_list, neg_conf_list = [], []
    idx_list, neg_classify_predictions, pos_classify_predictions = [], [], []
    frame_num = len(img_paths)
    # tqdm(enumerate(zip(img_paths, labels)), desc=f'Processing {data_dir}')
    for i, (img_file_name, label) in enumerate(zip(img_paths, labels)):
        print(f'当前处理第{i + 1}/{frame_num}帧')
        img_file = os.path.join(img_folder, img_file_name)
        label_temp = copy.deepcopy(label)

        if not os.path.exists(img_file):
            continue

        if img_read_method == 'gray':
            data = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            data = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        else:
            data = cv2.imread(img_file)

        img_orig = copy.deepcopy(data)
        label = cfg['classes'][list(label[2:]).index(1)]

        data = predictor.process_input(data)
        classify_prediction, cls_name_batch, inference_time = predictor.predict(data)
        inference_times.append(inference_time)

        if label == 'pos':
            pos_cnt += 1
            pos_conf_list.append(classify_prediction[0][1])
        else:
            neg_conf_list.append(classify_prediction[0][0])

        if cls_name_batch[0] == label:
            num_correct += 1
        elif label == 'pos' and cls_name_batch[0] == 'neg':
            pos_classify_neg_cnt += 1
        elif label == 'neg' and cls_name_batch[0] == 'pos':
            neg_classify_pos_cnt += 1

        if cls_name_batch[0] == label:
            pass
        else:
            # 保存原始标签
            label_temp[0] = idx
            # 添加保存分类的softmax结果
            label_temp = list(label_temp) + [int(img_file_name.split('.')[0])] + list(classify_prediction[0])
            save_labels.append(np.array(label_temp))
            # copy分类错误图像数据
            output_img_path = os.path.join(output_img_dir, f'{idx}.jpg')
            if mv:
                shutil.move(img_file, output_img_path)
            else:
                shutil.copy2(img_file, output_img_path)
            # 保存分类错误图像数据为npy格式
            output_npy_path = os.path.join(output_npy_dir, f'{idx}.npy')
            np.save(output_npy_path, img_orig)
            idx += 1
            print(f"分辨错误样本：{img_file}")

    # 写入标签
    write_label(save_labels, output_dir)
    # 写入classify_predictions
    # save_classify_predictions = {'index': idx_list, 'neg_conf': neg_classify_predictions, 'pos_conf': pos_classify_predictions}
    # col_data_save_csv(f'{output_dir}/classify_predictions.csv', save_classify_predictions)

    # save_conf = {'pos_conf': pos_conf_list}
    # col_data_save_csv(f'{output_dir}/conf1.csv', save_conf)

    # 可视化正负样本概率分布
    show_bar_chart(pos_conf_list, output_dir, file_name='Pos_Confidence_distribution.png',
                   # range_min=0, range_max=0.101, range_interval=0.005,
                   range_min=0, range_max=1.001, range_interval=0.05,
                   title='Confidence distribution', xlabel='Confidence', ylabel='Count/Scale')
    show_bar_chart(neg_conf_list, output_dir, file_name='Neg_Confidence_distribution.png',
                   # range_min=0, range_max=0.101, range_interval=0.005,
                   range_min=0, range_max=1.001, range_interval=0.05,
                   title='Confidence distribution', xlabel='Confidence', ylabel='Count/Scale')

    neg_cnt = frame_num - pos_cnt
    acc = num_correct / frame_num
    pos_classify_pos_cnt = pos_cnt - pos_classify_neg_cnt
    neg_classify_neg_cnt = neg_cnt - neg_classify_pos_cnt

    try:
        print(f'accuracy: {acc}')
        print(f'pos_cnt: {pos_cnt}')
        print(f'neg_cnt: {neg_cnt}')
        print(f'pos_classify_pos_cnt: {pos_classify_pos_cnt}')
        print(f'pos_classify_neg_cnt: {pos_classify_neg_cnt}')

        print(f'neg_classify_neg_cnt: {neg_classify_neg_cnt}')
        print(f'neg_classify_pos_cnt: {neg_classify_pos_cnt}')

        print(f'pos_classify_pos_cnt / pos_cnt: {pos_classify_pos_cnt / pos_cnt}')
        print(f'pos_classify_neg_cnt / pos_cnt: {pos_classify_neg_cnt / pos_cnt}')

        print(f'neg_classify_neg_cnt / neg_cnt: {neg_classify_neg_cnt / neg_cnt}')
        print(f'neg_classify_pos_cnt / neg_cnt: {neg_classify_pos_cnt / neg_cnt}')
        inference_times = inference_times[2:-2]
        print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')
    except:
        return


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, '../config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    '''使用test_data_dir来进行推理测试并筛选分辨错误样本'''
    # predict_with_gt(predictor, cfg_predict['test_data_dir'], save_folder='error_samples', mv=False)

    '''加载所有part来进行推理测试并筛选分辨错误样本'''
    parts = cfg_predict['parts']
    for part in parts:
        test_data_dir = os.path.join(cfg_predict['data_dir'], part)
        predict_with_gt(predictor, test_data_dir, save_folder='error_samples_2', mv=False)
