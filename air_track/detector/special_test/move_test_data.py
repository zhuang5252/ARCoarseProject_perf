# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - move_test_data.py
# @Created   : 2025/08/05 09:39
# @Desc      : 从数据集中拆出测试数据，并提供测试数据占比参数，当前为数据同分布

import os
import random
import shutil


def move_random_samples(src_root, dst_root, img_folder='rgb', label_folder='label', ratio=0.1):
    # 定义路径
    rgb_src = os.path.join(src_root, img_folder)
    label_src = os.path.join(src_root, label_folder)

    rgb_dst = os.path.join(dst_root, img_folder)
    label_dst = os.path.join(dst_root, label_folder)

    # 创建目标目录
    os.makedirs(rgb_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)

    # 获取所有图像文件名（不含路径）
    rgb_files = [f for f in os.listdir(rgb_src) if os.path.isfile(os.path.join(rgb_src, f))]

    # 计算需要移动的数量
    num_samples = int(len(rgb_files) * ratio)

    # 随机选择文件
    selected_files = random.sample(rgb_files, num_samples)

    # 移动文件和对应的标签
    for file in selected_files:
        # 移动图像文件
        src_img = os.path.join(rgb_src, file)
        dst_img = os.path.join(rgb_dst, file)

        # 移动对应的标签文件
        label_file = file.replace('.jpg', '.txt')  # 假设文件名相同
        src_label = os.path.join(label_src, label_file)
        dst_label = os.path.join(label_dst, label_file)

        if os.path.exists(src_label):
            shutil.move(src_img, dst_img)
            shutil.move(src_label, dst_label)
        else:
            print(f"Warning: Label file {label_file} not found")


# 使用示例
source_path = '/home/csz_changsha/data/FlySubjectAirToGround_merge_20260109/dataset_orig'  # 你的源路径
target_path = '/home/csz_changsha/data/FlySubjectAirToGround_merge_20260109/dataset_orig/test'  # 你的目标路径

move_random_samples(source_path, target_path, img_folder='images', label_folder='labels', ratio=0.1)
