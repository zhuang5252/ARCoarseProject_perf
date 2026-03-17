# -*- coding: utf-8 -*-
# @Author    :
# @File      : AirTrack - analyse_yolo_data_distribution.py
# @Created   : 2025/07/18 19:54
# @Desc      : yolo数据分析脚本（分析目标尺寸的一些信息）

"""
新数据分析脚本：
    1. 分析原始数据分布（目标尺寸的一些信息）
"""

import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from air_track.utils import combine_load_cfg_yaml


def plot_distribution(stats, title):
    plt.figure(figsize=(10, 6))
    plt.hist(stats, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Size')
    plt.ylabel('Count')
    plt.show()


def get_stats(data):
    """计算分布统计"""
    if len(data) == 0:
        return None
    return {
        'count': len(data),  # 目标总数
        'min': np.min(data),  # 目标最小尺寸
        'max': np.max(data),  # 目标最大尺寸
        'mean': np.mean(data),  # 目标尺寸均值
        'median': np.median(data),  # 目标尺寸中位数
        'std': np.std(data),  # 目标尺寸标准差
        '25%': np.percentile(data, 25),  # 目标尺寸25%的值
        '98%': np.percentile(data, 98)  # 目标尺寸98%的值
    }


def analyse_bbox_distribution(cfg, shape):
    """分析边界框尺寸分布，返回最大边长及宽高统计"""

    # 初始化统计容器
    max_side = 0.0
    widths = []
    heights = []

    # 配置解析
    data_dir = cfg['data_dir']
    img_folder = cfg['img_folder']
    label_folder = cfg['label_folder']
    img_format = cfg['img_format']
    gt_format = cfg['gt_format']
    parts = cfg['part_train'] + cfg['part_val'] + cfg['part_test']

    for part in parts:
        img_dir = os.path.join(data_dir, part, img_folder)
        label_dir = os.path.join(data_dir, part, label_folder)

        # 跳过不存在目录
        if not os.path.exists(img_dir):
            print(f"跳过不存在目录: {img_dir}")
            continue

        img_names = [n for n in os.listdir(img_dir) if n.endswith(img_format)]
        for img_name in img_names:
            # 获取图像尺寸
            img_w, img_h = shape

            # 处理对应标签
            label_name = img_name.replace(img_format, gt_format)
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # 解析标签
                    try:
                        _, xc, yc, w_norm, h_norm = line.split()
                        w = float(w_norm) * img_w
                        h = float(h_norm) * img_h
                    except Exception as e:
                        print(f"标签解析失败: {label_path} 行{line_num} ({str(e)})")
                        continue

                    # 记录数据
                    widths.append(w)
                    heights.append(h)
                    current_max = max(w, h)
                    if current_max > max_side:
                        max_side = current_max

    return {
        'max_side': max_side,
        'width_distribution': get_stats(widths),
        'height_distribution': get_stats(heights)
    }


if __name__ == '__main__':
    cfg_path = '/air_track/detector/config/dataset_dalachi.yaml'

    # 加载配置文件
    cfg = combine_load_cfg_yaml(yaml_paths_list=[cfg_path])
    bbox_distribution = analyse_bbox_distribution(cfg, shape=(1920, 1152))

    print()
