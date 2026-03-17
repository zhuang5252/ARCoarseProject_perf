# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - analyse_data_split_mv_test_data.py
# @Created   : 2025/07/22 16:46
# @Desc      : 新数据分析处理脚本

"""
新数据分析处理脚本：
    1. 分析原始数据（目标尺寸、目标类别）
    2. 划分训练集验证集、测试集（只将测试集移动到另一个指定目录下）
    3. 判定拆分的测试集在每个子文件夹下是否满足0.08的最小比例，不满足则补充
    4. 分析处理后的数据
"""

import os
import cv2
import random
import shutil
from collections import defaultdict

from air_track.detector.utils.analyse_yolo_data_cls_target_pixel_count import analyze_dataset, save_analysis_results


def split_dataset(root_dir, image_stats, output_dir, test_ratio=0.1, min_subdir_ratio=0.08):
    # 按尺寸分类图像
    small_images = []
    medium_images = []
    large_images = []

    # 按子文件夹统计
    subdir_stats = defaultdict(lambda: {
        'total': 0,
        'test': 0,
        'images': []
    })

    # 分类图像并统计子文件夹信息
    for img_key, stats in image_stats.items():
        subdir = img_key.split(os.path.sep)[0]
        subdir_stats[subdir]['total'] += 1
        subdir_stats[subdir]['images'].append(img_key)

        if stats['small']:
            small_images.append(img_key)
        elif stats['medium']:
            medium_images.append(img_key)
        elif stats['large']:
            large_images.append(img_key)

    # 随机打乱
    random.shuffle(small_images)
    random.shuffle(medium_images)
    random.shuffle(large_images)

    # 初始选择测试集
    test_images = set()

    # 从每个尺寸组中选择测试图像
    small_test_count = max(1, int(len(small_images) * test_ratio))
    test_images.update(small_images[:small_test_count])

    medium_test_count = max(1, int(len(medium_images) * test_ratio))
    test_images.update(medium_images[:medium_test_count])

    large_test_count = max(1, int(len(large_images) * test_ratio))
    test_images.update(large_images[:large_test_count])

    # 统计当前各子文件夹的测试比例
    for subdir in subdir_stats:
        subdir_test_count = sum(1 for img in subdir_stats[subdir]['images'] if img in test_images)
        subdir_stats[subdir]['test'] = subdir_test_count

    # 检查并补充不足的子文件夹
    for subdir, stats in subdir_stats.items():
        current_ratio = stats['test'] / stats['total'] if stats['total'] > 0 else 0

        if current_ratio < min_subdir_ratio:
            # 计算需要补充的数量
            needed = int(stats['total'] * min_subdir_ratio) - stats['test']
            if needed <= 0:
                continue

            # 从该子文件夹剩余图像中随机选择
            candidates = [img for img in stats['images']
                          if img not in test_images and
                          (img in small_images or img in medium_images or img in large_images)]

            if candidates:
                add_count = min(needed, len(candidates))
                new_test = random.sample(candidates, add_count)
                test_images.update(new_test)
                subdir_stats[subdir]['test'] += add_count

    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

    # 移动测试图像
    for img_key in test_images:
        move_image_and_label(root_dir, img_key, os.path.join(output_dir, 'test'))

    # 移动训练图像
    all_images = set(image_stats.keys())
    train_images = all_images - test_images

    # for img_key in train_images:
    #     move_image_and_label(root_dir, img_key, os.path.join(output_dir, 'train'))

    # 打印拆分统计信息
    print("\nSplit Statistics:")
    for subdir, stats in subdir_stats.items():
        ratio = stats['test'] / stats['total'] if stats['total'] > 0 else 0
        print(f"Subdir {subdir}: {stats['test']}/{stats['total']} ({ratio:.2%}) in test set")

    total_test = len(test_images)
    total_images = len(image_stats)
    print(f"\nTotal: {total_test}/{total_images} ({total_test / total_images:.2%}) in test set")


def move_image_and_label(root_dir, img_key, dest_dir):
    # 解析路径
    parts = img_key.split(os.path.sep)
    subdir = parts[0]
    img_file = parts[2]

    # 源路径
    src_img_path = os.path.join(root_dir, subdir, 'rgb', img_file)
    src_label_path = os.path.join(root_dir, subdir, 'label', os.path.splitext(img_file)[0] + '.txt')

    # 目标路径 (保持子目录结构)
    dest_subdir = os.path.join(dest_dir, subdir)
    os.makedirs(os.path.join(dest_subdir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(dest_subdir, 'label'), exist_ok=True)

    dest_img_path = os.path.join(dest_subdir, 'rgb', img_file)
    dest_label_path = os.path.join(dest_subdir, 'label', os.path.splitext(img_file)[0] + '.txt')

    # 移动文件
    shutil.move(src_img_path, dest_img_path)
    shutil.move(src_label_path, dest_label_path)


def main():
    # 配置路径
    root_dir = '/media/linana/C6F62B30F62B1FE3/data/hhhhh'  # 数据集根目录
    output_dir = '/media/linana/C6F62B30F62B1FE3/data/hhhhh_filter'  # 拆分结果输出目录
    analysis_file1 = '/media/linana/C6F62B30F62B1FE3/data/hhhhh/before_analysis_result.txt'  # 第一次分析结果
    analysis_file2 = '/media/linana/C6F62B30F62B1FE3/data/hhhhh/after_analysis_result.txt'  # 第二次分析结果
    analysis_file3 = '/media/linana/C6F62B30F62B1FE3/data/hhhhh_filter/after_test_analysis_result.txt'  # 第二次分析结果

    # 1. 第一次分析数据集
    print("Analyzing dataset (first pass)...")
    size_stats, class_stats, image_stats = analyze_dataset(root_dir)
    save_analysis_results(size_stats, class_stats, analysis_file1)
    print(f"First analysis saved to {analysis_file1}")

    # 2. 拆分数据集
    print("\nSplitting dataset...")
    split_dataset(root_dir, image_stats, output_dir)
    print(f"Dataset split completed. Results saved to {output_dir}")

    # 3. 第二次分析 (分析拆分后剩余的数据)
    print("\nAnalyzing remaining set after split...")
    # train_dir = os.path.join(output_dir, 'train')
    size_stats, class_stats, _ = analyze_dataset(root_dir)
    save_analysis_results(size_stats, class_stats, analysis_file2)
    print(f"Second analysis saved to {analysis_file2}")

    print("\nAnalyzing testing set after split...")
    train_dir = os.path.join(output_dir, 'test')
    size_stats, class_stats, _ = analyze_dataset(train_dir)
    save_analysis_results(size_stats, class_stats, analysis_file3)
    print(f"Second analysis saved to {analysis_file3}")


if __name__ == '__main__':
    main()
