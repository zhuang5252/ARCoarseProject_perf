# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - analyse_yolo_data_cls_target_pixel_count.py
# @Created   : 2025/07/22 19:54
# @Desc      : yolo数据分析脚本

"""
新数据分析脚本：
    1. 分析原始数据大中小（目标尺寸、目标类别）
"""
import os
import cv2
from collections import defaultdict


def analyze_dataset(root_dir):
    size_stats = {
        'total': 0,
        'small': 0,
        'medium': 0,
        'large': 0,
    }
    class_stats = defaultdict(lambda: {
        'total': 0,
        'small': 0,
        'medium': 0,
        'large': 0,
    })
    image_stats = defaultdict(lambda: {
        'small': False,
        'medium': False,
        'large': False,
        'classes': set()
    })

    # 遍历所有子文件夹
    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(sub_path):
            continue
        rgb_dir = os.path.join(root_dir, subdir, 'rgb')
        label_dir = os.path.join(root_dir, subdir, 'label')

        if not os.path.exists(rgb_dir) or not os.path.exists(label_dir):
            continue

        # 获取所有图片文件
        image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')]

        for img_file in image_files:
            # 获取对应的标签文件
            label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if not os.path.exists(label_file):
                continue

            # 读取图像尺寸
            img_path = os.path.join(rgb_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            # 读取标签文件
            with open(label_file, 'r') as f:
                lines = f.readlines()

            has_small = False
            has_medium = False
            has_large = False
            classes_in_image = set()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                # 转换到像素坐标
                w_px = w * img_w
                h_px = h * img_h
                long_edge = max(w_px, h_px)

                # 统计目标尺寸
                size_stats['total'] += 1
                class_stats[cls_id]['total'] += 1

                if long_edge < 30:
                    size_stats['small'] += 1
                    class_stats[cls_id]['small'] += 1
                    has_small = True
                elif 30 <= long_edge <= 90:
                    size_stats['medium'] += 1
                    class_stats[cls_id]['medium'] += 1
                    has_medium = True
                else:
                    size_stats['large'] += 1
                    class_stats[cls_id]['large'] += 1
                    has_large = True

                classes_in_image.add(cls_id)

            # 记录图像级别的尺寸信息
            image_key = os.path.join(subdir, 'rgb', img_file)
            image_stats[image_key]['small'] = has_small
            image_stats[image_key]['medium'] = has_medium and not has_small
            image_stats[image_key]['large'] = has_large and not (has_small or has_medium)
            image_stats[image_key]['classes'] = classes_in_image

    return size_stats, class_stats, image_stats


def save_analysis_results(size_stats, class_stats, output_file):
    with open(output_file, 'w') as f:
        f.write(f"Total BBoxes: {size_stats['total']}\n")
        f.write(f"Small BBoxes (<30px): {size_stats['small']}\n")
        f.write(f"Medium BBoxes (30-90px): {size_stats['medium']}\n")
        f.write(f"Large BBoxes (>90px): {size_stats['large']}\n")

        small_ratio = size_stats['small'] / size_stats['total'] * 100 if size_stats['total'] > 0 else 0
        f.write(f"Small Ratio: {small_ratio:.2f}%\n")
        medium_ratio = size_stats['medium'] / size_stats['total'] * 100 if size_stats['total'] > 0 else 0
        f.write(f"Medium Ratio: {medium_ratio:.2f}%\n")
        large_ratio = size_stats['large'] / size_stats['total'] * 100 if size_stats['total'] > 0 else 0
        f.write(f"Large Ratio: {large_ratio:.2f}%\n\n")

        f.write("Per-class statistics:\n")
        for cls_id in sorted(class_stats.keys()):
            stats = class_stats[cls_id]
            small_ratio = stats['small'] / stats['total'] * 100 if stats['total'] > 0 else 0
            medium_ratio = stats['medium'] / stats['total'] * 100 if stats['total'] > 0 else 0
            large_ratio = stats['large'] / stats['total'] * 100 if stats['total'] > 0 else 0

            f.write(f"  cls_{cls_id}: {stats['total']} boxes ")
            f.write(f"(small: {stats['small']}, ratio {small_ratio:.2f}%; ")
            f.write(f"medium: {stats['medium']}, ratio {medium_ratio:.2f}%; ")
            f.write(f"large: {stats['large']}, ratio {large_ratio:.2f}%)\n")


def main():
    # 配置路径
    root_dir = '/media/linana/1276341C76340351/yanqing_orig_data/0818_JD_part'  # 数据集根目录
    analysis_file = '/media/linana/1276341C76340351/yanqing_orig_data/0818_JD_part/analysis_target_size_result.txt'  # 分析结果

    print("Analyzing dataset...")
    size_stats, class_stats, image_stats = analyze_dataset(root_dir)
    save_analysis_results(size_stats, class_stats, analysis_file)
    print(f"First analysis saved to {analysis_file}")


if __name__ == '__main__':
    main()