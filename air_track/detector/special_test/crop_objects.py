# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - crop_objects.py
# @Created   : 2025/08/08 15:47
# @Desc      : 对yolo格式数据进行bbox扣图，并保存在指定路径下

import os
import cv2
import random
from pathlib import Path
from shutil import copyfile


def select_random_samples(rgb_dir, label_dir, output_rgb_dir, output_label_dir, num_samples=1000, seed=None):
    """
    从数据集中随机选择指定数量的样本和对应的标签

    参数:
        rgb_dir: 原始图片目录
        label_dir: 原始标签目录
        output_rgb_dir: 选中的图片输出目录
        output_label_dir: 选中的标签输出目录
        num_samples: 要选择的样本数量
        seed: 随机种子(可选)
    """
    # 设置随机种子(确保可重复性)
    if seed is not None:
        random.seed(seed)

    # 确保输出目录存在
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有有效的图片文件(确保有对应的标签文件)
    valid_images = []
    for img_file in Path(rgb_dir).glob('*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        label_file = Path(label_dir) / (img_file.stem + '.txt')
        if label_file.exists():
            valid_images.append(img_file)

    # 检查是否有足够的样本
    if len(valid_images) < num_samples:
        raise ValueError(f"数据集只有 {len(valid_images)} 个有效样本，少于请求的 {num_samples} 个")

    # 随机选择样本
    selected_images = random.sample(valid_images, num_samples)

    # 复制选中的图片和标签到输出目录
    for img_file in selected_images:
        # 复制图片
        dst_img = Path(output_rgb_dir) / img_file.name
        copyfile(img_file, dst_img)

        # 复制标签
        label_file = Path(label_dir) / (img_file.stem + '.txt')
        dst_label = Path(output_label_dir) / label_file.name
        copyfile(label_file, dst_label)

    print(f"已成功随机选择 {num_samples} 个样本到 {output_rgb_dir} 和 {output_label_dir}")


def crop_and_save_objects(rgb_dir, label_dir, output_dir, classes, filter_classes):
    """
    从rgb图片中根据yolo格式的标签裁剪出指定类别的目标对象并保存

    参数:
        rgb_dir: 包含原始图片的目录
        label_dir: 包含yolo格式标签的目录
        output_dir: 保存裁剪后小图的目录
        classes: 所有类别的列表
        filter_classes: 需要裁剪的类别名称列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 将filter_classes转换为小写以便不区分大小写比较
    filter_classes = [c.lower() for c in filter_classes]

    # 遍历rgb目录中的所有图片文件
    for img_file in Path(rgb_dir).glob('*'):
        # 只处理常见的图片格式
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        # 获取对应的标签文件路径
        label_file = Path(label_dir) / (img_file.stem + '.txt')

        # 如果标签文件不存在则跳过
        if not label_file.exists():
            continue

        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"无法读取图片: {img_file}")
            continue

        img_height, img_width = img.shape[:2]

        # 读取标签文件
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # 处理每个检测目标
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            # 解析yolo格式的标签
            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            # 检查当前类别是否在filter_classes中
            if class_id < 0 or class_id >= len(classes):
                continue

            class_name = classes[class_id].lower()
            if class_name not in filter_classes:
                continue

            # 将归一化坐标转换为绝对坐标
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # 计算边界框的左上角和右下角坐标
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # 确保坐标在图片范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width - 1, x_max)
            y_max = min(img_height - 1, y_max)

            # 裁剪目标区域
            cropped = img[y_min:y_max, x_min:x_max]

            if cropped.size == 0:
                continue

            # 生成保存路径
            save_path = Path(output_dir) / f"{img_file.stem}_{i}_{classes[class_id]}.jpg"

            # 保存裁剪后的小图
            cv2.imwrite(str(save_path), cropped)

        print(f"处理完成: {img_file}")


# 使用示例
if __name__ == "__main__":
    base_path = '/media/linana/1276341C76340351/yanqing_orig_data/0615_diff_focal/0615_focal_80/4'
    rgb_dir = base_path + "/rgb"  # 原始图片目录
    label_dir = base_path + "/label"  # 标签目录
    output_dir = base_path + "/cropped_objects"  # 输出目录

    # 所有类别的列表 (根据你的数据集修改)
    classes = [
              'himars_laucher',
              'hismars_carrier',
              'jinbei',
              'jinbei-fake',
              'haimars_laucher-fake',
              'car',
              'LT-2000',
              'tianbing-fasheche',
              'aiguozhe-fasheche',
              'tiangong-fasheche',
              'tianbing-leidache',
              'aiguozhe-leidache',
              'tiangong-leidache',
              'tianbing-zhihuiche',
              'aiguozhe-zhihuiche',
              'tiangong-zhihuiche',
              'hanma',
              'szb',
              'himars_laucher-wzw'
              ]

    # 需要裁剪的类别名称列表 (根据你的需求修改)
    # filter_classes = ['himars_laucher', 'hanma', 'LT-2000']
    filter_classes = [              'himars_laucher',
              'hismars_carrier',
              'haimars_laucher-fake',
              'LT-2000',
              'tianbing-fasheche',
              'aiguozhe-fasheche',
              'tiangong-fasheche',
              'tianbing-leidache',
              'aiguozhe-leidache',
              'tiangong-leidache',
              'tianbing-zhihuiche',
              'aiguozhe-zhihuiche',
              'tiangong-zhihuiche',
              'hanma',
              'himars_laucher-wzw']

    # # 1. 随机选择1000个样本
    # select_random_samples(
    #     rgb_dir=original_rgb_dir,
    #     label_dir=original_label_dir,
    #     output_rgb_dir=sampled_rgb_dir,
    #     output_label_dir=sampled_label_dir,
    #     num_samples=1000,
    #     seed=42  # 固定随机种子确保可重复性
    # )
    #
    # # 2. 对选中的样本执行扣图操作
    # crop_and_save_objects(
    #     rgb_dir=sampled_rgb_dir,
    #     label_dir=sampled_label_dir,
    #     output_dir=cropped_output_dir,
    #     classes=classes,
    #     filter_classes=filter_classes
    # )
    crop_and_save_objects(rgb_dir, label_dir, output_dir, classes, filter_classes)
    print("所有图片处理完成！")
