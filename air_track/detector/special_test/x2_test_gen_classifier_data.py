# -*- coding: utf-8 -*-
# @Author    :
# @File      : x2_test_gen_classifier_data.py
# @Created   : 2025/7/19 上午10:23
# @Desc      : 使用SmallTargetDetector推理生成的txts文件，来生成二级分类器数据

import os
import cv2
import time
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


def parse_yolo_label(label_path):
    """解析YOLO格式的标签文件"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = int(parts[0])
            if cls == 17:  # TODO 临时过滤szb代码
                continue
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def is_center_point_in_boxes(x, y, boxes):
    """判断点(x,y)是否在任意一个box内"""
    for box in boxes:
        cls_idx, cx, cy, w, h = box
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        if x1 <= x <= x2 and y1 <= y <= y2:
            return int(cls_idx), True
    return -1, False


def is_bbox_overlap(pred_bbox, boxes, threshold=0.5):
    """
    判断预测框与标签框的交集面积是否超过标签框面积的30%

    参数:
        pred_bbox: 预测框 (cls, cx, cy, w, h)
        boxes: 标签框列表 [(cls, cx, cy, w, h), ...]
        threshold: 面积比例阈值 (默认0.3)

    返回:
        (matched_cls, is_overlap):
            matched_cls: 匹配的类别ID (若无匹配返回-1)
            is_overlap: 是否满足阈值条件
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

    for box in boxes:
        cls_idx, cx, cy, w, h = box
        # 计算标签框坐标和面积
        box_x1 = cx - w / 2
        box_y1 = cy - h / 2
        box_x2 = cx + w / 2
        box_y2 = cy + h / 2
        box_area = w * h

        # 计算交集区域
        inter_x1 = max(pred_x1, box_x1)
        inter_y1 = max(pred_y1, box_y1)
        inter_x2 = min(pred_x2, box_x2)
        inter_y2 = min(pred_y2, box_y2)

        # 检查是否有交集
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue

        # 计算交集面积和比例
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        overlap_ratio = inter_area / box_area

        if overlap_ratio > threshold:
            return int(cls_idx), True

    return -1, False


def get_crop_points(bbox, expand_ratio=2.0):
    """获取2倍预测框为扣图框"""
    _, cx, cy, bw, bh = bbox
    # 确定扩展后的边长（取长边的expand_ratio倍）
    long_side = max(bw, bh)
    new_size = (long_side * expand_ratio)

    # 计算裁剪区域
    x1 = cx - new_size / 2
    y1 = cy - new_size / 2
    x2 = x1 + new_size
    y2 = y1 + new_size

    # 处理超出边界的情况
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > 1:
        x1 -= (x2 - 1)
        x2 = 1
    if y2 > 1:
        y1 -= (y2 - 1)
        y2 = 1

    # 确保坐标在合理范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(1, x2), min(1, y2)
    return x1, y1, x2, y2


def crop_image_with_bbox(image, bbox, expand_ratio=2.0):
    """
    根据bbox裁剪图像，以长边的expand_ratio倍进行扩展
    如果超出图像边界则平移回图像内
    """
    h, w = image.shape[:2]
    cx, cy, bw, bh = bbox

    # 转换为像素坐标
    cx_px = int(cx * w)
    cy_px = int(cy * h)
    bw_px = int(bw * w)
    bh_px = int(bh * h)

    # 确定扩展后的边长（取长边的expand_ratio倍）
    long_side = max(bw_px, bh_px)
    new_size = int(long_side * expand_ratio)

    # 计算裁剪区域
    x1 = cx_px - new_size // 2
    y1 = cy_px - new_size // 2
    x2 = x1 + new_size
    y2 = y1 + new_size

    # 处理超出边界的情况
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h

    # 确保坐标在合理范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # 裁剪图像
    cropped = image[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def process_data(pred_root, label_root, output_root, orig_img_folder='rgb', orig_label_folder='label',
                 predict_folder='rgb/txts', img_format='.jpg', label_format='.txt'):
    """处理所有数据"""
    # 创建输出目录
    os.makedirs(os.path.join(output_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'npy'), exist_ok=True)

    # 初始化CSV数据
    csv_data = []
    img_counter = 1

    # 遍历预测结果
    for pred_dir in sorted(glob(os.path.join(pred_root, '*'))):
        video_id = os.path.basename(pred_dir)

        # 获取对应的标签和图像路径
        label_dir = os.path.join(label_root, video_id, orig_label_folder)
        rgb_dir = os.path.join(label_root, video_id, orig_img_folder)

        if not os.path.exists(label_dir) or not os.path.exists(rgb_dir):
            continue

        # 遍历预测的txt文件
        pred_txts = glob(os.path.join(pred_dir, predict_folder, f'*{label_format}'))
        for pred_txt in tqdm(pred_txts, desc=f'Processing {video_id}'):
            # 获取对应的图像文件名
            base_name = os.path.splitext(os.path.basename(pred_txt))[0]
            img_name = base_name + img_format  # 假设图像是jpg格式
            img_path = os.path.join(rgb_dir, img_name)

            # 获取对应的标签文件名
            label_name = base_name + label_format
            label_path = os.path.join(label_dir, label_name)

            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue

            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                continue

            # 解析预测和标签
            pred_boxes = parse_yolo_label(pred_txt)
            label_boxes = parse_yolo_label(label_path)

            # 处理每个预测框
            for i, pred_box in enumerate(pred_boxes):
                cls, cx, cy, w, h = pred_box
                crop_box = get_crop_points(pred_box)
                # 判断是正样本还是负样本
                cls_idx, is_pos = is_bbox_overlap(crop_box, label_boxes)

                if cls_idx >= 0:
                    try:
                        cls_name = classes[cls_idx]
                    except:
                        print(f'cls_idx is error: {cls_idx}')
                        print('error_label_path: ', label_path)
                        continue
                else:
                    cls_name = 'other'

                # 裁剪图像
                cropped_img, _ = crop_image_with_bbox(image, (cx, cy, w, h))

                if cropped_img.size == 0:
                    continue

                # 保存图像和npy
                img_save_path = os.path.join(output_root, 'img', f'{img_counter}.jpg')
                npy_save_path = os.path.join(output_root, 'npy', f'{img_counter}.npy')

                cv2.imwrite(img_save_path, cropped_img)
                np.save(npy_save_path, cropped_img)

                # 添加到CSV数据
                csv_data.append({
                    'index': img_counter,
                    'cls_name': cls_name,  # 根据你的需求修改
                    'neg': 0 if is_pos else 1,
                    'pos': 1 if is_pos else 0
                })

                img_counter += 1

    # 保存CSV文件
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(output_root, 'label.csv'), index=False)


if __name__ == '__main__':
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

    parts = [
        '0611', '0615', '0615_bangwan', '0618',
        '0715_JD_background', '0715_JD_no_background',
        '0715_T18_no_background',
        '0717_T18_101_1', '0717_T18_101',
        '0727_JD_201', '0727_JD_202', '0727_T18_101', '0727_T18_101_background', '0727_T18_102',
        '0807_JD_part', '0812_FC_30', 'T18-2',
        '0815_JD_201_background', '0815_JD_202_background_1', '0815_JD_202_background_2',
        '0815_JD_202_background_3', '0815_JD_203_background', '0815_T18_101_background'
    ]

    # 添加耗时统计
    total_start = time.time()

    '''多个part一次性生成'''
    for part in parts:
        part_start = time.time()
        print(f"\n{'=' * 40}\n开始处理 part: {part}\n{'=' * 40}")

        pred_root = f'/media/linana/1276341C76340351/csz/SmallTargetDetector/build/test/0816_JD_model/JDModel_{part}'  # 预测结果根目录
        label_root = f'/media/linana/1276341C76340351/yanqing_orig_data/{part}'  # 标签数据根目录
        output_root = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0818_second_classifier_dataset_temp/JDModel_{part}_second'  # 输出根目录

        process_data(pred_root, label_root, output_root)

        part_time = time.time() - part_start
        print(f"\n{'=' * 40}\n完成处理 part: {part} | 耗时: {part_time:.2f}秒\n{'=' * 40}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 40}\n所有处理完成 | 总耗时: {total_time:.2f}秒\n{'=' * 40}")

    '''单个文件生成'''
    # pred_root = f'/media/linana/1276341C76340351/csz/github/SmallTargetDetector/build/test/output'  # 预测结果根目录
    # label_root = f'/media/linana/1276341C76340351/yanqing_orig_data/0818_JD_part'  # 标签数据根目录
    # output_root = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0721_second_classifier_dataset_temp/JDModel_0818_JD_part_second'  # 输出根目录
    #
    # process_data(pred_root, label_root, output_root)
