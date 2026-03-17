# -*- coding: utf-8 -*-
# @Author    :
# @File      : x2_test_gen_classifier_data.py
# @Created   : 2025/7/19 上午10:23
# @Desc      : 使用SmallTargetDetector推理生成的txts文件，来生成二级分类器数据（并行化版本）

import os
import cv2
import time
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


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


def process_single_file(args, video_id, label_root, output_root, classes, img_counter_lock=None):
    """处理单个文件的函数，用于并行化"""
    pred_txt, start_idx = args  # 改为使用起始索引而不是共享计数器
    base_name = os.path.splitext(os.path.basename(pred_txt))[0]
    img_name = base_name + '.jpg'
    label_name = base_name + '.txt'

    # 构造路径
    rgb_dir = os.path.join(label_root, video_id, 'rgb')
    label_dir = os.path.join(label_root, video_id, 'label')

    img_path = os.path.join(rgb_dir, img_name)
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(img_path) or not os.path.exists(label_path):
        return []

    image = cv2.imread(img_path)
    if image is None:
        return []

    pred_boxes = parse_yolo_label(pred_txt)
    label_boxes = parse_yolo_label(label_path)

    file_data = []
    current_idx = start_idx  # 使用局部计数器

    for i, pred_box in enumerate(pred_boxes):
        cls, cx, cy, w, h = pred_box
        crop_box = get_crop_points(pred_box)
        cls_idx, is_pos = is_bbox_overlap(crop_box, label_boxes)

        if cls_idx >= 0:
            try:
                cls_name = classes[cls_idx]
            except:
                print(f'cls_idx error: {cls_idx} in {label_path}')
                continue
        else:
            cls_name = 'other'

        cropped_img, _ = crop_image_with_bbox(image, (cx, cy, w, h))
        if cropped_img.size == 0:
            continue

        img_save_path = os.path.join(output_root, 'img', f'{current_idx}.jpg')
        npy_save_path = os.path.join(output_root, 'npy', f'{current_idx}.npy')

        cv2.imwrite(img_save_path, cropped_img)
        np.save(npy_save_path, cropped_img)

        file_data.append({
            'index': current_idx,
            'cls_name': cls_name,
            'neg': 0 if is_pos else 1,
            'pos': 1 if is_pos else 0
        })

        current_idx += 1  # 局部递增

    return file_data, current_idx - start_idx  # 返回数据和实际生成的数量


def process_video_folder(video_id, pred_root, label_root, output_root, classes):
    """并行处理单个视频文件夹"""
    pred_dir = os.path.join(pred_root, video_id)
    pred_txts = glob(os.path.join(pred_dir, 'rgb/txts', '*.txt'))

    if not pred_txts:
        return [], 0

    # 预计算每个文件可能生成的图像数量
    file_indices = []
    current_idx = 1
    for txt in pred_txts:
        with open(txt, 'r') as f:
            line_count = len(f.readlines())
        file_indices.append((txt, current_idx))
        current_idx += line_count

    # 创建进程池
    num_workers = min(cpu_count(), len(pred_txts))

    # 准备参数
    partial_func = partial(process_single_file,
                           video_id=video_id,
                           label_root=label_root,
                           output_root=output_root,
                           classes=classes)

    # 并行处理
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(partial_func, file_indices),
                            total=len(pred_txts),
                            desc=f'Processing {video_id}'))

    # 合并结果
    all_data = []
    total_generated = 0
    for r, count in results:
        all_data.extend(r)
        total_generated += count

    return all_data, total_generated


def process_data_parallel(pred_root, label_root, output_root, classes):
    """并行化处理主函数"""
    # 创建输出目录
    os.makedirs(os.path.join(output_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'npy'), exist_ok=True)

    # 获取所有视频文件夹
    video_folders = sorted([d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))])

    # 预分配索引范围
    global_start_idx = 1
    all_data = []
    total_files = 0

    # 先扫描所有文件确定总范围
    for video_id in video_folders:
        pred_dir = os.path.join(pred_root, video_id)
        pred_txts = glob(os.path.join(pred_dir, 'rgb/txts', '*.txt'))
        for txt in pred_txts:
            with open(txt, 'r') as f:
                total_files += len(f.readlines())

    # 处理每个视频文件夹
    current_global_idx = 1
    for video_id in video_folders:
        video_data, count = process_video_folder(video_id, pred_root, label_root, output_root, classes)
        all_data.extend(video_data)
        current_global_idx += count

    # 按索引排序确保顺序一致
    all_data.sort(key=lambda x: x['index'])

    # 保存CSV
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_root, 'label.csv'), index=False)

    # 验证数量一致性
    img_count = len(glob(os.path.join(output_root, 'img', '*.jpg')))
    npy_count = len(glob(os.path.join(output_root, 'npy', '*.npy')))
    csv_count = len(df)

    print(f"\n验证结果:")
    print(f"图片数量: {img_count}")
    print(f"NPY数量: {npy_count}")
    print(f"CSV记录: {csv_count}")

    if img_count == npy_count == csv_count:
        print("✅ 所有文件数量一致")
    else:
        print("❌ 文件数量不一致，请检查")


if __name__ == '__main__':
    classes = [
        'himars_laucher', 'hismars_carrier', 'jinbei', 'jinbei-fake',
        'haimars_laucher-fake', 'car', 'LT-2000', 'tianbing-fasheche',
        'aiguozhe-fasheche', 'tiangong-fasheche', 'tianbing-leidache',
        'aiguozhe-leidache', 'tiangong-leidache', 'tianbing-zhihuiche',
        'aiguozhe-zhihuiche', 'tiangong-zhihuiche', 'hanma', 'szb',
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

        pred_dir = f'/media/linana/1276341C76340351/csz/SmallTargetDetector/build/test/0816_JD_model/JDModel_{part}'
        label_dir = f'/media/linana/1276341C76340351/yanqing_orig_data/{part}'
        output_dir = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0816_second_classifier_dataset_0.5_test/JDModel_{part}_second'

        process_data_parallel(pred_dir, label_dir, output_dir, classes)

        part_time = time.time() - part_start
        print(f"\n{'=' * 40}\n完成处理 part: {part} | 耗时: {part_time:.2f}秒\n{'=' * 40}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 40}\n所有处理完成 | 总耗时: {total_time:.2f}秒\n{'=' * 40}")

    '''单个文件生成'''
    # pred_root = f'/media/linana/1276341C76340351/csz/SmallTargetDetector/build/test/output'  # 预测结果根目录
    # label_root = f'/media/linana/1276341C76340351/yanqing_orig_data/0727_t18_101_background'  # 标签数据根目录
    # output_root = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0721_second_classifier_dataset/JDModel_0727_t18_101_background_second'  # 输出根目录
    #
    # process_data(pred_root, label_root, output_root)
