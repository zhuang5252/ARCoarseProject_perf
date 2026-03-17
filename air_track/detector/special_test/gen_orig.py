import os
import cv2
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
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def is_point_in_boxes(x, y, boxes):
    """判断点(x,y)是否在任意一个box内"""
    for box in boxes:
        cls_idx, cx, cy, w, h = box
        # TODO 过滤szb目标
        # if cls_idx == 17:
        #     continue
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        if x1 <= x <= x2 and y1 <= y <= y2:
            return int(cls_idx), True
    return -1, False


def is_bbox_overlap(pred_bbox, boxes, threshold=0.3):
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
    _, pred_cx, pred_cy, pred_w, pred_h = pred_bbox
    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2
    pred_area = pred_w * pred_h

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


def process_data(pred_root, label_root, output_root):
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
        label_dir = os.path.join(label_root, video_id, 'label')
        rgb_dir = os.path.join(label_root, video_id, 'rgb')

        if not os.path.exists(label_dir) or not os.path.exists(rgb_dir):
            continue

        # 遍历预测的txt文件
        pred_txts = glob(os.path.join(pred_dir, 'rgb', 'txts', '*.txt'))
        for pred_txt in tqdm(pred_txts, desc=f'Processing {video_id}'):
            # 获取对应的图像文件名
            base_name = os.path.splitext(os.path.basename(pred_txt))[0]
            img_name = base_name + '.jpg'  # 假设图像是jpg格式
            img_path = os.path.join(rgb_dir, img_name)

            # 获取对应的标签文件名
            label_name = base_name + '.txt'
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
            for i, (cls, cx, cy, w, h) in enumerate(pred_boxes):
                # 判断是正样本还是负样本
                cls_idx, is_pos = is_bbox_overlap(cx, cy, label_boxes)

                if cls_idx >= 0:
                    try:
                        cls_name = classes[cls_idx]
                    except:
                        cls_name = 'other'
                        print(label_path)
                        continue

                        # def is_point_in1_boxes(x, y, boxes):
                        #     """判断点(x,y)是否在任意一个box内"""
                        #     for box in boxes:
                        #         cls_idx, cx1, cy1, w, h = box
                        #         # TODO 过滤szb目标
                        #         if cls_idx == 17:
                        #             continue
                        #         x1, y1 = cx1 - w / 2, cy1 - h / 2
                        #         x2, y2 = cx1 + w / 2, cy1 + h / 2
                        #         if x1 <= x <= x2 and y1 <= y <= y2:
                        #             return int(cls_idx), True
                        #     return -1, False
                        #
                        # is_point_in1_boxes(cx, cy, label_boxes)
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
              # 'szb'
    ]
    # 配置路径
    pred_root = '/media/linana/1276341C76340351/csz/SmallTargetDetector/build/test/T18_model/0716_JD_no_background'  # 预测结果根目录
    label_root = '/media/linana/1276341C76340351/yanqing_orig_data/0716_JD_no_background'  # 标签数据根目录
    output_root = '/media/linana/2C5821EF5821B88A1/yanqing_train_data/second_classifier_dataset/T18Model_0716_JD_no_background_second'  # 输出根目录

    process_data(pred_root, label_root, output_root)