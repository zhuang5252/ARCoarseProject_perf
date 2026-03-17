import copy
import math
import os

import torch
import numpy as np
import pandas as pd
from imgaug import BoundingBox, BoundingBoxesOnImage
from air_track.detector.data.base_dataset.gaussian_render import gaussian2D


def xyxy2xywh(box):
    box_xywh = []

    box_xywh.append((box[0] + box[2]) / 2)  # x center
    box_xywh.append((box[1] + box[3]) / 2)  # y center
    box_xywh.append((box[2] - box[0]))  # width
    box_xywh.append((box[3] - box[1]))  # height
    return box_xywh


def xywh2xyxy(box):
    box_xyxy = []

    box_xyxy.append(box[0] - box[2] / 2)  # x left
    box_xyxy.append(box[1] - box[3] / 2)  # y top
    box_xyxy.append(box[0] + box[2] / 2)  # x right
    box_xyxy.append(box[1] + box[3] / 2)  # y bottom
    return box_xyxy


def xyxy2cxcywhn(box, w=640, h=640):
    box_cxcywh = copy.deepcopy(box)

    box_cxcywh[0] = (box[0] + box[2]) / 2 / w  # x center
    box_cxcywh[1] = (box[1] + box[3]) / 2 / h  # y center
    box_cxcywh[2] = (box[2] - box[0]) / w  # width
    box_cxcywh[3] = (box[3] - box[1]) / h  # height
    return box_cxcywh


def cxcywhn2xyxy(box, img_w=640, img_h=640, min_w=0, min_h=0):
    box_xyxy = []
    cx = box[0] * img_w
    cy = box[1] * img_h
    w = max(box[2] * img_w, min_w)
    h = max(box[3] * img_h, min_h)

    box_xyxy.append(cx - w / 2)  # x left
    box_xyxy.append(cy - h / 2)  # y top
    box_xyxy.append(cx + w / 2)  # x right
    box_xyxy.append(cy + h / 2)  # y bottom
    return box_xyxy


def is_all_nan(lst):
    # 将列表转换为numpy数组
    array = np.array(lst)
    # 使用numpy的isnan函数检查每个元素是否为nan
    return np.isnan(array).all()


def read_flight_id_csv(cfg, data_dir, flight_id, part):
    """从 groundtruth.csv 文件中提取指定 flight_id 所需信息"""
    img_folder = cfg['img_folder']
    label_file = cfg['label_file']
    img_format = cfg['img_format']

    label_path = f'{data_dir}/{part}/{label_file}'
    if not os.path.exists(label_path):
        print(f'{label_path} not exists')
        return None, None, None

    df = pd.read_csv(label_path)

    # 支持类别是id和cls_name两种key
    if 'cls_name' in df.keys():
        cls_key = 'cls_name'
    elif 'id' in df.keys():
        cls_key = 'id'
    else:
        raise ValueError('No cls_name or id in the csv file')

    df_flight = df[df.flight_id == flight_id][
        [
            'img_name', cls_key, 'frame', 'gt_left', 'gt_right', 'gt_top', 'gt_bottom', 'range_distance_m'
        ]].drop_duplicates().reset_index(
        drop=True)  # .drop_duplicates()去除重复的行
    frame_numbers = df_flight.frame.values
    file_names = [f'{fn[:-4]}{img_format}' for fn in df_flight.img_name]

    return df_flight, frame_numbers, file_names


def combine_images(prev_tensor_images: list, cur_tensor_img):
    """拼接待输入模型的几帧图像"""
    data = prev_tensor_images + [cur_tensor_img]
    data = torch.cat(data, dim=1)

    return data


def merge_dataset(batch):
    """合并新旧数据"""
    data = {}
    for k in batch[0].keys():
        for x in batch:
            # tensor合并
            if isinstance(x[k], torch.Tensor):
                data[k] = torch.stack([x[k] for x in batch])
            # 字符串合并
            elif isinstance(x[k], str) or isinstance(x[k], int) or isinstance(x[k], list):
                if k not in data:
                    data[k] = []
                data[k].append(x[k])
            else:
                raise RuntimeError('Unsupported data type: ' + str(type(x[k])))

    return data


def argmax2d(x):
    return np.unravel_index(x.argmax(), x.shape)


def argmin2d(x):
    return np.unravel_index(x.argmin(), x.shape)


def calc_iou_single_img(obj_1, obj_2):
    x1 = obj_1['cx']
    y1 = obj_1['cy']
    w1 = obj_1['w']
    h1 = obj_1['h']

    x2 = obj_2['cx']
    y2 = obj_2['cy']
    w2 = obj_2['w']
    h2 = obj_2['h']

    x_min = max(x1 - w1 / 2, x2 - w2 / 2)
    y_min = max(y1 - h1 / 2, y2 - h2 / 2)
    x_max = min(x1 + w1 / 2, x2 + w2 / 2)
    y_max = min(y1 + h1 / 2, y2 + h2 / 2)

    w = max(x_max - x_min, 0.)
    h = max(y_max - y_min, 0.)

    intersections = w * h
    unions = (obj_1['w'] * obj_1['h'] + obj_2['w'] * obj_2['h'] - intersections)

    iou = intersections / unions
    return iou


def calc_iou_multi_frame(obj_1, obj_2):
    x1 = obj_1['cx'] + obj_1['offset'][0]
    y1 = obj_1['cy'] + obj_1['offset'][1]
    w1 = obj_1['w']
    h1 = obj_1['h']

    x2 = obj_2['cx'] + obj_2['offset'][0]
    y2 = obj_2['cy'] + obj_2['offset'][1]
    w2 = obj_2['w']
    h2 = obj_2['h']

    x_min = max(x1 - w1 / 2, x2 - w2 / 2)
    y_min = max(y1 - h1 / 2, y2 - h2 / 2)
    x_max = min(x1 + w1 / 2, x2 + w2 / 2)
    y_max = min(y1 + h1 / 2, y2 + h2 / 2)

    w = max(x_max - x_min, 0.)
    h = max(y_max - y_min, 0.)

    intersections = w * h
    unions = (obj_1['w'] * obj_1['h'] + obj_2['w'] * obj_2['h'] - intersections)

    iou = intersections / unions
    return iou


def pred_to_detections_2_output(
        classes,
        comb_pred,
        size,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.025,
        down_scale: int = 2,
        target_min_size: int = 2,
        x_pad: int = 0,
        y_pad: int = 0
):
    comb_pred = comb_pred.copy()
    res_detections = []

    # comb_pred = comb_pred[0]
    h4, w4 = comb_pred.shape

    while True:
        y, x = argmax2d(comb_pred)
        # conf = comb_pred[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].sum()
        conf = comb_pred[y, x]
        # print('conf', conf)
        if conf < conf_threshold or np.isnan(conf):
            break

        w = 2 ** size[0, y, x]
        h = 2 ** size[1, y, x]

        # 按照输出图的大小确定最大值（值为图像尺寸 / pred_scale），后续如果尺寸多变，需进一步优化
        w = min(w4, max(target_min_size, w))
        h = min(h4, max(target_min_size, h))
        # item_cls = cls[:, y, x]

        cx_img = x
        cy_img = y
        cx = (x + 0.5) * down_scale
        cy = (y + 0.5) * down_scale

        new_item = dict(
            conf=conf,
            cls='obj',
            cx=cx + x_pad,
            cy=cy + y_pad,
            w=w,
            h=h,
        )

        overlaps = False
        for prev_item in res_detections:
            if calc_iou_single_img(prev_item, new_item) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            res_detections.append(new_item)

        w = math.ceil(w * 2 / down_scale)
        h = math.ceil(h * 2 / down_scale)
        w = max(1, w // 2 * 2 + 1)
        h = max(1, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 2, sigma_y=h / 2)

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        comb_pred[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w] -= item_mask

    return res_detections


def pred_to_detections_3_output(
        classes,
        comb_pred,
        cls,
        size,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.025,
        down_scale: int = 2,
        target_min_size: int = 2,
        x_pad: int = 0,
        y_pad: int = 0
):
    comb_pred = comb_pred.copy()
    res_detections = []

    # comb_pred = comb_pred[0]
    h4, w4 = comb_pred.shape

    while True:
        y, x = argmax2d(comb_pred)
        # conf = comb_pred[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].sum()
        conf = comb_pred[y, x]
        # print('conf', conf)
        if conf < conf_threshold or np.isnan(conf):
            break

        cls_pred = cls[:, y, x]
        cls_idx = np.argmax(cls_pred)
        # cls_softmax = torch.softmax(cls_pred, dim=0)
        # cls_idx = cls_softmax.argmax(dim=0)
        cls_name = classes[cls_idx.item()]

        w = 2 ** size[0, y, x]
        h = 2 ** size[1, y, x]

        # 按照输出图的大小确定最大值（值为图像尺寸 / pred_scale），后续如果尺寸多变，需进一步优化
        w = min(w4, max(target_min_size, w))
        h = min(h4, max(target_min_size, h))
        # item_cls = cls[:, y, x]

        cx_img = x
        cy_img = y
        cx = (x + 0.5) * down_scale
        cy = (y + 0.5) * down_scale

        new_item = dict(
            conf=conf,
            cls=cls_name,
            cx=cx + x_pad,
            cy=cy + y_pad,
            w=w,
            h=h,
        )

        overlaps = False
        for prev_item in res_detections:
            if calc_iou_single_img(prev_item, new_item) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            res_detections.append(new_item)

        w = math.ceil(w * 2 / down_scale)
        h = math.ceil(h * 2 / down_scale)
        w = max(1, w // 2 * 2 + 1)
        h = max(1, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 2, sigma_y=h / 2)

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        comb_pred[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w] -= item_mask

    return res_detections


def pred_to_detections_5_output(
        classes,
        comb_pred,
        offset,
        size,
        tracking,
        distance,
        offset_scale: float = 256.0,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.025,
        down_scale: int = 2,
        target_min_size: int = 2,
        x_pad: int = 0,
        y_pad: int = 0
):
    comb_pred = comb_pred.copy()
    res_detections = []

    # comb_pred = comb_pred[0]
    h4, w4 = comb_pred.shape

    while True:
        y, x = argmax2d(comb_pred)
        # conf = comb_pred[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].sum()
        conf = comb_pred[y, x]
        # print('conf', conf)
        if conf < conf_threshold or np.isnan(conf):
            break

        w = 2 ** size[0, y, x]
        h = 2 ** size[1, y, x]
        d = 2 ** distance[y, x]

        # 按照输出图的大小确定最大值（值为图像尺寸 / pred_scale），后续如果尺寸多变，需进一步优化
        w = min(w4, max(target_min_size, w))
        h = min(h4, max(target_min_size, h))
        # item_cls = cls[:, y, x]
        item_tracking = tracking[:, y, x] * offset_scale

        cx_img = x
        cy_img = y
        cx = (x + 0.5) * down_scale
        cy = (y + 0.5) * down_scale

        new_item = dict(
            conf=conf,
            cls='obj',
            cx=cx + x_pad,
            cy=cy + y_pad,
            w=w,
            h=h,
            distance=d,
            tracking=list(item_tracking),
            offset=list(offset[:, y, x] * down_scale),
        )

        overlaps = False
        for prev_item in res_detections:
            if calc_iou_multi_frame(prev_item, new_item) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            res_detections.append(new_item)

        w = math.ceil(w * 2 / down_scale)
        h = math.ceil(h * 2 / down_scale)
        w = max(1, w // 2 * 2 + 1)
        h = max(1, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 2, sigma_y=h / 2)

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        comb_pred[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w] -= item_mask

    return res_detections


def pred_to_detections_6_output(
        classes,
        comb_pred,
        cls,
        offset,
        size,
        tracking,
        distance,
        offset_scale: float = 256.0,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.025,
        down_scale: int = 2,
        target_min_size: int = 2,
        x_pad: int = 0,
        y_pad: int = 0
):
    comb_pred = comb_pred.copy()
    res_detections = []

    # comb_pred = comb_pred[0]
    h4, w4 = comb_pred.shape

    while True:
        y, x = argmax2d(comb_pred)
        # conf = comb_pred[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].sum()
        conf = comb_pred[y, x]
        # print('conf', conf)
        if conf < conf_threshold or np.isnan(conf):
            break

        cls_pred = cls[:, y, x]
        cls_idx = np.argmax(cls_pred)
        # cls_softmax = torch.softmax(cls_pred, dim=0)
        # cls_idx = cls_softmax.argmax(dim=0)
        cls_name = classes[cls_idx.item()]

        w = 2 ** size[0, y, x]
        h = 2 ** size[1, y, x]
        d = 2 ** distance[y, x]

        # 按照输出图的大小确定最大值（值为图像尺寸 / pred_scale），后续如果尺寸多变，需进一步优化
        w = min(w4, max(target_min_size, w))
        h = min(h4, max(target_min_size, h))
        # item_cls = cls[:, y, x]
        item_tracking = tracking[:, y, x] * offset_scale

        cx_img = x
        cy_img = y
        cx = (x + 0.5) * down_scale
        cy = (y + 0.5) * down_scale

        new_item = dict(
            conf=conf,
            cls=cls_name,
            cx=cx + x_pad,
            cy=cy + y_pad,
            w=w,
            h=h,
            distance=d,
            tracking=list(item_tracking),
            offset=list(offset[:, y, x] * down_scale),
        )

        overlaps = False
        for prev_item in res_detections:
            if calc_iou_multi_frame(prev_item, new_item) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            res_detections.append(new_item)

        w = math.ceil(w * 2 / down_scale)
        h = math.ceil(h * 2 / down_scale)
        w = max(1, w // 2 * 2 + 1)
        h = max(1, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 2, sigma_y=h / 2)

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        comb_pred[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w] -= item_mask

    return res_detections


def check_boundary(bbox_xyxy, img_size_w, img_size_h, orig_w, orig_h):
    """检查bbox_xyxy边界条件并调整"""
    x1, y1, x2, y2 = bbox_xyxy
    if not (x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h):
        return bbox_xyxy

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x2)
    y2 = min(orig_h, y2)

    if x1 == 0:
        x2 = x1 + img_size_w
    if y1 == 0:
        y2 = y1 + img_size_h
    if x2 == orig_w:
        x1 = x2 - img_size_w
    if y2 == orig_h:
        y1 = y2 - img_size_h

    bbox_xyxy_new = (x1, y1, x2, y2)

    return bbox_xyxy_new


def check_boundary_norm(bbox_norm_xyxy):
    """检查归一化的bbox_xyxy边界条件并调整"""
    x1, y1, x2, y2 = bbox_norm_xyxy
    if not (x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1):
        return bbox_norm_xyxy

    # 平移处理（保持尺寸不变）
    if x1 < 0:
        offset = -x1
        x1 += offset
        x2 += offset
    elif x2 > 1:
        offset = x2 - 1
        x1 -= offset
        x2 -= offset

    if y1 < 0:
        offset = -y1
        y1 += offset
        y2 += offset
    elif y2 > 1:
        offset = y2 - 1
        y1 -= offset
        y2 -= offset

    bbox_norm_xyxy_new = (x1, y1, x2, y2)

    return bbox_norm_xyxy_new


def yolo_to_imgaug_bbs(labels, img_shape):
    """YOLO格式转imgaug边界框"""
    bbs = []
    for label in labels:
        cx, cy, w, h = map(float, label[1:5])
        x1 = (cx - w / 2) * img_shape[1]
        y1 = (cy - h / 2) * img_shape[0]
        x2 = (cx + w / 2) * img_shape[1]
        y2 = (cy + h / 2) * img_shape[0]
        bbs.append(BoundingBox(
            x1=max(0, x1),
            y1=max(0, y1),
            x2=min(img_shape[1], x2),
            y2=min(img_shape[0], y2),
            label=label
        ))
    return BoundingBoxesOnImage(bbs, shape=img_shape)


def imgaug_to_yolo(bbs, img_shape):
    """imgaug格式转换回YOLO格式"""
    labels = []
    for bb in bbs:
        orig_label = bb.label
        # 处理越界
        x1 = max(0, bb.x1)
        y1 = max(0, bb.y1)
        x2 = min(img_shape[1], bb.x2)
        y2 = min(img_shape[0], bb.y2)

        w = (x2 - x1) / img_shape[1]
        h = (y2 - y1) / img_shape[0]
        cx = ((x1 + x2) / 2) / img_shape[1]
        cy = ((y1 + y2) / 2) / img_shape[0]

        new_label = [
            orig_label[0],  # 保持原始类别
            f"{cx:.6f}",
            f"{cy:.6f}",
            f"{w:.6f}",
            f"{h:.6f}"
        ]
        labels.append(new_label)

    return labels


def cxcywhn_to_imgaug_bbs(labels, img_shape):
    """YOLO格式转imgaug边界框"""
    bbs = []
    for _label in labels:
        label = [_label['cx'], _label['cy'], _label['w'], _label['h']]
        cx, cy, w, h = map(float, label)
        x1 = (cx - w / 2) * img_shape[1]
        y1 = (cy - h / 2) * img_shape[0]
        x2 = (cx + w / 2) * img_shape[1]
        y2 = (cy + h / 2) * img_shape[0]
        bbs.append(BoundingBox(
            x1=max(0, x1),
            y1=max(0, y1),
            x2=min(img_shape[1], x2),
            y2=min(img_shape[0], y2),
            label=label
        ))
    return BoundingBoxesOnImage(bbs, shape=img_shape)


def imgaug_to_cxcywhn(bbs, img_shape):
    """imgaug格式转换回YOLO格式"""
    labels = []
    for bb in bbs:
        orig_label = bb.label
        # 处理越界
        x1 = max(0, bb.x1)
        y1 = max(0, bb.y1)
        x2 = min(img_shape[1], bb.x2)
        y2 = min(img_shape[0], bb.y2)

        w = (x2 - x1) / img_shape[1]
        h = (y2 - y1) / img_shape[0]
        cx = ((x1 + x2) / 2) / img_shape[1]
        cy = ((y1 + y2) / 2) / img_shape[0]

        new_label = [
            # orig_label[0],  # 保持原始类别
            f"{cx:.6f}",
            f"{cy:.6f}",
            f"{w:.6f}",
            f"{h:.6f}"
        ]
        labels.append(new_label)

    return labels


def augment_single(seq, img, labels):
    """辅助函数：应用增强到单个图像和标注"""
    # 转换边界框
    bbs = cxcywhn_to_imgaug_bbs(labels, img.shape)
    # 应用增强
    img_aug = seq.augment_image(img)
    bbs_aug = seq.augment_bounding_boxes([bbs])[0]
    # 转换回 cxcywhn 格式
    labels_aug = imgaug_to_cxcywhn(bbs_aug.bounding_boxes, img_aug.shape)
    # 返回增强后的图像和标注
    return img_aug, labels_aug


def convert_detections(detected_objects, scale_w, scale_h):
    """模型输入与原图尺寸不同形成的scale，转换检测结果到原图基准下"""
    for item in detected_objects:
        item['cx'] /= scale_w
        item['cy'] /= scale_h
        item['w'] /= scale_w
        item['h'] /= scale_h

    return detected_objects
