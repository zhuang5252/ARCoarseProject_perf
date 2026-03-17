import copy
import math
import numpy as np


def gaussian2D(shape, sigma_x, sigma_y):
    """
    生成一个二维高斯（正态）分布的矩阵
    图像处理中的高斯模糊或生成用于关键点检测的高斯响应图。
    标记中心点附近为正样本区域

    sigma_x: x方向的标准差
    sigma_y: y方向的标准差
    """
    m, n = [(ss - 1.) / 2. for ss in shape]  # m、n为目标框的高、宽
    y, x = np.ogrid[-m:m + 1, -n:n + 1]  # 创建网格：均匀分布的(5,1)和(1,5)

    h = np.exp(-x * x / (2 * sigma_x * sigma_x) - y * y / (2 * sigma_y * sigma_y))  # (5,5)

    h[h < 1e-4] = 0
    cy = (shape[0] - 1) // 2  # 找目标的中心点
    cx = (shape[1] - 1) // 2    # 找目标的中心点

    # h[cy, cx] = 1.0   # 目标中心点的附近3x3的区域为正样本区域，现修改为1×1
    h[cy - 1:cy + 2, cx - 1:cx + 2] = 1.0

    # 返回二维高斯分布矩阵、y坐标矩阵、x坐标矩阵
    return h, y + np.zeros_like(x), x + np.zeros_like(y)


def denormalize_detection(detection, img_w, img_h):
    """反归一化"""
    detection['cx'] = detection['cx'] * img_w
    detection['cy'] = detection['cy'] * img_h
    detection['w'] = detection['w'] * img_w
    detection['h'] = detection['h'] * img_h

    return detection


def render_y(cfg, prev_steps_detections, cur_step_detections, img_w, img_h,
             down_scale, target_min_size, non_important_items_scale=1.0):
    """生成模型所用的标签"""
    w4 = img_w // down_scale  # w = 256 -> 1024
    h4 = img_h // down_scale  # h = 200 -> 1024
    # print(w4, h4)       # w4=256/8=32  h4=200/8=25

    mask = np.zeros((1, h4, w4), dtype=np.float32)  # (7, 512, 512)
    mask_planned = np.zeros((1, h4, w4), dtype=np.float32)  # (7, 512, 512)

    reg_cls = np.zeros((cfg['nb_classes'], h4, w4), dtype=np.float32)  # (7, 512, 512)
    reg_cls_planned = np.zeros((cfg['nb_classes'], h4, w4), dtype=np.float32)  # (7, 512, 512)

    # print(cls.shape, cls_planned.shape)   # (7, 25, 32) (7, 25, 32)  7类，用于渲染类别 zhaoliang

    reg_size_mask = np.zeros((h4, w4), dtype=np.float32)  # (512, 512)
    reg_size = np.zeros((2, h4, w4), dtype=np.float32)  # (2, 512, 512)
    # print(reg_size_mask.shape, reg_size.shape)  # (25, 32) (2, 25, 32)  2，用于预测w，h

    reg_offset = np.zeros((2, h4, w4), dtype=np.float32)
    reg_offset_mask = np.zeros((h4, w4), dtype=np.float32)
    # print(reg_offset.shape, reg_offset_mask.shape)  # (2, 25, 32) (25, 32) 2，用于中心点x，y方向的offset

    reg_tracking = np.zeros((2, h4, w4), dtype=np.float32)
    reg_tracking_mask = np.zeros((h4, w4), dtype=np.float32)
    # print(reg_tracking.shape, reg_tracking_mask.shape)  # (2, 25, 32) (25, 32)  2，用于预测track

    reg_distance = np.zeros((h4, w4), dtype=np.float32)
    reg_distance_mask = np.zeros((h4, w4), dtype=np.float32)
    # print(reg_distance.shape, reg_distance_mask.shape)  # (25, 32) (25, 32) 用于预测距离

    for normalized_item in cur_step_detections:
        # 反归一化得到正常数值
        item = copy.deepcopy(normalized_item)
        item = denormalize_detection(item, img_w, img_h)

        distance = item['distance']
        have_distance = not np.isnan(distance)  # 是否有距离值
        # item_important = have_distance and (distance < config.UPPER_BOUND_MAX_DIST_SELECTED_TRAIN)  # 距离小于1000
        item_important = have_distance  # 是否重要

        cx = item['cx']         # 中心点坐标（还是没有根据图像新尺寸变换的状态）
        cy = item['cy']         #

        cx_img = math.floor(cx / down_scale)   # 中心点坐标x放缩   向下取整
        cy_img = math.floor(cy / down_scale)   # 中心点坐标y放缩

        # 最小目标框，按照pred_scale来设置
        orig_w = max(target_min_size, item['w'])             # 原始目标框w最小为10
        orig_h = max(target_min_size, item['h'])             # 原始目标框h最小为10
        # print(orig_w, orig_h)                # 10.0 10.0

        w = math.ceil(orig_w / down_scale)     # 原始宽高下采样8倍 向上取整
        h = math.ceil(orig_h / down_scale)
        # print(w, h)                          # 5.0 5.0
        w = max(1, w // 2 * 2 + 1)              # 这里保证目标框最小为3x3 -> 5x5
        h = max(1, h // 2 * 2 + 1)
        # print(w, h)                            # 3 3 -> 5 5

        # 计算二维高斯分布的矩阵(h, w为目标框的高、宽)
        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 4, sigma_y=h / 4)
        # print(item_mask.shape, ogrid_y.shape, ogrid_x.shape)
        #  (3, 3) (3, 3) (3, 3) -> (5, 5) (5, 5) (5, 5)

        if not item_important:  # 不重要进分支
            item_mask = item_mask * non_important_items_scale  # 与 1 相乘

        # 获取上边cx_img向下取整后与不取整的偏移量 TODO + 0.5 ？
        x_offset = (cx / down_scale - (cx_img + 0.5))
        y_offset = (cy / down_scale - (cy_img + 0.5))

        # TODO 留个记号，是否需要按照item_mask将3×3修改为1×1
        if (0 < cx_img < w4 - 1) and (0 < cy_img < h4 - 1):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    # 以目标中心辐射的3 * 3矩阵存入针对中心点x、y的偏移量
                    reg_offset[0, cy_img + dy, cx_img + dx] = x_offset - dx
                    reg_offset[1, cy_img + dy, cx_img + dx] = y_offset - dy
                    # 以目标中心辐射的reg_offset_mask 的3*3矩阵为全 1
                    reg_offset_mask[cy_img + dy, cx_img + dx] = 1
        # 取distance的log2
        log_distance = np.log2(distance)

        # clip masks
        w2 = (w - 1) // 2  # 2
        h2 = (h - 1) // 2  # 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        """去除超出左、上边界的值"""
        # 目标中心点cx_img到图像边界不够w2，则从边界位置截断
        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0
        # 目标中心点cy_img到图像边界不够h2，则从边界位置截断
        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        """去除超出右、下边界的值"""
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        if mask_w > 0 and mask_h > 0:
            # 创建切片对象，存入mask图的y的范围和x的范围
            res_slice = np.s_[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w]
            # (slice(92, 97, None), slice(78, 83, None))  # 5*5的item_mask图

            cls_name = item['cls_name']

            '''五个heatmap标签数据目标位置均和item_mask比较，取其最大值更新原矩阵'''
            # 取出目标位置的cls（5*5）
            cls_crop = reg_cls[cfg['classes'].index(cls_name)][res_slice]
            # 对两个矩阵逐元素比较，取其最大值更新cls_crop矩阵，同时会更新cls矩阵
            np.maximum(cls_crop, item_mask, out=cls_crop)

            # if have_distance and item.distance < config.UPPER_BOUND_MAX_DIST * 1.4:
            if have_distance:
                # 取出目标位置的cls_planned（5*5）
                cls_planned_crop = reg_cls_planned[cfg['classes'].index(cls_name)][res_slice]
                # 对两个矩阵逐元素比较，取其最大值更新cls_planned_crop矩阵
                np.maximum(cls_planned_crop, item_mask, out=cls_planned_crop)

            # 取出目标位置的reg_size_mask（5*5）全0
            orig_mask_crop = reg_size_mask[res_slice]  # 全0矩阵
            # 逐个元素比较，返回True或者False（5*5）
            pixels_to_update = item_mask > orig_mask_crop

            # 对两个矩阵逐元素比较，取其最大值更新orig_mask_crop矩阵
            np.maximum(orig_mask_crop, item_mask, out=orig_mask_crop)

            # reg_size[0][res_slice]的(5*5)通过pixels_to_update来确定是否存入np.log2(orig_w/h=10)
            reg_size[0][res_slice][pixels_to_update] = np.log2(orig_w)
            reg_size[1][res_slice][pixels_to_update] = np.log2(orig_h)

            if have_distance:
                # reg_distance[res_slice]的(5*5)通过pixels_to_update来确定是否存入log_distance
                reg_distance[res_slice][pixels_to_update] = log_distance
                distance_map_crop = reg_distance_mask[res_slice]  # 全0矩阵
                # 对两个矩阵逐元素比较，取其最大值更新distance_map_crop矩阵
                np.maximum(distance_map_crop, item_mask, out=distance_map_crop)

            # TODO 标签制作现只支持连续两帧（后续若想多个帧，需考虑下方代码reg_tracking中prev_item['cx'] - item['cx']的设计，多帧图像取均值或者只用前一帧来计算，或者其他）
            best_distance = 1e20
            # 遍历前一帧的标签
            prev_step_detections = prev_steps_detections[-1]  # 现只使用前一帧
            for normalized_prev_item in prev_step_detections:
                # 反归一化得到正常数值
                prev_item = copy.deepcopy(normalized_prev_item)
                prev_item = denormalize_detection(prev_item, img_w, img_h)
                # 前一帧和当前帧的目标类别和target_id均相同，认为两帧是同一个目标（时序上前后两帧的target_id要一一对应）
                if item['cls_name'] == prev_item['cls_name'] and item['target_id'] == prev_item['target_id']:
                    # 计算两帧目标中心点的直线距离
                    distance = ((prev_item['cx'] - item['cx']) ** 2 + (prev_item['cy'] - item['cy']) ** 2) ** 0.5

                    if best_distance < 1e19:
                        print('Found multiple matches')

                    if distance < best_distance:
                        # TODO config.OFFSET_SCALE=256 ？ 其作用 ？
                        reg_tracking[0][res_slice][pixels_to_update] = np.clip(
                            (prev_item['cx'] - item['cx']) / cfg['offset_scale'], -1, 1)
                        reg_tracking[1][res_slice][pixels_to_update] = np.clip(
                            (prev_item['cy'] - item['cy']) / cfg['offset_scale'], -1, 1)

                        best_distance = distance
                        tracking_map_crop = reg_tracking_mask[res_slice]  # 全0矩阵
                        # 对两个矩阵逐元素比较，取其最大值更新tracking_map_crop矩阵
                        np.maximum(tracking_map_crop, item_mask, out=tracking_map_crop)

    # TODO 为什么要将cls用1减去
    mask[0] = np.clip(1.0 - np.sum(reg_cls[:], axis=0), 0.0, 1.0)
    mask_planned[0] = np.clip(1.0 - np.sum(reg_cls_planned[:], axis=0), 0.0, 1.0)

    return dict(
        mask=mask,
        mask_planned=mask_planned,

        cls=reg_cls,

        reg_size=reg_size,
        reg_size_mask=reg_size_mask,

        reg_offset=reg_offset,
        reg_offset_mask=reg_offset_mask,

        reg_tracking=reg_tracking,
        reg_tracking_mask=reg_tracking_mask,

        reg_distance=reg_distance,
        reg_distance_mask=reg_distance_mask,

    )


if __name__ == '__main__':
    h, w = 2, 2
    item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 4, sigma_y=h / 4)