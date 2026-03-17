import numpy as np


def apply_transform(detection, transform, img_size_w, img_size_h):
    """使用transform对单个detection进行变换"""
    w = detection['w'] * img_size_w
    h = detection['h'] * img_size_h

    # 计算左上角
    x = detection['cx'] * img_size_w - w / 2
    y = detection['cy'] * img_size_h - h / 2

    # 在bbox的每个边上均匀分成3份取2个点
    points = np.array([
        [x, y + h / 3],
        [x, y + h * 2 / 3],
        [x + w, y + h / 3],
        [x + w, y + h * 2 / 3],
        [x + w / 3, y],
        [x + w * 2 / 3, y],
        [x + w / 3, y + h],
        [x + w * 2 / 3, y + h],
    ])

    # 与帧对齐的计算方式类似
    points = (transform[:2, :2] @ points.T).T + transform[:2, 2]

    p0 = np.min(points, axis=0)
    p1 = np.max(points, axis=0)

    # 使用p0和p1重新计算中心点坐标cx、cy和宽、高
    detection['cx'] = (p0[0] + p1[0]) / 2 / img_size_w
    detection['cy'] = (p0[1] + p1[1]) / 2 / img_size_h
    detection['w'] = (p1[0] - p0[0]) / img_size_w
    detection['h'] = (p1[1] - p0[1]) / img_size_h

    return detection
