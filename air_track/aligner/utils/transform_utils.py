import math
import cv2
import skimage
import numpy as np


"""
根据各种方法得到的transform矩阵
对当前帧warpAffine，返回与上一帧相同坐标系下的图像
"""


def gen_transform(shape, dx, dy, angle):
    """在当前给定尺寸下，根据旋转角度和平移量生成仿射变换矩阵"""
    w, h = shape  # 图像尺寸

    # 计算中心点（浮点坐标）
    center = (w / 2.0, h / 2.0)

    # 生成旋转矩阵（角度为度数，缩放因子为1.0）
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 调整平移分量
    transform[0, 2] += dx  # x方向平移
    transform[1, 2] += dy  # y方向平移

    return transform


def img_apply_transform(image, transform):
    """将仿射变换矩阵应用到图像，图像尺寸不变"""
    orig_height, orig_width = image.shape[:2]  # 原图尺寸
    # 应用仿射变换，使用逆映射并保持输出尺寸与原图一致
    image_aligned = cv2.warpAffine(
        image,
        transform[:2, :],
        (orig_width, orig_height)
    )

    return image_aligned


def build_geom_transform_predict(
        translation_x,
        translation_y,
        scale_x=1.0,
        scale_y=1.0,
        angle=0.0,
        shear=0.0,
        hflip=False,
        vflip=False,
        return_params=False
):
    """
    说明：构建几何变换的转换矩阵

    @param translation_x: x方向上的平移
    @param translation_y: y方向上的平移
    @param scale_x: 沿 x 轴的缩放因子，默认为 1.0，不缩放
    @param scale_y: 沿 y 轴的缩放因子，默认为 1.0，不缩放
    @param angle: 旋转角度（以度为单位），默认为 0.0
    @param shear: 裁剪角度（以度为单位），默认为 0.0
    @param hflip: 布尔值，是否进行水平翻转
    @param vflip: 布尔值，是否进行垂直翻转
    @param return_params: transform matrix 布尔值，是否返回转换矩阵的参数矩阵的伪逆矩阵
    @return: transform matrix 布尔值，是否返回转换矩阵的参数矩阵

    Usage with cv2:

    crop = cv2.warpAffine(img, np.linalg.pinv(tform.params)[:2, :],
                          dsize=(self.frame_size, self.frame_size), flags=cv2.INTER_LINEAR)
    """

    # 左右翻转
    if hflip:
        scale_x *= -1
    # 上下翻转
    if vflip:
        scale_y *= -1

    # 构建仿射变换矩阵
    # 平移转换
    tform = skimage.transform.AffineTransform(translation=(translation_x, translation_y))
    # 缩放转换
    tform = skimage.transform.AffineTransform(scale=(1.0 / scale_x, 1.0 / scale_y)) + tform
    # 旋转和裁剪转换
    tform = skimage.transform.AffineTransform(rotation=angle * math.pi / 180,
                                              shear=shear * math.pi / 180) + tform

    if return_params:
        return np.linalg.pinv(tform.params)  # 返回转换矩阵的参数矩阵的伪逆矩阵
    else:
        return tform  # 直接返回转换矩阵


def build_geom_transform(
        dst_w,
        dst_h,
        src_center_x,
        src_center_y,
        scale_x=1.0,
        scale_y=1.0,
        angle=0.0,
        shear=0.0,
        hflip=False,
        vflip=False,
        return_params=False
):
    """
    @param dst_size_x: 目标图像的宽度
    @param dst_size_y: 目标图像的高度
    @param src_center_x: 源图像中心点的 x 坐标
    @param src_center_y: 源图像中心点的 y 坐标
    @param scale_x: 沿 x 轴的缩放因子，默认为 1.0
    @param scale_y: 沿 y 轴的缩放因子，默认为 1.0
    @param angle: 旋转角度（以度为单位），默认为 0.0
    @param shear: 裁剪角度（以度为单位），默认为 0.0
    @param hflip: 布尔值，是否进行水平翻转
    @param vflip: 布尔值，是否进行垂直翻转
    @return: transform matrix 布尔值，是否返回转换矩阵的参数矩阵

    Usage with cv2:

    crop = cv2.warpAffine(img, np.linalg.pinv(tform.params)[:2, :],
                          dsize=(self.frame_size, self.frame_size), flags=cv2.INTER_LINEAR)

    说明：构建几何变换的转换矩阵，前后两次平移是为了消除黑边
    """

    if hflip:
        scale_x *= -1
    if vflip:
        scale_y *= -1

    # 构建仿射变换矩阵
    # 平移转换
    tform = skimage.transform.AffineTransform(translation=(src_center_x, src_center_y))
    # 缩放转换
    tform = skimage.transform.AffineTransform(scale=(1.0 / scale_x, 1.0 / scale_y)) + tform  # TODO 1.0 / scale_x ？
    # 旋转和裁剪转换
    tform = skimage.transform.AffineTransform(rotation=angle * math.pi / 180,
                                              shear=shear * math.pi / 180) + tform
    # 平移转换
    tform = skimage.transform.AffineTransform(translation=(-dst_w / 2, -dst_h / 2)) + tform

    if return_params:
        return np.linalg.pinv(tform.params)  # 返回转换矩阵的参数矩阵的伪逆矩阵
    else:
        return tform  # 直接返回转换矩阵


def create_points(points_shape, crop_w=640, crop_h=512):
    """创建帧对齐所需的均匀分布的特征点"""
    # prev_points的设定：需要根据模型的输出来指定维度
    prev_points = np.zeros(points_shape, dtype=np.float32)
    '''高在前、宽在后'''
    interval_h = int(crop_h / prev_points.shape[1] + 0.5)
    interval_w = int(crop_w / prev_points.shape[2] + 0.5)
    # 一行的选点
    prev_points[0, :, :] = np.arange(int(interval_w / 2), crop_w, interval_w)[None, :]  # 第一维16-1024
    # 一列的选点
    prev_points[1, :, :] = np.arange(int(interval_h / 2), crop_h, interval_h)[:, None]  # 第二维16-1024

    prev_points_1d = prev_points.reshape((2, -1))  # (2, 1024)

    return prev_points, prev_points_1d


def points_apply_transform(prev_points_1d, tr_params, points_shape, crop_w, crop_h):
    """使用仿射变换矩阵对特征点进行变换"""
    # 仿射变换矩阵：平移，缩放，旋转，斜切
    transform = build_geom_transform(
        dst_w=crop_w,
        dst_h=crop_h,
        src_center_x=crop_w / 2 + tr_params['dx'],
        src_center_y=crop_h / 2 + tr_params['dy'],
        scale_x=tr_params['scale'],
        scale_y=tr_params['scale'],
        angle=tr_params['angle'],
        return_params=True
    )  # (3, 3)
    # TODO cur_points的设定、计算原理，为何是用prev计算cur而不是用cur计算prev？
    cur_points = ((transform[:2, :2] @ prev_points_1d).T + transform[:2, 2]).T

    cur_points = cur_points.reshape(points_shape)

    return cur_points


def transform_img_for_transform_params(cur_img, prev_tr_params, crop_w=1024, crop_h=1024, downscale=1):
    """使用当前帧图像进行仿射变换得到前一帧图像"""
    h, w = cur_img.shape

    # 对当前帧进行仿射变换：平移，缩放，旋转，斜切
    prev_img_transform = build_geom_transform(
        dst_w=w,
        dst_h=h,
        src_center_x=w / 2 + prev_tr_params['dx'],
        src_center_y=h / 2 + prev_tr_params['dy'],
        scale_x=prev_tr_params['scale'],
        scale_y=prev_tr_params['scale'],
        angle=prev_tr_params['angle'],
        return_params=True  # 返回转换矩阵的参数矩阵
    )
    # print(prev_img_transform)
    """
    [[ 1.01367981e+00 -6.00957372e-03  5.03475075e+01]
     [ 6.00957372e-03  1.01367981e+00  1.97982088e+01]
     [ 3.29152918e-18  1.16470534e-17  1.00000000e+00]]
    """
    # downscale = 1 相当于只缩放为1024×1024
    prev_img_transform_scale = build_geom_transform(
        dst_w=crop_w,
        dst_h=crop_h,
        src_center_x=w / 2,
        src_center_y=h / 2,
        scale_x=1.0 / downscale,
        scale_y=1.0 / downscale,
        angle=0,
        return_params=True
    )
    # print(prev_img_transform_scale)
    """
    [[ 1.00000000e+00  1.49233822e-16 -7.12000000e+02]
     [ 8.92889037e-17  1.00000000e+00 -5.12000000e+02]
     [-4.20937562e-17 -3.08223558e-17  1.00000000e+00]]
    """
    combined_tr = prev_img_transform_scale @ np.linalg.pinv(prev_img_transform)
    # print(combined_tr)
    """
    [[ 9.86470131e-01  5.84826187e-03 -7.61782097e+02]
     [-5.84826187e-03  9.86470131e-01 -5.31235896e+02]
     [-4.27575887e-17 -4.65275123e-17  1.00000000e+00]]
    """
    # 使用变换矩阵变换图像，图像会变为指定的尺寸
    prev_img = cv2.warpAffine(
        cur_img,
        combined_tr[:2, :],
        dsize=(crop_w, crop_h),
        # dsize=(w, h),
        flags=cv2.INTER_AREA if downscale > 1 else cv2.INTER_LINEAR)

    return prev_img


def transform_img_size(img, w=640, h=512):
    """使用当前帧图像进行仿射变换得到前一帧图像"""
    img_h, img_w = img.shape[:2]

    if (w, h) > (img_w, img_h):
        # 使用双线性插值方法放大图像
        img_new = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        center_x = img_w // 2
        center_y = img_h // 2
        img_new = img[center_y - h // 2: center_y + h // 2, center_x - w // 2: center_x + w // 2]

    return img_new


if __name__ == "__main__":
    """用于测试阶段使用，现作为展示代码，若需单独使用，将下段代码拿出去到单独脚本中使用，莫要修改此文件"""
    cur_img_path = '/home/csz_changsha/data/Aot_small_640_512/part1/Images/part100bb96a5a68f4fa5bc5c5dc66ce314d2/157304365656683977300bb96a5a68f4fa5bc5c5dc66ce314d2.png'

    cur_img = cv2.imread(cur_img_path, cv2.IMREAD_UNCHANGED)

    # w, h = 1024, 1024
    # prev_tr_params = dict(scale=0.98, angle=0.2, dx=10, dy=20)
    #
    # cur_img = transform_img_size(cur_img, w, h)
    # prev_img = transform_img_for_transform_params(cur_img, prev_tr_params)

    img = gen_transform_img(cur_img, angle=10, dx=10, dy=10, target_w=640, target_h=512)

    cv2.imwrite('1.png', cur_img)
    cv2.imwrite('2.png', img)
