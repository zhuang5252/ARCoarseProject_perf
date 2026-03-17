import csv
import os
import re
import cv2
import math
import yaml
import json
import time
import torch
import shutil
import random
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
import pandas as pd
from pathlib import Path
from importlib import import_module
from imgaug import augmenters as iaa
from contextlib import contextmanager
from matplotlib import pyplot as plt
from collections import defaultdict


def reprod_init(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_yaml(path: str) -> dict:
    """读取yaml配置文件"""
    with open(path, 'rb') as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def load_json(path: str):
    """读取json配置文件"""
    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)  # 返回字典或列表

    return json_data


def load_csv(path: str):
    """读取csv文件"""
    data = pd.read_csv(path)

    return data


def dataset_config_json_to_csv(json_path: str, csv_path: str):
    """
    将JSON配置文件转换为CSV格式

    参数:
        json_path: 输入的JSON文件路径
        csv_path: 输出的CSV文件路径
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备CSV数据
    csv_rows = []

    # 遍历所有part类型(train/val/test)
    for part_name, part_data in data.items():
        if not part_data:  # 跳过空的部分(如part_test)
            continue

        # 遍历每个模型配置
        for model_name, config in part_data.items():
            # 处理可能的引号问题(如"'JDModel_0727_JD_201_second")
            clean_model_name = model_name.strip("'")

            csv_rows.append({
                "part": part_name,
                "data_name": clean_model_name,
                "nums": config["nums"],
                "shuffle": config["shuffle"]
            })

    # 写入CSV文件
    if csv_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["part", "data_name", "nums", "shuffle"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV文件已生成: {Path(csv_path).resolve()}")
    else:
        print("警告: 未找到有效数据")


def dataset_config_csv_to_json(csv_path: str, json_path: str):
    """
    将CSV配置文件转换为JSON格式

    参数:
        json_path: 输入的JSON文件路径
        csv_path: 输出的CSV文件路径
    """
    # 初始化数据结构
    result = {
        "part_train": defaultdict(dict),
        "part_val": defaultdict(dict),
        "part_test": defaultdict(dict)
    }

    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            part = row['part']
            data_name = row['data_name']
            nums = int(row['nums'])
            shuffle = row['shuffle'].lower() == 'true'

            # 将数据添加到对应的部分
            result[part][data_name] = {
                "nums": nums,
                "shuffle": shuffle
            }

    # 将defaultdict转换为普通dict
    for part in result:
        result[part] = dict(result[part])

    # 写入JSON文件
    with open(json_path, mode='w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)

    print(f"json文件已生成: {Path(json_path).resolve()}")


def process_data_parts_config(parts_config_path):
    """一级检测器和二级分类器数据配置文件处理，dataset代码中只接收json文件"""
    if parts_config_path.endswith('.csv'):
        parts_config_csv = parts_config_path
        parts_config_json = parts_config_csv.replace('.csv', '.json')
        dataset_config_csv_to_json(parts_config_csv, parts_config_json)

        return parts_config_json

    elif parts_config_path.endswith('.json'):
        return parts_config_path

    else:
        assert ValueError(f'数据配置文件非csv或json格式：{parts_config_path}')


def combine_load_cfg_yaml(yaml_paths_list: list):
    """合并读取若干个yaml的配置文件内容"""
    cfg_data = None
    for item in yaml_paths_list:
        item_data = load_yaml(item)
        if not cfg_data:
            cfg_data = item_data
        else:
            cfg_data.update(item_data)

    return cfg_data


def round_up_to_nearest_power_of_two(x):
    """向上取整到最近的2的倍数"""
    rounded_up = math.ceil(x)
    # 找到最接近的2的倍数
    if rounded_up % 2 != 0:
        return rounded_up + 1
    else:
        return rounded_up


def img_fn(cfg, part, flight_id, img_name):
    """根据当前帧信息，获取当前帧图像的完整路径并返回"""
    data_dir = cfg['data_dir']
    img_folder = cfg['img_folder']
    img_format = cfg['img_format']

    return f'{data_dir}/{part}/{img_folder}/{flight_id}/{img_name}{img_format}'


@contextmanager  # 将一个生成器函数转换为上下文管理器，使其能够用于 `with` 语句
def timeit_context(name):
    """当使用 with 语句和这个上下文管理器时，它将会在进入和退出代码块时分别记录时间，并在退出时打印出经过的时间"""
    start_time = time.time()
    yield  # 这个关键字将这个函数成为一个生成器函数。它允许程序在进入和退出上下文管理器时执行必要的操作
    elapsed_time = time.time() - start_time
    print(f"[{name}] finished in {elapsed_time:0.3f}s")


def load_model(model, model_path, device='cpu'):
    """读取模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    print('The loaded model epoch is: ', checkpoint['epoch'])
    # 非严格模式加载，允许缺失部分网络或权重
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)

    return model


def get_all_files(path, target_depth, current_depth=1, file_end=None):
    """
    递归找出path路径下的target_depth深度的文件完整路径，并保存列表。
    如果file_end为None，返回所有文件路径，否则只返回以file_end中字符串结尾的文件路径。
    """
    folders_paths = []

    if current_depth == target_depth:
        for f in os.listdir(path):
            temp = os.path.join(path, f)
            if os.path.isfile(temp):
                if file_end is None or any(f.endswith(end) for end in file_end):
                    folders_paths.append(temp)
    else:
        for f in os.listdir(path):
            temp = os.path.join(path, f)
            if os.path.isdir(temp):
                folders_paths.extend(get_all_files(temp, target_depth, current_depth + 1, file_end))

    return folders_paths


def row_data_save_csv(csv_file_path, csv_data, column_labels):
    """按行写入csv文件"""
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_labels)
        writer.writeheader()
        for row in csv_data:
            # 将字典中的所有值转换为字符串
            string_row = {key: str(value) for key, value in row.items()}
            writer.writerow(string_row)

    print(f"CSV file created at {csv_file_path}")


def col_data_save_csv(csv_file_path, data: dict):
    """按列写入csv文件"""
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # 获取所有的键作为列标题
    headers = list(data.keys())

    # 获取每个键对应的值的长度，确保所有列表长度一致
    num_entries = len(next(iter(data.values())))

    # 打开文件准备写入
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(headers)

        # 遍历所有条目，确保每个键的值列表长度一致
        for i in range(num_entries):
            row = []
            for header in headers:
                value = data[header][i]
                # 如果值是NaN，则替换为空字符串
                if isinstance(value, np.float64) and np.isnan(value):
                    value = ''
                row.append(value)
            # 写入当前行的数据
            writer.writerow(row)

    print(f"CSV file created at {csv_file_path}")


def transform_imgs(imgs, transforms, img_size_w=None, img_size_h=None):
    """使用transform对多个prev_image进行变换"""
    imgs_new = []
    for img, key in zip(imgs, transforms):
        if img_size_w is not None and img_size_h is not None:
            orig_height, orig_width = img_size_h, img_size_w
        else:
            orig_height, orig_width = img.shape[:2]
        transform = transforms[key]
        """将仿射变换矩阵应用到图像，图像尺寸不变"""
        # 应用仿射变换，使用逆映射并保持输出尺寸与原图一致
        img_new = cv2.warpAffine(
            img,
            transform[:2, :],
            (orig_width, orig_height)
        )
        imgs_new.append(img_new)

    return imgs_new


def check_and_change_img_size(img, img_size_w=640, img_size_h=512):
    """校对并改变图像尺寸"""
    if img is not None and img.shape[:2] != (img_size_h, img_size_w):
        img = cv2.resize(img, (img_size_w, img_size_h))

    return img


def numpy_2_tensor(img, max_pixel):
    """将numpy的图像数据转为归一化后的tensor数据"""
    # 和模型训练过程中保持一致
    tensor_img = torch.from_numpy(img.astype(np.float32) / max_pixel).float()

    return tensor_img


def extract_number(filename):
    """提取文件名中的数字"""
    match = re.search(r'\d+', filename)

    return int(match.group()) if match else 0


def natural_sort_key(s):
    """匹配文件名中的数字部分（包括前导零），用于文件排序使用"""
    parts = re.split(r'(\d+)', s)
    # 将数字部分转为整数，非数字部分保留原样
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def normalize(data, from_range: list, to_range: list):
    """
    normalize or denormalize

    Args:
        data: data needs to be normalized
        from_range: origin scale
        to_range: target scale

    Returns:
        normalized data
    """
    # 缩放因子
    scale_factor = to_range[1] - to_range[0]
    # 归一化到[0, 1]区间
    data_normalized = (data - from_range[0]) / (from_range[1] - from_range[0])
    # 映射到目标区间
    data_normalized = scale_factor * data_normalized - to_range[0]

    return data_normalized


def from_txt_read_label(label_path):
    """yolo格式的标签读取"""
    # 读取标签
    with open(label_path, 'r') as file:
        labels = file.readlines()
    labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的

    return labels


def copy_split_data(stage, img_paths, img_folder, label_folder, img_format, gt_format, save_dir=None):
    """拷贝拆分后的数据到指定目录"""
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(img_format, gt_format)
        label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), label_folder, label_name)
        if save_dir is not None:
            img_save_path = os.path.join(save_dir, stage, img_folder)
            label_save_path = os.path.join(save_dir, stage, label_folder)
        else:
            img_save_path = os.path.join(img_path.split(img_folder)[0], stage, img_folder)
            label_save_path = os.path.join(label_path.split(label_folder)[0], stage, label_folder)
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(label_save_path, exist_ok=True)
        shutil.copy(img_path, os.path.join(img_save_path, img_name))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(label_save_path, label_name))


def plt_feature_map(tag, feature_map, show_channel=0, cmap='viridis'):
    """
    可视化特征图

    Args:
        tag (str): 图像标题
        feature_map (torch.Tensor | np.ndarray): 特征图，支持2D/3D/4D
        show_channel (int): 多通道时选择显示的通道（默认第0通道）
        cmap (str): 颜色映射（默认'viridis'）
    """
    # 统一转换为numpy数组并处理维度
    if isinstance(feature_map, torch.Tensor):
        data = feature_map.detach().cpu().numpy()
    elif isinstance(feature_map, np.ndarray):
        data = feature_map.copy()
    else:
        raise TypeError("输入必须是torch.Tensor或numpy.ndarray")

    # 处理维度
    if data.ndim == 4:  # [B, C, H, W]
        data = data[0, show_channel]  # 取第0批次的指定通道
    elif data.ndim == 3:  # [C, H, W]
        data = data[show_channel]
    data = np.squeeze(data)  # 压缩单维度

    if data.ndim != 2:
        raise ValueError("特征图必须是2D（H, W）或可压缩至2D")

    # 可视化
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(tag)
    plt.tight_layout()  # 避免标题重叠

    return fig


def build_augmenter(aug_config):
    """构建可配置的数据增强流水线

    参数示例：
    aug_config = {
        # 几何变换
        'flip': {
            'prob': 0.6,                # 总翻转概率
            'hflip_prob': 0.5,           # 水平翻转概率（占总体翻转的比例）
            'vflip_prob': 0.3            # 垂直翻转概率
        },
        'affine': {
            'prob': 0.8,                 # 仿射变换触发概率
            'scale': {'x': [0.8, 1.2], 'y': [0.9, 1.1]},  # 缩放范围
            'translate': {'x': [-0.15, 0.15], 'y': [-0.1, 0.1]}, # 平移比例
            'rotate': [-20, 20],         # 旋转角度范围
            'shear': [-10, 10]           # 剪切角度范围
        },
        # 颜色变换
        'color': {
            'prob': 0.7,                 # 颜色操作总概率
            'hue_shift': [-30, 30],      # 色相调整范围
            'saturation_shift': [-50, 50], # 饱和度调整范围
            'grayscale_alpha': [0.0, 1.0], # 灰度化强度
            'contrast_range': [0.7, 1.5] # 对比度范围
        },
        # 噪声与模糊
        'noise': {
            'prob': 0.4,                 # 噪声操作总概率
            'blur_sigma': [0, 2.0],      # 高斯模糊强度
            'gaussian_noise_scale': [0, 0.1] # 高斯噪声强度（0-255比例）
        }
    }
    """
    augmentors = []

    # ==================== 几何翻转增强 ====================
    flip_cfg = aug_config.get('flip', {})
    if flip_cfg.get('prob', 0) > 0:
        flip_ops = []

        # 水平翻转配置（独立概率控制）
        if flip_cfg.get('hflip_prob', 0) > 0:
            flip_ops.append(
                iaa.Sometimes(
                    flip_cfg['hflip_prob'],  # 水平翻转独立概率
                    iaa.Fliplr(flip_cfg['hflip_prob'])
                )
            )

        # 垂直翻转配置（独立概率控制）
        if flip_cfg.get('vflip_prob', 0) > 0:
            flip_ops.append(
                iaa.Sometimes(
                    flip_cfg['vflip_prob'],  # 垂直翻转独立概率
                    iaa.Flipud(flip_cfg['vflip_prob'])
                )
            )

        if flip_ops:
            augmentors.append(
                iaa.Sometimes(
                    flip_cfg['prob'],  # 总翻转触发概率
                    iaa.OneOf(flip_ops)
                )
            )

    # ==================== 仿射变换增强 ====================
    affine_cfg = aug_config.get('affine', {})
    if affine_cfg.get('prob', 0) > 0:
        augmentors.append(
            iaa.Sometimes(
                affine_cfg['prob'],
                iaa.Affine(
                    scale={
                        'x': affine_cfg.get('scale', {}).get('x', (0.9, 1.1)),
                        'y': affine_cfg.get('scale', {}).get('y', (0.9, 1.1))
                    },
                    translate_percent={
                        'x': affine_cfg.get('translate', {}).get('x', (-0.1, 0.1)),
                        'y': affine_cfg.get('translate', {}).get('y', (-0.1, 0.1))
                    },
                    rotate=affine_cfg.get('rotate', (-8, 8)),
                    shear=affine_cfg.get('shear', (-5, 5))
                )
            )
        )

    # ==================== 颜色空间增强 ====================
    color_cfg = aug_config.get('color', {})
    if color_cfg.get('prob', 0) > 0:
        color_ops = []

        # 色相/饱和度调整（参数分离）
        if 'hue_shift' in color_cfg or 'saturation_shift' in color_cfg:
            # 解析色相范围
            hue_min, hue_max = color_cfg.get('hue_shift', (-30, 30))
            # 解析饱和度范围
            sat_min, sat_max = color_cfg.get('saturation_shift', (-50, 50))

            color_ops.append(
                iaa.AddToHueAndSaturation(
                    value_hue=(hue_min, hue_max),  # 独立色相参数
                    value_saturation=(sat_min, sat_max),  # 独立饱和度参数
                    per_channel=False
                )
            )

        # 灰度化处理
        if color_cfg.get('grayscale_alpha', (0.0, 1.0)):
            color_ops.append(
                iaa.Grayscale(alpha=tuple(color_cfg['grayscale_alpha']))
            )

        # 对比度调整
        if color_cfg.get('contrast_range', (0.8, 1.2)):
            color_ops.append(
                iaa.LinearContrast(alpha=tuple(color_cfg['contrast_range']))
            )

        if color_ops:
            augmentors.append(
                iaa.Sometimes(
                    color_cfg['prob'],
                    iaa.OneOf(color_ops)
                )
            )

    # ==================== 噪声模糊增强 ====================
    noise_cfg = aug_config.get('noise', {})
    if noise_cfg.get('prob', 0) > 0:
        noise_ops = []

        # 高斯模糊
        if noise_cfg.get('blur_sigma', (0.0, 2.0)):
            noise_ops.append(
                iaa.GaussianBlur(sigma=tuple(noise_cfg['blur_sigma']))
            )

        # 高斯噪声
        if noise_cfg.get('gaussian_noise_scale', (0.0, 0.1)):
            scale_range = tuple(int(255 * x) for x in noise_cfg['gaussian_noise_scale'])
            noise_ops.append(iaa.AdditiveGaussianNoise(scale=scale_range))

        if noise_ops:
            augmentors.append(
                iaa.Sometimes(
                    noise_cfg['prob'],
                    iaa.OneOf(noise_ops)
                )
            )

    return iaa.Sequential(augmentors, random_order=True)


def auto_import_module(path, name):
    # 动态导入类
    try:
        base_module = import_module(path)
        module = getattr(base_module, name)
        return module
    except (ImportError, AttributeError) as e:
        raise KeyError(f'Failed to load module: {path}.{name}') from e


def check_image_bit_depth(
    image: np.ndarray,
    img_path: str,
    cfg_max_pixel: int = 255,
    allowed_dtypes: tuple = ('uint8', 'uint16')
) -> None:
    """
    校验图像位深度与配置文件中指定的最大值是否匹配

    Args:
        image: 输入的图像数据（NumPy 数组）
        img_path: 图像路径（仅用于错误提示）
        cfg_max_pixel: 配置文件中的max_pixel值（默认255）
        allowed_dtypes: 允许的图像数据类型（默认支持8/16位）

    Raises:
        ValueError: 当位深度不匹配或配置错误时抛出
        TypeError: 当图像类型不在允许范围内时抛出
    """
    if image is None:
        raise ValueError(f"图像读取失败: {img_path}")

    # 检查是否为允许的数据类型
    if image.dtype.name not in allowed_dtypes:
        raise TypeError(
            f"图像 {img_path} 的数据类型 {image.dtype} 不合法，"
            f"仅支持: {allowed_dtypes}"
        )

    # 定义标准位深度校验规则
    bit_depth_standards = {
        'uint8': np.iinfo(np.uint8).max,
        'uint16': np.iinfo(np.uint16).max
    }

    # 获取配置中的期望最大值
    expected_max = cfg_max_pixel

    # 校验实际位深度与配置是否匹配
    current_max = bit_depth_standards.get(image.dtype.name)
    if current_max is None:
        raise TypeError(f"不支持的图像数据类型: {image.dtype}")

    if expected_max != current_max:
        raise ValueError(
            f"图像位深度与配置不匹配\n"
            f"路径: {img_path}\n"
            f"当前类型: {image.dtype} (max={current_max})\n"
            f"配置要求: max_pixel={expected_max}"
        )


def check_data_is_normalized(data: np.ndarray):
    """校验数据是否是归一化后的数据"""
    if data.dtype == np.uint8:
        return False  # uint8 通常是 [0, 255]
    elif data.dtype in (np.float32, np.float64):
        return (data.min() >= 0 and data.max() <= 1)  # 检查是否在 [0, 1]
    else:
        raise ValueError("Unsupported dtype: {}".format(data.dtype))


def is_center_point_in_boxes(x, y, bbox_xyxy):
    """判断点(x,y)是否在一个box内"""
    x1, y1, x2, y2 = bbox_xyxy
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    return False


def is_bbox_overlap(pred_xyxy, bbox_xyxy, threshold=0.5):
    """
    判断预测框与标签框的交集面积是否超过标签框面积的50%

    参数:
        pred_xyxy: 预测框 (x1, y1, x2, y2)
        bbox_xyxy: 标签框 (x1, y1, x2, y2)
        threshold: 面积比例阈值 (默认0.5)

    返回:
        (matched_cls, is_overlap):
            matched_cls: 匹配的类别ID (若无匹配返回-1)
            is_overlap: 是否满足阈值条件
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_xyxy
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

    # 计算标签框坐标和面积
    box_x1, box_y1, box_x2, box_y2 = bbox_xyxy
    w = box_x2 - box_x1
    h = box_y2 - box_y1
    box_area = w * h

    if pred_area == 0 or box_area == 0:
        raise ValueError("预测框或标签框面积为0，无法计算交集面积")

    # 计算交集区域
    inter_x1 = max(pred_x1, box_x1)
    inter_y1 = max(pred_y1, box_y1)
    inter_x2 = min(pred_x2, box_x2)
    inter_y2 = min(pred_y2, box_y2)

    # 检查是否有交集
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    # 计算交集面积和比例
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    overlap_ratio = inter_area / box_area

    # 预测框与标签框的交集面积是否超过标签框面积的阈值
    if overlap_ratio > threshold:
        return True

    return False


if __name__ == '__main__':
    path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/config/dataset.yaml'
    data = load_yaml(path)
    print()
