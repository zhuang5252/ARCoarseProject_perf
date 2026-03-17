import copy
import os
import math
import pickle
import re
import pandas as pd
import numpy as np
from air_track.utils import common_utils
from air_track.utils.common_utils import img_fn


def is_target_class(item, target_class):
    """
    检查检测项是否属于目标类别。

    :param item: 检测项，包含类别信息。
    :param target_class: 目标类别名称。
    :return: 如果检测项属于目标类别，返回 True；否则返回 False。
    """
    return item['cls_name'] in target_class


def is_within_size_range(item, size_range, img_width, img_height):
    """
    检查检测项的尺寸是否在指定范围内。

    :param item: 检测项，包含宽度和高度信息（归一化后的）。
    :param size_range: 尺寸范围，包含最小和最大宽度和高度。
    :param img_width: 图像的宽。
    :param img_height: 图像的高。
    :return: 如果检测项的尺寸在范围内，返回 True；否则返回 False。
    """
    w = float(item['w']) * img_width
    h = float(item['h']) * img_height
    return (size_range['w_min'] <= w <= size_range['w_max'] and
            size_range['h_min'] <= h <= size_range['h_max'])


def is_above_horizon(item, horizon):
    """
    检查检测项是否在指定地平线之上。

    :param item: 检测项，包含是否在地平线之上的信息。
    :param horizon: 指定的地平线，None 表示不检查地平线。
    :return: 如果检测项在地平线之上或地平线未指定，返回 True；否则返回 False。
    """
    return horizon is None or int(item['is_above_horizon']) == int(horizon)


def filter_detections(detections, target_class, size_range, horizon, img_width, img_height):
    """
    过滤检测项，保留符合条件的检测项。

    :param detections: 检测项列表。
    :param target_class: 目标类别名称。
    :param size_range: 尺寸范围。
    :param horizon: 指定的地平线。
    :param img_width: 图像的宽。
    :param img_height: 图像的高。
    :return: 过滤后的检测项列表和是否放弃当前帧的标志。
    """
    filtered_detections = []
    give_up_frame = False

    for detection in detections:
        if not is_target_class(detection, target_class):
            continue

        if not is_within_size_range(detection, size_range, img_width, img_height):
            give_up_frame = True
            break

        if not is_above_horizon(detection, horizon):
            give_up_frame = True
            break

        filtered_detections.append(detection)

    return filtered_detections, give_up_frame


def filter_frames(frames_dict, cfg):
    """
    过滤帧，保留符合条件的帧。
        '''
        筛选设定目标：
            1. 第一步判断类别
            2. 第二步判断尺寸
            3. 第三步判断海天线
            只有类别符合，尺寸或者海天线不符合，为避免干扰训练，则舍弃当前帧
        '''

    :param frames_dict: 帧字典，包含每帧的检测项。
    :param cfg: 配置参数，包含目标类别、尺寸范围和地平线信息。
    :return: 过滤后的帧列表。
    """
    frames_dict_copy = copy.deepcopy(frames_dict)
    filtered_frames = []
    target_cls = cfg['target_cls']
    size_range = {
        'w_min': float(cfg['range_bbox_w'][0]),
        'w_max': float(cfg['range_bbox_w'][1]),
        'h_min': float(cfg['range_bbox_h'][0]),
        'h_max': float(cfg['range_bbox_h'][1])
    }
    horizon = cfg['is_above_horizon'] if cfg['is_above_horizon'] != 'None' else None

    for frame_key in sorted(frames_dict_copy.keys()):
        img_width = frames_dict_copy[frame_key]['img_width']
        img_height = frames_dict_copy[frame_key]['img_height']
        detections = frames_dict_copy[frame_key]['detections']
        # 筛选 detections
        filtered_detections, give_up_frame = filter_detections(detections, target_cls, size_range, horizon, img_width, img_height)

        # 如果舍弃当前帧，则不进分支
        if not give_up_frame:
            frames_dict_copy[frame_key]['detections'] = filtered_detections
            filtered_frames.append(frames_dict_copy[frame_key])

    return filtered_frames


def write_single_part_data(cfg, part, label_file, cache_fn):
    """讲标签数据写入缓存文件"""
    frames_dict = {}
    df = pd.read_csv(label_file)
    rx = re.compile(r'(\D+)(\d+)')  # 这个正则表达式模式用于匹配由非数字字符开头，后面跟着一个或多个数字字符的字符串序列
    # 按行（顺序）读取
    for _, row in df.iterrows():
        flight_id = row['flight_id']
        img_name = row['img_name'][:-4]  # 截至到-4是为了舍去.png
        frame_num = row['frame']
        distance = row['range_distance_m']
        size_width = row['size_width']
        size_height = row['size_height']
        gt_left = row['gt_left']
        gt_right = row['gt_right']
        gt_top = row['gt_top']
        gt_bottom = row['gt_bottom']
        cls_name = row['id']
        is_above_horizon = row['is_above_horizon']

        img_file = img_fn(cfg, part, flight_id, img_name)

        # 判断图片是否存在
        if not os.path.exists(img_file):
            print('skip missing img', img_file)
            continue

        key = (flight_id, frame_num)
        # 存储Frame的几个要素，frames_dict存储所有帧数据
        if key not in frames_dict:
            frames_dict[key] = {
                'part': part,
                'flight_id': flight_id,
                'frame_num': frame_num,
                'img_name': img_name,
                'img_path': img_file,
                'have_distance': False,
                'img_width': size_width,
                'img_height': size_height,
                'detections': [],
                'error': False
            }

        if not isinstance(cls_name, str) and math.isnan(cls_name):
            continue  # 不是字符串类型且且为 NaN（无目标的空图像）

        m = rx.search(cls_name)  # 使用正则表达式对象 rx 来搜索匹配 item_id 的字符串
        target_id = int(m.group(2))  # 获取数字字符序列并转为int型

        frame = frames_dict[key]
        # 给frame添加detections信息
        frame['detections'].append(
            {
                'cls_name': cls_name,  # 目标类别名
                'target_id': target_id,  # 目标类别编号
                'distance': distance,
                'is_above_horizon': is_above_horizon,
                'cx': (gt_left + gt_right) / 2 / size_width,
                'cy': (gt_top + gt_bottom) / 2 / size_height,
                'w': (gt_right - gt_left) / size_width,
                'h': (gt_bottom - gt_top) / size_height,
            }
        )

        # 距离不为Nan，have_distance = True
        if not np.isnan(distance):
            frame['have_distance'] = True

    '''筛选设定目标'''
    if cfg['filter_flag']:
        frames = filter_frames(frames_dict, cfg)
    else:
        frames = [frames_dict[k] for k in sorted(frames_dict.keys())]  # 所有帧的frames，key=(flight_id, frame_num)

    frame_nums_with_objs = [i for i, f in enumerate(frames) if len(f['detections'])]  # 所有帧的detections列表（具备检测标签的）
    frame_nums_with_distance_objs = [i for i, f in enumerate(frames) if f['have_distance']]  # 所有有距离distance帧的索引

    pickle.dump((frames, frame_nums_with_objs, frame_nums_with_distance_objs), open(cache_fn, 'wb'))


def read_single_part_data(cfg, part, data_dir, cache_fn):
    """读取当前part的数据集缓存文件"""
    # 文件不存在需先写入
    if not os.path.exists(cache_fn):
        label_name = cfg['label_file']
        label_file = f'{data_dir}/{part}/{label_name}'
        write_single_part_data(cfg, part, label_file, cache_fn)

    frames, frame_nums_with_items, frame_nums_with_match_items = pickle.load(open(cache_fn, 'rb'))

    # 返回每一帧的frame（类信息）、frame中有Detection标签信息的帧索引、有距离distance帧的索引
    return frames, frame_nums_with_items, frame_nums_with_match_items


def load_datasets(cfg, parts, stage):
    """根据stage和parts 读取整个训练集或者验证集"""
    save_dir = cfg['data_dir']
    data_dir = cfg['data_dir']

    frames, frame_nums_with_objs, frame_nums_with_distance_objs = [], [], []

    # 从每个part中读取数据
    for part in parts:
        with common_utils.timeit_context('load ds ' + str(part)):  # 计算数据load的耗时
            # 缓存文件
            cache_fn = f'{save_dir}/ds_{part}_{stage}.pkl'
            print(cache_fn)
            # 获取每一帧的frame（帧信息）、frame中有Detection标签信息的帧索引、有距离distance帧的索引
            part_frames, part_frame_nums_with_objs, part_frame_nums_with_distance_objs = read_single_part_data(
                cfg=cfg,
                part=part,
                data_dir=data_dir,
                cache_fn=cache_fn
            )

            length = len(frames)
            frames += part_frames
            frame_nums_with_objs += [i + length for i in part_frame_nums_with_objs]
            frame_nums_with_distance_objs += [i + length for i in part_frame_nums_with_distance_objs]

    # 返回每一帧的frame（类信息）、frame中有Detection标签信息的帧索引、有距离distance帧的索引
    return frames, frame_nums_with_objs, frame_nums_with_distance_objs


if __name__ == '__main__':
    save_dir = '/home/csz_changsha/data/Aot_small_test'
    cfg_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/detector/config/dataset_aot.yaml'
    cfg_data = common_utils.load_yaml(cfg_path)

    frames, frame_nums_with_objs, frame_nums_with_distance_objs = load_datasets(cfg=cfg_data,
                                                                                parts=cfg_data['part_train'],
                                                                                stage='train')
