import os
import math
import pickle
import pandas as pd
import numpy as np
from air_track.utils import common_utils
from air_track.utils.common_utils import img_fn


def write_single_part_data(cfg, part, label_file, cache_fn):
    """讲标签数据写入缓存文件"""
    frames_dict = {}
    df = pd.read_csv(label_file)

    # 支持类别是id和cls_name两种key
    if 'cls_name' in df.keys():
        cls_key = 'cls_name'
    elif 'id' in df.keys():
        cls_key = 'id'
    else:
        raise ValueError('No cls_name or id in the csv file')

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
        target_id = row['Target_id']
        cls_name = row[cls_key]
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
                'detections': [],
                'error': False
            }

        if math.isnan(target_id):
            continue  # 无目标的空图像

        frame = frames_dict[key]
        # 给frame添加detections信息
        frame['detections'].append(
            {
                'cls_name': cls_name,  # 目标类别名
                'target_id': int(target_id),  # 目标编号
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

    # TODO 由于AR数据集有目标的都有distance，考虑二合一
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
    save_dir = '/home/csz_changsha/data/AR_2410/normal'
    cfg_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/config/dataset.yaml'
    cfg_data = common_utils.load_yaml(cfg_path)

    frames, frame_nums_with_objs, frame_nums_with_distance_objs = load_datasets(cfg=cfg_data,
                                                                                parts=cfg_data['part_train'],
                                                                                stage='train')
