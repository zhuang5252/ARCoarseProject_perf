import csv
import math
import os
import numpy as np
import pandas as pd
from PIL import Image
from air_track.detector.visualization.visualize_and_save import show_scatter_plot
from air_track.utils import common_utils, img_fn, col_data_save_csv


def cal_snr_orig(position, img_path, min_width=10, min_height=10):
    """
    计算信噪比snr
    来自于matlab代码翻译过来，已经与matlab代码核对无误

    position：[Top_left_x, Top_left_y, Bottom_right_x, Bottom_right_y, Center_x, Center_y]
    """
    # 读取图像并转换为double类型
    data0 = np.array(Image.open(img_path), dtype=np.double)
    data0_h = data0.shape[0]
    data0_w = data0.shape[1]

    # 计算宽度
    if abs(position[0] - position[2]) <= min_width:
        width = min_width
    else:
        width = abs(position[0] - position[2])

    # 计算高度
    if abs(position[3] - position[1]) <= min_height:
        height = min_height
    else:
        height = abs(position[3] - position[1])

    # 提取目标区域
    data_MB = data0[max(position[3] - round(0.5 * height) - 1, 0):min(position[1] + round(0.5 * height), data0_h),
              max(position[4] - round(0.5 * width) - 1, 0):min(position[4] + round(0.5 * width), data0_w)]
    data_MB_mean = np.mean(data_MB)

    # 提取背景区域
    data_BJ1 = data0[position[5] + round(0.5 * height) - 1:, :]
    data_BJ2 = data0[:int(position[5] + (0.5 * height)),
               int(position[4] - round(1.5 * width) - 1):int(position[4] - round(0.5 * width))]
    data_BJ3 = data0[int(position[5] - round(0.5 * height) - 1):data0_h,
               int(position[4] + round(0.5 * width) - 1):int(position[4] + round(1.5 * width))]
    data_BJ4 = data0[:int(position[5] - round(0.5 * height)), :data0_w]

    # 将背景区域展平
    data_BJ1_one = data_BJ1.flatten(order='F')
    data_BJ2_one = data_BJ2.flatten(order='F')
    data_BJ3_one = data_BJ3.flatten(order='F')
    data_BJ4_one = data_BJ4.flatten(order='F')

    data_BJ_one = np.concatenate((data_BJ1_one, data_BJ2_one, data_BJ3_one, data_BJ4_one))
    data_BJ_mean = np.mean(data_BJ_one)
    data_BJ_std = np.std(data_BJ_one)

    # 计算AA和snr
    # aa = (data_MB_mean - data_BJ_mean) / data_BJ_std
    aa = data_MB_mean / data_BJ_mean
    snr = 10 * np.log10(aa)

    return snr


def cal_snr(position, img_path, min_width=10, min_height=10, BJ_range=10, apply_std=False):
    """
    计算信噪比snr
    目标范围最小为min_width*min_height，背景范围为上下左右各延伸BJ_range范围（除去目标范围）

    position：[Top_left_x, Top_left_y, Bottom_right_x, Bottom_right_y, Center_x, Center_y]
    """
    # 读取图像并转换为double类型
    data0 = np.array(Image.open(img_path), dtype=np.double)
    data0_h = data0.shape[0]
    data0_w = data0.shape[1]

    # 计算宽度
    if abs(position[0] - position[2]) <= min_width:
        width = min_width
        position[0] = max(position[4] - round(0.5 * width), 0)
        position[2] = min(position[4] + round(0.5 * width), data0_w)

    # 计算高度
    if abs(position[3] - position[1]) <= min_height:
        height = min_height
        position[1] = max(position[5] - round(0.5 * height), 0)
        position[3] = min(position[5] + round(0.5 * height), data0_h)

    # 提取目标区域
    data_MB = data0[position[1]:position[3] + 1, position[0]:position[2] + 1]
    data_MB_mean = np.mean(data_MB)

    data_BJ_and_MB = data0[position[1] - BJ_range:position[3] + BJ_range + 1,
                     position[0] - BJ_range:position[2] + BJ_range + 1]

    # 提取背景区域
    data_BJ1 = data0[max(position[1] - BJ_range, 0):position[3] + 1, max(position[0] - BJ_range, 0):position[0]]
    data_BJ2 = data0[max(position[1] - BJ_range, 0):position[1], position[0]:min(position[2] + BJ_range + 1, data0_w)]
    data_BJ3 = data0[position[1]:min(position[3] + BJ_range + 1, data0_h),
               min(position[2] + 1, data0_w - 1):min(position[2] + BJ_range + 1, data0_w)]
    data_BJ4 = data0[min(position[3] + 1, data0_h - 1):min(position[3] + BJ_range + 1, data0_h),
               max(position[0] - BJ_range, 0):position[2] + 1]

    # 将背景区域展平
    data_BJ1_one = data_BJ1.flatten(order='F')
    data_BJ2_one = data_BJ2.flatten(order='F')
    data_BJ3_one = data_BJ3.flatten(order='F')
    data_BJ4_one = data_BJ4.flatten(order='F')

    data_BJ_one = np.concatenate((data_BJ1_one, data_BJ2_one, data_BJ3_one, data_BJ4_one))
    data_BJ_mean = np.mean(data_BJ_one)
    data_BJ_std = np.std(data_BJ_one)

    # 计算aa和snr
    if not apply_std:
        aa = data_MB_mean / data_BJ_mean
    else:
        if data_MB_mean < data_BJ_mean:
            aa = data_MB_mean / data_BJ_mean
        else:
            aa = (data_MB_mean - data_BJ_mean) / data_BJ_std
    snr = 10 * np.log10(aa)

    return snr


def cal_multi_snr_and_save(cfg_data, parts, save_path, min_width=10, min_height=10, BJ_range=10, apply_std=False):
    """计算所传入parts的SNR，并保存在csv文件中，且进行可视化保存"""
    data_path = cfg_data['data_dir']
    for part in parts:
        csv_file = os.path.join(data_path, part, cfg_data['label_file'])

        df = pd.read_csv(csv_file)
        frames = df.frame.values
        distances = df.DM_distance.values

        snr_list, w_list, h_list = [], [], []
        area_list = []
        for i, frame in enumerate(frames):
            target_id = df.Target_id[i]
            # 有目标
            if not np.isnan(target_id):
                flight_id = df.flight_id[i]
                img_name = df.img_name[i]
                img_path = img_fn(cfg_data, part, flight_id, img_name[:-4])

                position = np.array([df.Top_left_x[i], df.Top_left_y[i], df.Bottom_right_x[i],
                                     df.Bottom_right_y[i], df.Center_x[i], df.Center_y[i]]).astype(int)
                # 计算SNR
                snr = cal_snr(position, img_path, min_width, min_height, BJ_range, apply_std)
                snr_list.append(snr)

                w = math.ceil(max(df.Bottom_right_x[i] - df.Top_left_x[i], 1))
                h = math.ceil(max(df.Bottom_right_y[i] - df.Top_left_y[i], 1))
                w_list.append(w)
                h_list.append(h)
                area_list.append(w * h)
            # 无目标
            else:
                snr_list.append('')
                w_list.append('')
                h_list.append('')
                area_list.append('')

        assert len(frames) == len(snr_list), 'len(frames) != len(snr_list)'

        # 写入SNR.csv文件
        save_csv_file = os.path.join(save_path, part, 'SNR.csv')
        data = {'frame': frames, 'distance': distances, 'SNR': snr_list,
                'width': w_list, 'height': h_list, 'area': area_list}
        # 按列写入csv文件
        col_data_save_csv(save_csv_file, data)

        df_snr = pd.read_csv(save_csv_file)

        # 将SNR画图
        save_show_snr_file = os.path.join(save_path, part, 'SNR.png')
        show_scatter_plot(x=df_snr.distance.values, y=df_snr.SNR.values, save_path=save_show_snr_file,
                          title="The variation of Target's SNR with distance",
                          xlabel='distance', ylabel='SNR', label='SNR', color='red')
        # 将目标框的宽画图
        save_show_width_file = os.path.join(save_path, part, 'width.png')
        show_scatter_plot(x=df_snr.distance.values, y=df_snr.width.values, save_path=save_show_width_file,
                          title="The variation of target's width with distance",
                          xlabel='distance', ylabel='width', label='width', color='red')
        # 将目标框的高画图
        save_show_height_file = os.path.join(save_path, part, 'height.png')
        show_scatter_plot(x=df_snr.distance.values, y=df_snr.height.values, save_path=save_show_height_file,
                          title="The variation of target's height with distance",
                          xlabel='distance', ylabel='height', label='height', color='red')

        # 将目标框的面积画图
        save_show_area_file = os.path.join(save_path, part, 'area.png')
        show_scatter_plot(x=df_snr.distance.values, y=df_snr.area.values, save_path=save_show_area_file,
                          title="The variation of target's area with distance",
                          xlabel='distance', ylabel='area', label='area', color='red')


def csv_add_snr(csv_file):
    # 读取CSV文件
    pass


if __name__ == '__main__':
    dataset_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/config/dataset_old.yaml'
    # 合并若干个yaml的配置文件内容
    cfg_data = common_utils.load_yaml(dataset_yaml)

    # save_path = cfg_data['data_dir']
    save_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/output/目标最小尺寸1×1_背景范围上下左右各5个像素'
    # parts = cfg_data['part_train'] + cfg_data['part_val'] + cfg_data['part_test']
    parts = cfg_data['part_test']

    cal_multi_snr_and_save(cfg_data, parts, save_path, min_width=1, min_height=1, BJ_range=5, apply_std=False)
