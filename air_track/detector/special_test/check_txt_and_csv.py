import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from air_track.utils import get_all_files

"""按照Aot数据集csv文件格式，构建AR数据集的csv文件"""

# 创建列标签
column_labels = ['Channel', 'Img_id', 'X', 'Y', 'Z', 'Pos_angle', 'Pitch_angle', 'Roll_angle',
                 'Min_brightness', 'Max_brightness', 'Target_conut', 'Target_id', 'Top_left_x',
                 'Top_left_y', 'Bottom_right_x', 'Bottom_right_y', 'Center_x', 'Center_y', 'DM_distance',
                 'flight_id', 'img_name', 'frame', 'id', 'range_distance_m', 'is_above_horizon',
                 'size_width', 'size_height', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']


def create_line_chart(x_axis, y_axis, save_path, save_name):
    l = len(x_axis)
    save_file = os.path.join(save_path, save_name)
    base_name = os.path.basename(save_path)
    # 绘制折线图
    plt.figure(figsize=(8, 6))
    # 设置 y 轴刻度
    plt.yticks(range(0, 81, 10), ['0', '10', '20', '30', '40', '50', '60', '70', '80'])
    # plt.plot(x_axis, y_axis)  # 折线图
    # plt.bar(x_axis, y_axis)  # 柱状图
    plt.scatter(x_axis, y_axis)  # 散点图
    plt.title(f'{base_name} distance len: {l}')
    plt.xlabel('idx')
    plt.ylabel('distance_list')
    # plt.grid(True)
    plt.savefig(save_file)


def txt_2_csv(orig_path, save_path):
    # 数据字符串
    # 打开一个文件
    with open(orig_path, 'r') as file:
        # 初始化变量
        grouped_data = []
        current_group = []

        # 按行读取文件内容
        for line in file:
            # 去除行尾的换行符，并处理每一行
            line = line.strip()
            # 分组数据
            if line.startswith('User'):
                if current_group:  # 如果已经有一个组开始了，保存它并开始一个新组
                    grouped_data.append(current_group)
                    current_group = []
            current_group.append(line)

    # 添加最后一个组
    if current_group:
        grouped_data.append(current_group)

    # 准备CSV数据
    csv_data = []

    idx_list = []
    distance_list = []

    # 重新整理数据以适应CSV格式
    for idx, group in enumerate(grouped_data):
        # 将组内的所有行合并为一个字符串，然后分割
        merged_values = " ".join(group).split()
        row = {label: '' for label in column_labels}  # 初始化行，所有列都有空字符串
        for i, key in enumerate(row):
            if i < len(merged_values):
                row[key] = merged_values[i]
            if key == 'flight_id':
                row[key] = os.path.basename(os.path.dirname(orig_path)) + '_result'
            if key == 'img_name':
                row[key] = row['Channel'] + '_' + str(row['Img_id']).zfill(6) + '.png'
            if key == 'frame':
                row[key] = row['Img_id']
            if key == 'id':  # 目标名称
                row[key] = row['Target_id']
            if key == 'range_distance_m':
                row[key] = row['DM_distance']
            if key == 'is_above_horizon':
                row[key] = ''
            if key == 'size_width':
                row[key] = '640'
            if key == 'size_height':
                row[key] = '512'
            if key == 'gt_left':
                row[key] = row['Top_left_x']
            if key == 'gt_top':
                row[key] = row['Top_left_y']
            if key == 'gt_right':
                row[key] = row['Bottom_right_x']
            if key == 'gt_bottom':
                row[key] = row['Bottom_right_y']

        if len(merged_values) == 19:
            distance_list.append(merged_values[-1])
            idx_list.append(idx)

        csv_data.append(row)

    create_line_chart(idx_list, distance_list, os.path.dirname(save_path), save_name='txt.png')


def check_txt_and_csv(txt_file, csv_file):
    # 数据字符串
    # 打开一个文件
    with open(txt_file, 'r') as file:
        # 初始化变量
        grouped_data = []
        current_group = []

        # 按行读取文件内容
        for line in file:
            # 去除行尾的换行符，并处理每一行
            line = line.strip()
            # 分组数据
            if line.startswith('User'):
                if current_group:  # 如果已经有一个组开始了，保存它并开始一个新组
                    grouped_data.append(current_group)
                    current_group = []
            current_group.append(line)

    # 添加最后一个组
    if current_group:
        grouped_data.append(current_group)

    txt_idx_list = []
    txt_distance_list = []

    # 按行将distance保存为列表
    for idx, group in enumerate(grouped_data):
        # 将组内的所有行合并为一个字符串，然后分割
        merged_values = " ".join(group).split()

        if len(merged_values) == 19:
            txt_distance_list.append(float(merged_values[-1]))
        else:
            txt_distance_list.append(None)
        txt_idx_list.append(idx)

    csv_idx_list = []
    csv_distance_list = []

    df = pd.read_csv(csv_file)
    csv_distances = df.DM_distance.values
    for i, item in enumerate(csv_distances):
        if np.isnan(item):
            csv_distance_list.append(None)
        else:
            csv_distance_list.append(item)
        csv_idx_list.append(i)

    error_idx_list = []
    for idx, (i, j) in enumerate(zip(txt_distance_list, csv_distance_list)):
        if i == j:
            continue
        else:
            error_idx_list.append(idx)

    if txt_idx_list == csv_idx_list and txt_distance_list == csv_distance_list:
        print(os.path.basename(os.path.dirname(txt_file)), 'PASS ')
    else:
        print(os.path.basename(os.path.dirname(txt_file)), 'ERROR ')


if __name__ == '__main__':
    path = '/home/csz_changsha/data/AR_2410/normal_airtrack_test'
    files_paths = get_all_files(path, target_depth=2, file_end=['.txt'])

    for txt_file in files_paths:
        csv_file = txt_file.replace('.txt', '.csv')
        check_txt_and_csv(txt_file, csv_file)
