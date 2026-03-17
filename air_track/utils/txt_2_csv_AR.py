import os
from air_track.utils import get_all_files, row_data_save_csv, common_utils

"""按照Aot数据集csv文件格式，构建AR数据集的csv文件，目前为单目标数据集"""

# AR数据集标签按顺序表示
AR_column_labels = ['Channel', 'Img_id', 'X', 'Y', 'Z', 'Pos_angle', 'Pitch_angle', 'Roll_angle',
                    'Min_brightness', 'Max_brightness', 'Target_conut', 'Target_id', 'Top_left_x',
                    'Top_left_y', 'Bottom_right_x', 'Bottom_right_y', 'Center_x', 'Center_y', 'DM_distance']

# aot数据集标签未按顺序表示（'cls_name'为人为后加的）
aot_column_labels = ['flight_id', 'img_name', 'frame', 'id', 'cls_name', 'range_distance_m', 'is_above_horizon',
                     'size_width', 'size_height', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']

# 创建列标签（二者相加）
column_labels = AR_column_labels + aot_column_labels


def txt_2_csv(cfg, orig_path, save_path):
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

    # TODO 目前按照AR数据集格式为单目标数据，后续多目标需要针对性修改
    # 重新整理数据以适应CSV格式
    for group in grouped_data:
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
            if key == 'id':  # 目标id(不同id的目标可以是同一类别，也可以是不同类别)
                row[key] = row['Target_id']
            if key == 'cls_name':  # 目标名称 目前只有一类
                if int(row['Target_conut']) > 0:
                    row[key] = cfg['classes'][1]
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

        csv_data.append(row)

    # 写入CSV文件
    csv_file_path_1 = save_path
    csv_file_path_2 = os.path.join(os.path.dirname(save_path), 'ImageSets/groundtruth.csv')
    row_data_save_csv(csv_file_path_1, csv_data, column_labels)
    row_data_save_csv(csv_file_path_2, csv_data, column_labels)


if __name__ == '__main__':
    cfg_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/detector/config/dataset.yaml'
    cfg_data = common_utils.load_yaml(cfg_path)

    path = '/home/csz_changsha/data/AR_2410/normal_airtrack_test_1'
    files_paths = get_all_files(path, target_depth=2, file_end=['.txt'])

    for orig_path in files_paths:
        save_path = orig_path.replace('.txt', '.csv')
        txt_2_csv(cfg_data, orig_path, save_path)
