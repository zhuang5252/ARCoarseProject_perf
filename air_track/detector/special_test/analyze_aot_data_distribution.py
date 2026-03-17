import os
import pandas as pd
from matplotlib import pyplot as plt
from air_track.utils import get_all_files, row_data_save_csv


"""分析Aot数据集的类别及面积分布"""


def plot_bar_chart_cls_counts(dictionary, save_path):
    # 提取字典的键和值
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # 提取排序后的键和值
    keys = keys[:20]
    values = values[:20]

    # 绘制柱状图
    plt.figure(figsize=(20, 10))
    bars = plt.bar(keys, values)
    # 设置x轴标签旋转角度
    plt.xticks(rotation=45)

    # 在每个柱子上方添加具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')  # va='bottom' 表示数值在柱子底端

    plt.xlabel('cls_name')
    plt.ylabel('count')
    plt.title('Class Count')

    # 保存图表
    plt.savefig(save_path)
    plt.close()


def plot_bar_chart_areas(dictionary, save_path):
    # 提取字典的键和值
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # 提取排序后的键和值
    keys = [str(key).replace(')', ']') for key in keys]

    # 绘制柱状图
    plt.figure(figsize=(20, 10))
    bars = plt.bar(keys, values)
    # 设置x轴标签旋转角度
    plt.xticks(rotation=20)

    # 在每个柱子上方添加具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')  # va='bottom' 表示数值在柱子底端

    plt.xlabel('area_range')
    plt.ylabel('area_count')
    plt.title('Area')

    # 保存图表
    plt.savefig(save_path)
    plt.close()


def create_bbox_areas_interval(interval=50, max_value=1000):
    """创建bbox_areas的区间字典"""
    bbox_areas = {}
    start = 0
    end = interval

    while end <= max_value:
        bbox_areas[(start, end)] = 0
        start = end
        end += interval

    print(bbox_areas)
    return bbox_areas


if __name__ == '__main__':
    above_horizon = -1
    path = '/media/csz_changsha/2C5821EF5821B88A/data/Aot'
    save_csv_file_path = path + '/data.csv'
    files_paths = get_all_files(path, target_depth=3, file_end=['groundtruth.csv'])

    column_labels = ['flight_id', 'img_name', 'frame', 'cls_name', 'range_distance_m', 'is_above_horizon',
                     'size_width', 'size_height', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom', 'box_w', 'box_h', 'area']
    bbox_areas = create_bbox_areas_interval(interval=50, max_value=1000)

    max_cls_counts = {}
    is_above_horizons_idx_all = 0

    # 指定每次迭代返回的行数
    chunk_size = 1  # 这里设置为1，表示按行读取
    save_data = []
    for csv_file in files_paths:
        df = pd.read_csv(csv_file)

        is_above_horizons = df.is_above_horizon.values
        cls_names = df.id.values
        gt_lefts = df.gt_left.values
        gt_tops = df.gt_top.values
        gt_rights = df.gt_right.values
        gt_bottoms = df.gt_bottom.values

        is_above_horizons_idx = []
        '''先取出符合海天线条件的索引'''
        for i, value in enumerate(is_above_horizons):
            if float(value) == above_horizon:
                is_above_horizons_idx.append(i)
        is_above_horizons_idx_all += len(is_above_horizons_idx)
        '''从符合海天线条件的索引中计算类别分布'''
        for i in is_above_horizons_idx:
            # 计算各个类别的数量
            if cls_names[i] not in max_cls_counts.keys():
                max_cls_counts[cls_names[i]] = 1
            else:
                max_cls_counts[cls_names[i]] += 1

            # 计算目标框的大小
            bbox_w = gt_rights[i] - gt_lefts[i]
            bbox_h = gt_bottoms[i] - gt_tops[i]

            # 面积分布
            bbox_area = bbox_w * bbox_h
            for key in bbox_areas.keys():
                if key[0] < bbox_area <= key[1]:
                    bbox_areas[key] += 1

        # 创建一个迭代器，按块读取CSV文件
        csv_reader = pd.read_csv(csv_file, chunksize=chunk_size)
        for row in csv_reader:
            is_above_horizon = row['is_above_horizon'].values.item()

            if is_above_horizon == -1:
                save_row = {label: '' for label in column_labels}  # 初始化行，所有列都有空字符串
                save_row['flight_id'] = row['flight_id'].values.item()
                save_row['img_name'] = row['img_name'].values.item()
                save_row['frame'] = row['frame'].values.item()
                save_row['cls_name'] = row['id'].values.item()
                save_row['range_distance_m'] = row['range_distance_m'].values.item()
                save_row['is_above_horizon'] = row['is_above_horizon'].values.item()
                save_row['size_width'] = row['size_width'].values.item()
                save_row['size_height'] = row['size_height'].values.item()
                save_row['gt_left'] = row['gt_left'].values.item()
                save_row['gt_top'] = row['gt_top'].values.item()
                save_row['gt_right'] = row['gt_right'].values.item()
                save_row['gt_bottom'] = row['gt_bottom'].values.item()
                save_row['box_w'] = save_row['gt_right'] - save_row['gt_left']
                save_row['box_h'] = save_row['gt_bottom'] - save_row['gt_top']
                save_row['area'] = save_row['box_w'] * save_row['box_h']

                save_data.append(save_row)
            else:
                continue

    row_data_save_csv(save_csv_file_path, save_data, column_labels)

    print(is_above_horizons_idx_all)
    print(max_cls_counts)
    print(bbox_areas)
    plot_bar_chart_cls_counts(max_cls_counts, os.path.join(path, 'max_cls_counts.png'))
    plot_bar_chart_areas(bbox_areas, os.path.join(path, 'bbox_areas.png'))
