import copy
import csv
import os
import pandas as pd
from air_track.utils import get_all_files


def find_prev_normal_idx(peaks, idx):
    """递归找前一个正常数据的idx索引"""
    if idx < 0:
        return 0
        # assert 'Not find prev_idx, or data all error.'
    if idx in peaks:
        idx -= 1
        return find_prev_normal_idx(peaks, idx)
    else:
        return idx


def find_next_normal_idx(peaks, idx, max_idx=749):
    """递归找下一个正常数据的idx索引"""
    if idx > max_idx:
        assert 'Not find next_idx, or data all error.'
    if idx in peaks:
        idx += 1
        return find_next_normal_idx(peaks, idx)
    else:
        return idx


def find_peaks(data, min_difference, max_distance=30000):
    """从列表数据中找出不合理的数据idx"""
    peaks = []
    n = len(data)

    # 检查第一个元素
    if n > 1 and data[0] >= max_distance:
        peaks.append(0)

    # 检查中间的元素
    for i in range(1, n - 1):
        # 递归找到前一个正常数据
        prev_idx = find_prev_normal_idx(peaks, i - 1)
        if data[i] >= data[prev_idx] + min_difference:
            peaks.append(i)

    # 检查最后一个元素
    if n > 1 and data[-1] >= data[-2] + min_difference:
        peaks.append(n - 1)

    return peaks


def change_distance(changed_distances, error_idx_list, max_distance=35000):
    """修正标签中错误的distance，并返回修正后正确的所有的distance列表"""
    error_idx_list = copy.deepcopy(error_idx_list)
    while len(error_idx_list) != 0:
        error_idx = error_idx_list[0]
        # 第0帧distance就错误
        if error_idx == 0:
            changed_distances[error_idx] = max_distance

        # 找前一帧正常的distance索引
        if error_idx - 1 not in error_idx_list:
            prev_idx = error_idx - 1
        else:
            prev_idx = find_prev_normal_idx(error_idx_list, error_idx - 1)

        if error_idx + 1 not in error_idx_list:
            next_idx = error_idx + 1
        else:
            next_idx = find_next_normal_idx(error_idx_list, error_idx + 1)

        temp = next_idx - prev_idx
        temp = (changed_distances[next_idx] - changed_distances[prev_idx]) / temp
        changed_distances[error_idx] = round(changed_distances[prev_idx] + temp, 1)

        error_idx_list.pop(0)

    return changed_distances


def check_distances(csv_file, min_diff=5000, max_distance=30000):
    df = pd.read_csv(csv_file)
    csv_distances = df.DM_distance.values

    error_idx_list = find_peaks(csv_distances, min_diff, max_distance)

    if len(error_idx_list) == 0:
        print(csv_file, 'PASS')

    return error_idx_list, csv_distances


if __name__ == '__main__':
    path = '/home/csz_changsha/data/AR_2410/normal_airtrack_test'
    files_paths = get_all_files(path, target_depth=3, file_end=['.csv'])

    save_file = 'ImageSets/groundtruth.csv'
    min_diff = 5000  # 设置最小距离差为5000米
    max_distance = 35000  # 设置最大距离为35000米

    for csv_file in files_paths:
        base_path = os.path.dirname(os.path.dirname(csv_file))
        save_path = os.path.join(base_path, save_file)

        # 现初步校对错误数据
        error_idx_list, csv_distances = check_distances(csv_file, min_diff=min_diff, max_distance=max_distance)

        if len(error_idx_list) != 0:
            # 修改错误数据
            changed_distances = change_distance(csv_distances, error_idx_list, max_distance=max_distance)

            # 读取CSV文件
            with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            # 修改指定的值
            for i in error_idx_list:
                rows[i]['DM_distance'] = changed_distances[i]
                rows[i]['range_distance_m'] = changed_distances[i]

            # 将修改后的数据写回新的CSV文件
            with open(save_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"文件已保存为 {save_path}")

            # 再次校对数据是否全部正确
            check_distances(csv_file, min_diff=min_diff, max_distance=max_distance)
