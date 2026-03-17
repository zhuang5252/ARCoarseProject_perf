# -*- coding: utf-8 -*-
# @Author    : 
# @File      : check_AR2511_csv_json.py
# @Created   : 2025/11/24 下午4:20
# @Desc      : 检查CSV文件与JSON文件的数据一致性

"""
检查CSV文件与JSON文件的数据一致性
"""

import os
import json
import pandas as pd
import glob
from pathlib import Path


def validate_csv_with_json(csv_file_path, json_file_path):
    """
    校对CSV文件与JSON文件的数据一致性

    Args:
        csv_file_path: CSV文件路径
        json_file_path: JSON文件路径

    Returns:
        dict: 校对结果
    """

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 读取CSV文件
    csv_data = pd.read_csv(csv_file_path)

    # 提取JSON基本信息
    json_width = json_data["Head"]["Width"]
    json_height = json_data["Head"]["Height"]
    json_m_filename = json_data["Head"]["M_FileName"]

    # 按Pid分组JSON数据
    json_by_pid = {}
    for biaozhu_info in json_data["Body"]["BiaoZhuInfo"]:
        pid = biaozhu_info["Pid"]
        if pid not in json_by_pid:
            json_by_pid[pid] = []
        json_by_pid[pid].append(biaozhu_info)

    # 按frame分组CSV数据
    csv_by_frame = {}
    for _, row in csv_data.iterrows():
        frame = row['frame']
        if frame not in csv_by_frame:
            csv_by_frame[frame] = []
        csv_by_frame[frame].append(row)

    # 检查项目
    validation_results = {
        'total_frames': 0,
        'total_targets': 0,
        'errors': [],
        'warnings': [],
        'summary': {}
    }

    # 检查所有帧的一致性
    all_frames = set(list(json_by_pid.keys()) + list(csv_by_frame.keys()))

    for frame in all_frames:
        json_targets = json_by_pid.get(frame, [])
        csv_targets = csv_by_frame.get(frame, [])

        # 检查帧是否存在
        if frame not in json_by_pid:
            validation_results['errors'].append(f"帧 {frame}: JSON文件中不存在")
            continue
        if frame not in csv_by_frame:
            validation_results['errors'].append(f"帧 {frame}: CSV文件中不存在")
            continue

        # 检查目标数量
        json_target_count = len(json_targets)
        csv_target_count = len(csv_targets)

        if json_target_count != csv_target_count:
            validation_results['errors'].append(
                f"帧 {frame}: 目标数量不一致 (JSON: {json_target_count}, CSV: {csv_target_count})"
            )

        # 检查每个目标的数据
        for i, (json_target, csv_target) in enumerate(zip(json_targets, csv_targets)):
            target_id = i + 1

            # 检查Distance
            json_distance = json_target["Distance"]
            csv_distance = csv_target["DM_distance"]
            if abs(json_distance - csv_distance) > 0.001:
                validation_results['errors'].append(
                    f"帧 {frame} 目标 {target_id}: Distance不一致 (JSON: {json_distance}, CSV: {csv_distance})"
                )

            # 检查边界框坐标
            json_bounds = json_target["ImageBounds"]
            checks = [
                ('Xmin', 'Top_left_x', 'gt_left'),
                ('Ymin', 'Top_left_y', 'gt_top'),
                ('Xmax', 'Bottom_right_x', 'gt_right'),
                ('Ymax', 'Bottom_right_y', 'gt_bottom')
            ]

            for json_key, csv_key1, csv_key2 in checks:
                json_val = json_bounds[json_key]
                csv_val1 = csv_target[csv_key1]
                csv_val2 = csv_target[csv_key2]

                if json_val != csv_val1:
                    validation_results['errors'].append(
                        f"帧 {frame} 目标 {target_id}: {csv_key1}不一致 (JSON: {json_val}, CSV: {csv_val1})"
                    )

                if json_val != csv_val2:
                    validation_results['errors'].append(
                        f"帧 {frame} 目标 {target_id}: {csv_key2}不一致 (JSON: {json_val}, CSV: {csv_val2})"
                    )

            # 检查中心点计算
            center_x_calculated = (json_bounds["Xmin"] + json_bounds["Xmax"]) / 2
            center_y_calculated = (json_bounds["Ymin"] + json_bounds["Ymax"]) / 2

            if abs(center_x_calculated - csv_target["Center_x"]) > 0.001:
                validation_results['errors'].append(
                    f"帧 {frame} 目标 {target_id}: Center_x计算错误 (计算值: {center_x_calculated}, CSV: {csv_target['Center_x']})"
                )

            if abs(center_y_calculated - csv_target["Center_y"]) > 0.001:
                validation_results['errors'].append(
                    f"帧 {frame} 目标 {target_id}: Center_y计算错误 (计算值: {center_y_calculated}, CSV: {csv_target['Center_y']})"
                )

            # 检查文件名
            expected_img_name = f"output_{frame}.png"
            if csv_target["img_name"] != expected_img_name:
                validation_results['errors'].append(
                    f"帧 {frame} 目标 {target_id}: img_name错误 (期望: {expected_img_name}, 实际: {csv_target['img_name']})"
                )

            # 检查M_FileName
            if csv_target["id"] != json_m_filename:
                validation_results['errors'].append(
                    f"帧 {frame} 目标 {target_id}: id不一致 (JSON: {json_m_filename}, CSV: {csv_target['id']})"
                )

        validation_results['total_frames'] += 1
        validation_results['total_targets'] += len(json_targets)

    # 检查宽高信息
    for _, row in csv_data.iterrows():
        if row['size_width'] != json_width:
            validation_results['warnings'].append(f"size_width不一致 (JSON: {json_width}, CSV: {row['size_width']})")
            break

        if row['size_height'] != json_height:
            validation_results['warnings'].append(f"size_height不一致 (JSON: {json_height}, CSV: {row['size_height']})")
            break

    # 生成摘要
    validation_results['summary'] = {
        'json_file': json_file_path,
        'csv_file': csv_file_path,
        'total_frames_checked': validation_results['total_frames'],
        'total_targets_checked': validation_results['total_targets'],
        'error_count': len(validation_results['errors']),
        'warning_count': len(validation_results['warnings']),
        'is_valid': len(validation_results['errors']) == 0
    }

    return validation_results


def validate_all_files(save_root_dir, label_root_dir):
    """
    校对所有生成的CSV文件与对应的JSON文件

    Args:
        save_root_dir: 保存CSV文件的根目录
        label_root_dir: label文件夹的根目录
    """

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(save_root_dir, "**", "groundtruth.csv"), recursive=True)

    if not csv_files:
        print("未找到任何CSV文件，请先运行转换程序")
        return

    overall_results = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'file_details': [],
        'all_errors': []
    }

    for csv_file in csv_files:
        print(f"\n校对文件: {csv_file}")

        # 解析CSV文件路径，找到对应的JSON文件
        csv_path = Path(csv_file)
        scene_name = csv_path.parent.parent.name
        channel_name = csv_path.parent.name  # ch1, ch2等

        # 构建对应的JSON文件名
        channel_num = channel_name.replace('ch', 'channel')
        json_pattern = os.path.join(label_root_dir, scene_name, f"*_{channel_num}_39nPoint.json")

        json_files = glob.glob(json_pattern)
        if not json_files:
            print(f"警告: 找不到对应的JSON文件: {json_pattern}")
            continue

        json_file = json_files[0]  # 取第一个匹配的文件

        # 进行校对
        results = validate_csv_with_json(csv_file, json_file)

        # 记录总体结果
        overall_results['total_files'] += 1
        overall_results['file_details'].append({
            'scene': scene_name,
            'channel': channel_name,
            'csv_file': csv_file,
            'json_file': json_file,
            'results': results
        })

        if results['summary']['is_valid']:
            overall_results['valid_files'] += 1
            print(f"✓ 验证通过 - 帧数: {results['total_frames']}, 目标数: {results['total_targets']}")
        else:
            overall_results['invalid_files'] += 1
            overall_results['all_errors'].extend(results['errors'])
            print(f"✗ 验证失败 - 错误数: {len(results['errors'])}")

        # 显示详细错误信息
        if results['errors']:
            print("详细错误:")
            for error in results['errors'][:5]:  # 只显示前5个错误
                print(f"  - {error}")
            if len(results['errors']) > 5:
                print(f"  ... 还有 {len(results['errors']) - 5} 个错误")

        if results['warnings']:
            print("警告信息:")
            for warning in results['warnings']:
                print(f"  - {warning}")

    # 生成总体报告
    print("\n" + "=" * 60)
    print("总体校对报告")
    print("=" * 60)
    print(f"总文件数: {overall_results['total_files']}")
    print(f"验证通过: {overall_results['valid_files']}")
    print(f"验证失败: {overall_results['invalid_files']}")
    print(f"总错误数: {len(overall_results['all_errors'])}")

    if overall_results['all_errors']:
        print("\n主要错误类型统计:")
        error_types = {}
        for error in overall_results['all_errors']:
            error_type = error.split(':')[0] if ':' in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {error_type}: {count} 次")

    # 保存详细报告到文件
    report_file = os.path.join(save_root_dir, "validation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("数据转换验证报告\n")
        f.write("=" * 50 + "\n\n")

        for file_detail in overall_results['file_details']:
            f.write(f"场景: {file_detail['scene']}, 通道: {file_detail['channel']}\n")
            f.write(f"CSV文件: {file_detail['csv_file']}\n")
            f.write(f"JSON文件: {file_detail['json_file']}\n")

            results = file_detail['results']
            summary = results['summary']

            f.write(f"状态: {'通过' if summary['is_valid'] else '失败'}\n")
            f.write(f"检查帧数: {summary['total_frames_checked']}\n")
            f.write(f"检查目标数: {summary['total_targets_checked']}\n")
            f.write(f"错误数: {summary['error_count']}\n")
            f.write(f"警告数: {summary['warning_count']}\n")

            if results['errors']:
                f.write("\n错误详情:\n")
                for error in results['errors']:
                    f.write(f"  - {error}\n")

            if results['warnings']:
                f.write("\n警告详情:\n")
                for warning in results['warnings']:
                    f.write(f"  - {warning}\n")

            f.write("\n" + "-" * 50 + "\n\n")

    print(f"\n详细报告已保存至: {report_file}")


def main():
    # 设置路径（与转换程序相同）
    label_root_dir = "/home/csz_changsha/data/AR_2511/label"  # label文件夹的根目录
    save_root_dir = "/home/csz_changsha/data/AR_2511/labels"  # 保存CSV文件的根目录

    # 检查目录是否存在
    if not os.path.exists(save_root_dir):
        print(f"错误: 找不到保存目录: {save_root_dir}")
        print("请先运行转换程序生成CSV文件")
        return

    if not os.path.exists(label_root_dir):
        print(f"错误: 找不到label目录: {label_root_dir}")
        return

    print("开始数据校对...")
    validate_all_files(save_root_dir, label_root_dir)
    print("校对完成！")


if __name__ == "__main__":
    main()
