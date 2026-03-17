# -*- coding: utf-8 -*-
# @Author    : 
# @File      : process_AR2511_json_to_csv.py
# @Created   : 2025/11/24 上午10:08
# @Desc      : 将AR2511的JSON文件标签转换为csv格式

"""
将AR2511的JSON文件标签转换为csv格式
"""

import os
import json
import pandas as pd
import glob
from pathlib import Path


def process_json_to_csv(label_root_dir, save_root_dir):
    """
    将label文件夹中的所有JSON文件转换为CSV格式

    Args:
        label_root_dir: label文件夹的根目录路径
        save_root_dir: 保存CSV文件的根目录路径
    """

    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(label_root_dir, "**", "*_39nPoint.json"), recursive=True)

    for json_file in json_files:
        print(f"处理文件: {json_file}")

        # 解析文件路径，提取scene和channel信息
        file_path = Path(json_file)
        scene_name = file_path.parent.name  # 例如: scne1
        channel_name = file_path.stem.split('_')[-2]  # 例如: channel1 -> ch1

        # 提取channel编号 (去掉'channel'前缀)
        channel_num = channel_name.replace('channel', 'ch')

        # 创建对应的保存目录
        save_dir = Path(save_root_dir) / scene_name / channel_num
        save_dir.mkdir(parents=True, exist_ok=True)

        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取基本信息
        width = data["Head"]["Width"]
        height = data["Head"]["Height"]
        m_filename = data["Head"]["M_FileName"]

        # 准备存储所有目标的数据
        all_targets_data = []

        # 处理每个标注信息  TODO 若当前帧无目标，根据未来所给标签再去调整
        for biaozhu_info in data["Body"]["BiaoZhuInfo"]:
            distance = biaozhu_info["Distance"]
            image_bounds = biaozhu_info["ImageBounds"]
            pid = biaozhu_info["Pid"]

            # 计算中心点坐标
            center_x = (image_bounds["Xmin"] + image_bounds["Xmax"]) / 2
            center_y = (image_bounds["Ymin"] + image_bounds["Ymax"]) / 2

            # 构建目标数据 - 只填充指定的字段，其余留空
            target_data = {
                "Channel": channel_num,
                "Img_id": pid,
                "X": "",  # 留空
                "Y": "",  # 留空
                "Z": "",  # 留空
                "Pos_angle": "",  # 留空
                "Pitch_angle": "",  # 留空
                "Roll_angle": "",  # 留空
                "Min_brightness": "",  # 留空
                "Max_brightness": "",  # 留空
                "Target_conut": 1,  # 每个目标单独记录，后面会调整
                "Target_id": 1,  # 后面会根据实际数量调整
                "Top_left_x": image_bounds["Xmin"],
                "Top_left_y": image_bounds["Ymin"],
                "Bottom_right_x": image_bounds["Xmax"],
                "Bottom_right_y": image_bounds["Ymax"],
                "Center_x": center_x,
                "Center_y": center_y,
                "DM_distance": distance,
                "flight_id": channel_num,
                "img_name": f"output_{pid}.png",  # 格式化为6位数
                "frame": pid,
                "cls_name": 'object',
                "range_distance_m": distance,
                "is_above_horizon": "",  # 留空
                "size_width": width,
                "size_height": height,
                "gt_left": image_bounds["Xmin"],
                "gt_top": image_bounds["Ymin"],
                "gt_right": image_bounds["Xmax"],
                "gt_bottom": image_bounds["Ymax"],
                "User_channel_1": ""  # 留空
            }

            all_targets_data.append(target_data)

        # 按帧号(Pid)分组，统计每帧的目标数量并分配Target_id
        frame_groups = {}
        for target in all_targets_data:
            frame = target["frame"]
            if frame not in frame_groups:
                frame_groups[frame] = []
            frame_groups[frame].append(target)

        # 更新每帧的目标数量和目标ID
        final_data = []
        for frame, targets in frame_groups.items():
            target_count = len(targets)
            for i, target in enumerate(targets, 1):
                target["Target_conut"] = target_count
                target["Target_id"] = i
                final_data.append(target)

        # 创建DataFrame
        df = pd.DataFrame(final_data)

        # 定义CSV文件的完整列顺序（保持原有表头结构）
        columns_order = [
            "Channel", "Img_id", "X", "Y", "Z", "Pos_angle", "Pitch_angle", "Roll_angle",
            "Min_brightness", "Max_brightness", "Target_conut", "Target_id",
            "Top_left_x", "Top_left_y", "Bottom_right_x", "Bottom_right_y",
            "Center_x", "Center_y", "DM_distance", "flight_id", "img_name", "frame",
            "cls_name", "range_distance_m", "is_above_horizon", "size_width", "size_height",
            "gt_left", "gt_top", "gt_right", "gt_bottom"
        ]

        # 按指定顺序排列列
        df = df[columns_order]

        # 保存为CSV文件
        csv_file_path = save_dir / "groundtruth.csv"
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"已保存: {csv_file_path}")

        # 显示处理结果摘要
        print(f"场景: {scene_name}, 通道: {channel_num}")
        print(f"处理帧数: {len(frame_groups)}, 总目标数: {len(final_data)}")
        print("-" * 50)


def main():
    # 设置路径
    label_root_dir = "/home/csz_changsha/data/AR_2511/label"  # label文件夹的根目录
    save_root_dir = "/home/csz_changsha/data/AR_2511/labels"  # 保存结果的根目录

    # 检查label目录是否存在
    if not os.path.exists(label_root_dir):
        print(f"错误: 找不到label目录: {label_root_dir}")
        print("请确保label目录存在并包含正确的文件夹结构")
        return

    print("开始处理JSON文件...")
    process_json_to_csv(label_root_dir, save_root_dir)
    print("处理完成！")


if __name__ == "__main__":
    main()
