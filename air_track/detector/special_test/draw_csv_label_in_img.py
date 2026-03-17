# -*- coding: utf-8 -*-
# @Author    : 
# @File      : draw_csv_label_in_img.py
# @Created   : 2025/11/26 上午11:38
# @Desc      : 将csv格式的数据标签绘制在图像上可视化，并保存到新的文件夹中

"""
将aot/AR_csv格式的数据标签绘制在图像上可视化，并保存到新的文件夹中
"""

import os
import shutil
import pandas as pd
from air_track.detector.visualization.visualize_and_save import visualize_and_save


def process_scene_data(root_dir, output_root_dir):
    """
    处理所有场景的数据，保留原有文件结构

    参数:
        root_dir: 原始数据根目录
        output_root_dir: 输出数据根目录
    """
    # 遍历根目录下的所有场景文件夹
    for scene_dir in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene_dir)

        # 检查是否是目录（跳过文件）
        if not os.path.isdir(scene_path):
            continue

        print(f"处理场景: {scene_dir}")

        # 构建对应的CSV文件路径
        csv_path = os.path.join(scene_path, "ImageSets", "groundtruth.csv")

        # 检查CSV文件是否存在
        if not os.path.exists(csv_path):
            print(f"跳过 {scene_dir}，未找到 groundtruth.csv")
            continue

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            print(f"读取 {scene_dir} 的CSV文件，共 {len(df)} 条记录")
        except Exception as e:
            print(f"读取 {scene_dir} 的CSV文件失败: {e}")
            continue

        # 检查必要的列是否存在
        required_columns = ['img_name', 'DM_distance', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"{scene_dir} 的CSV文件中缺少必要的列: {missing_columns}")
            continue

        # 处理该场景下的每张图像
        for index, row in df.iterrows():
            try:
                img_name = row['img_name']

                # 构建原始图像路径（根据您的文件结构推断）
                # 假设图像路径为: scne1_ch1/Images/ch1/output_1.png
                # flight_id = scene_dir.split('_')[-1]  # 获取视频序列
                flight_id = row['flight_id']  # 获取视频序列
                img_relative_path = os.path.join("Images", flight_id, img_name)
                original_img_path = os.path.join(scene_path, img_relative_path)

                # 检查图像文件是否存在
                if not os.path.exists(original_img_path):
                    print(f"图像文件不存在: {original_img_path}")
                    continue

                # 构建输出路径（保留原有结构）
                output_scene_path = os.path.join(output_root_dir, scene_dir)
                output_img_dir = os.path.join(output_scene_path, "Images", flight_id)

                # 准备可视化参数
                bbox_gts = [[0, row['gt_left'], row['gt_top'], row['gt_right'], row['gt_bottom']]]  # 添加类别ID
                distance = row['DM_distance']

                # 调用可视化函数
                visualize_and_save(
                    img_path=original_img_path,
                    bbox_gts=bbox_gts,
                    distance=distance,
                    visualize_save_dir=output_img_dir
                )

            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {e}")
                continue

        # 复制CSV文件到输出目录（保持结构完整）
        output_csv_dir = os.path.join(output_root_dir, scene_dir, "ImageSets")
        os.makedirs(output_csv_dir, exist_ok=True)
        output_csv_path = os.path.join(output_csv_dir, "groundtruth.csv")
        shutil.copy2(csv_path, output_csv_path)
        print(f"已复制CSV文件到: {output_csv_path}")

    print("所有场景处理完成！")


def process_single_scene(scene_dir, root_dir, output_root_dir):
    """
    处理单个场景的数据
    """
    scene_path = os.path.join(root_dir, scene_dir)

    if not os.path.isdir(scene_path):
        print(f"场景目录不存在: {scene_path}")
        return

    print(f"处理场景: {scene_dir}")

    # 构建对应的CSV文件路径
    csv_path = os.path.join(scene_path, "ImageSets", "groundtruth.csv")

    if not os.path.exists(csv_path):
        print(f"跳过 {scene_dir}，未找到 groundtruth.csv")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"读取 {scene_dir} 的CSV文件，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取 {scene_dir} 的CSV文件失败: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ['img_name', 'DM_distance', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"{scene_dir} 的CSV文件中缺少必要的列: {missing_columns}")
        return

    processed_count = 0
    error_count = 0

    for index, row in df.iterrows():
        try:
            img_name = row['img_name']

            # 构建原始图像路径
            # flight_id = scene_dir.split('_')[-1]  # 获取视频序列
            flight_id = row['flight_id']  # 获取视频序列
            img_relative_path = os.path.join("Images", flight_id, img_name)
            original_img_path = os.path.join(scene_path, img_relative_path)

            if not os.path.exists(original_img_path):
                print(f"图像文件不存在: {original_img_path}")
                error_count += 1
                continue

            # 构建输出路径
            output_scene_path = os.path.join(output_root_dir, scene_dir)
            output_img_dir = os.path.join(output_scene_path, "Images", flight_id)

            # 准备GT边界框数据（格式：[class_id, x1, y1, x2, y2]）
            bbox_gts = [[0, row['gt_left'], row['gt_top'], row['gt_right'], row['gt_bottom']]]
            distance = row['DM_distance']

            # 调用可视化函数
            visualize_and_save(
                img_path=original_img_path,
                bbox_gts=bbox_gts,
                distance=distance,
                visualize_save_dir=output_img_dir
            )

            processed_count += 1

        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {e}")
            error_count += 1
            continue

    # 复制CSV文件
    output_csv_dir = os.path.join(output_root_dir, scene_dir, "ImageSets")
    os.makedirs(output_csv_dir, exist_ok=True)
    output_csv_path = os.path.join(output_csv_dir, "groundtruth.csv")
    shutil.copy2(csv_path, output_csv_path)

    print(f"场景 {scene_dir} 处理完成: 成功 {processed_count}, 失败 {error_count}")
    print(f"CSV文件已复制到: {output_csv_path}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    original_root_dir = "/home/lx/csz/data/AR_2410_2511/all_data"  # 原始数据根目录
    new_output_root_dir = "/home/lx/csz/data/AR_2410_2511/all_data_show"  # 新输出根目录

    # 方法1: 处理所有场景
    print("开始处理所有场景...")
    process_scene_data(original_root_dir, new_output_root_dir)

    # 方法2: 处理单个场景（如果需要）
    # specific_scene = "scne1_ch1"  # 指定要处理的场景
    # process_single_scene(specific_scene, original_root_dir, new_output_root_dir)
