# -*- coding: utf-8 -*-
# @Author    : 
# @File      : merge_sub_AR_data.py
# @Created   : 2025/08/28 17:13
# @Desc      : 根据CSV文件合并多个子目录的数据集

"""
根据CSV文件中的信息合并多个子目录的数据集
1. 从每个子目录的ImageSets/groundtruth.csv读取数据
2. 根据csv中的flight_id和img_name字段构建图片路径
3. 复制图片到目标目录并重命名
4. 合并所有CSV数据，更新img_name和文件路径
"""

import os
import csv
import shutil
import argparse
from pathlib import Path


def read_csv_file(csv_path):
    """读取CSV文件"""
    data = []
    if not csv_path.exists():
        print(f"警告: CSV文件 {csv_path} 不存在")
        return data

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"成功读取CSV文件: {csv_path}, 共 {len(data)} 行数据")
    except Exception as e:
        print(f"读取CSV文件 {csv_path} 时出错: {e}")

    return data


def find_image_path(src_root, flight_id, img_name):
    """根据flight_id和img_name查找图片路径"""
    # 可能的图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 构建基础路径
    base_path = Path(src_root) / "Images" / flight_id

    # 如果路径不存在，尝试直接查找
    if not base_path.exists():
        print(f"警告: 路径 {base_path} 不存在")
        return None

    # 首先尝试直接匹配文件名（包含扩展名）
    for ext in image_extensions:
        img_path = base_path / img_name
        if img_path.exists():
            return img_path

    # 如果不带扩展名，尝试添加扩展名
    for ext in image_extensions:
        img_path = base_path / f"{img_name}{ext}"
        if img_path.exists():
            return img_path

    # 尝试查找不带扩展名的匹配
    for file in base_path.iterdir():
        if file.stem == img_name and file.suffix.lower() in image_extensions:
            return file

    return None


def process_subdirectory(sub_dir, dst_img_dir, dst_label_dir, counter, csv_writer=None):
    """
    处理单个子目录

    Args:
        sub_dir: 源子目录路径
        dst_img_dir: 目标图片目录
        dst_label_dir: 目标标签目录
        counter: 当前计数器
        csv_writer: CSV写入器（用于累积写入）

    Returns:
        counter: 更新后的计数器
        processed_count: 处理的图片数量
        total_count: CSV中的总行数
    """
    print(f"\n处理子目录: {sub_dir}")

    # 读取CSV文件
    csv_path = Path(sub_dir) / "ImageSets" / "groundtruth.csv"
    csv_data = read_csv_file(csv_path)

    if not csv_data:
        print(f"子目录 {sub_dir} 中没有CSV数据，跳过")
        return counter, 0, 0

    processed_count = 0
    total_count = len(csv_data)

    # 处理CSV中的每一行
    for i, row in enumerate(csv_data, 1):
        flight_id = row.get('flight_id', '')
        img_name = row.get('img_name', '')

        if not flight_id or not img_name:
            print(f"警告: 第 {i} 行缺少flight_id或img_name字段，跳过")
            continue

        # 查找图片
        img_path = find_image_path(sub_dir, flight_id, img_name)

        if not img_path or not img_path.exists():
            print(f"警告: 找不到图片 {sub_dir}/Images/{flight_id}/{img_name}，跳过")
            continue

        # 生成新的文件名
        new_name = f"{counter:06d}"
        new_img_name = f"{new_name}{img_path.suffix}"

        # 复制图片
        new_img_path = dst_img_dir / new_img_name
        try:
            shutil.copy2(img_path, new_img_path)
            print(f"复制: {img_path.name} -> {new_img_name}")
        except Exception as e:
            print(f"复制图片 {img_path} 时出错: {e}")
            continue

        # 更新CSV行
        row['Channel'] = 'ch'  # 统一Channel为'ch'
        row['flight_id'] = 'ch'  # 统一flight_id为'ch'
        row['img_name'] = new_img_name

        # 写入到合并的CSV（如果提供了writer）
        if csv_writer:
            csv_writer.writerow(row)

        counter += 1
        processed_count += 1

    print(f"子目录 {sub_dir} 处理完成: {processed_count}/{total_count} 个文件")
    return counter, processed_count, total_count


def merge_datasets(src_roots, dst_root, img_dir="Images", label_dir="ImageSets", mode="auto"):
    """
    合并多个数据集目录

    Args:
        src_roots: 源数据根目录列表
        dst_root: 目标根目录
        img_dir: 目标图片目录名
        label_dir: 目标标签目录名
        mode: 合并模式，'auto'或'manual'
    """
    # 创建目标目录
    dst_img = Path(dst_root) / img_dir / "ch"
    dst_label = Path(dst_root) / label_dir
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_label.mkdir(parents=True, exist_ok=True)

    # 准备要处理的子目录列表
    subdirectories = []

    if mode == "auto":
        # 自动模式：处理每个src_root下的所有子目录
        for src_root in src_roots:
            src_path = Path(src_root)
            if src_path.is_dir():
                # 遍历所有子目录
                for item in src_path.iterdir():
                    if item.is_dir():
                        # 检查是否包含必要的子目录结构
                        images_dir = item / "Images"
                        imagesets_dir = item / "ImageSets"
                        if images_dir.exists() and imagesets_dir.exists():
                            subdirectories.append(item)
                        else:
                            print(f"警告: {item.name} 缺少Images或ImageSets目录，跳过")
            else:
                print(f"警告: {src_root} 不是目录，跳过")
    else:
        # 手动模式：src_roots本身就是要处理的子目录
        subdirectories = [Path(src_root) for src_root in src_roots]

    if not subdirectories:
        print("错误: 没有找到要处理的子目录")
        return

    print(f"找到 {len(subdirectories)} 个子目录需要处理:")
    for sub_dir in subdirectories:
        print(f"  - {sub_dir}")

    # 准备合并的CSV文件
    csv_file_path = dst_label / "groundtruth.csv"
    csv_fieldnames = None
    counter = 1
    total_processed = 0
    total_in_csv = 0

    # 首先读取一个CSV文件获取字段名
    for sub_dir in subdirectories:
        csv_path = Path(sub_dir) / "ImageSets" / "groundtruth.csv"
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_fieldnames = reader.fieldnames
                break

    if not csv_fieldnames:
        print("错误: 无法确定CSV字段名")
        return

    # 写入CSV文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()

        # 处理每个子目录
        for sub_dir in subdirectories:
            counter, processed_count, total_count = process_subdirectory(
                sub_dir, dst_img, dst_label, counter, writer
            )
            total_processed += processed_count
            total_in_csv += total_count

    print(f"\n{'=' * 50}")
    print("处理完成!")
    print(f"共处理 {len(subdirectories)} 个子目录")
    print(f"CSV总行数: {total_in_csv}")
    print(f"成功处理图片数: {total_processed}")
    print(f"目标图片目录: {dst_img}")
    print(f"目标CSV文件: {csv_file_path}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description='合并多个子目录的数据集')
    parser.add_argument('--mode', type=str, choices=['auto', 'manual'], default='auto',
                        help='合并模式，可选 "auto（合并目录下的所有子目录）" 或 "manual（合并指定的子目录）"。默认为 "auto"')
    parser.add_argument('--src_roots', nargs='+', default=['/home/lx/csz/data/AR_2410_2511/AR_2410_test_data'],
                        help='源数据根目录路径列表，例如: ./scne31_ch3 ./scne31_ch4')
    parser.add_argument('--dst', type=str, default='/home/lx/csz/data/AR_2410_2511/AR_2410_test_data_merge',
                        help='目标根目录路径，例如: ./merged_datasets')
    parser.add_argument('--img', type=str, default='Images', help='目标img目录名')
    parser.add_argument('--label', type=str, default='ImageSets', help='目标标签目录名')

    args = parser.parse_args()

    # 验证源目录是否存在
    for src_root in args.src_roots:
        if not Path(src_root).exists():
            print(f"错误: 源目录 {src_root} 不存在")
            return

    merge_datasets(args.src_roots, args.dst, args.img, args.label, args.mode)


if __name__ == "__main__":
    # 直接运行示例
    # merge_datasets(["./scne31_ch3", "./scne31_ch4"], "./merged_datasets", mode="auto")

    # 使用命令行参数运行
    main()
