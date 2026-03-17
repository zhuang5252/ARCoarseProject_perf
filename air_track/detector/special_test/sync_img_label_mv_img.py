# -*- coding: utf-8 -*-
# @Author    :
# @File      : sync_img_label_mv_img.py
# @Created   : 2025/7/18 上午10:23
# @Desc      : 同步数据与标签是否一一对应，并将没有对应标签的图像移除到指定路径下，并在新路径下为其创建空标签文件

import os
import glob
import shutil
from typing import List, Dict, Tuple


def find_all_data_dirs(root_dir: str, img_folder: str, label_folder: str) -> List[Tuple[str, str, str]]:
    """
    查找所有包含图片和标签的目录

    Args:
        root_dir: 根目录
        img_folder: 图片文件夹名
        label_folder: 标签文件夹名

    Returns:
        返回三元组列表：(数据目录路径, 图片目录路径, 标签目录路径)
    """
    data_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if img_folder in dirnames and label_folder in dirnames:
            data_dir = os.path.relpath(dirpath, root_dir)
            img_dir = os.path.join(dirpath, img_folder)
            label_dir = os.path.join(dirpath, label_folder)
            data_dirs.append((data_dir, img_dir, label_dir))
    return data_dirs


def process_directory(
        total_counter: int,
        img_folder: str,
        label_folder: str,
        data_dir: str,
        img_dir: str,
        label_dir: str,
        output_root: str,
        all_bg_rgb: str,
        all_bg_label: str,
        img_formats: List[str],
        label_format: str,
        dry_run: bool,
        verbose: bool
) -> int:
    """
    处理单个目录

    Returns:
        处理的图片数量
    """
    counter = 0

    # 查找当前目录下的所有图片和标签
    img_files = []
    for ext in img_formats:
        img_files.extend(glob.glob(os.path.join(img_dir, '*' + ext)))

    label_files = glob.glob(os.path.join(label_dir, '*.' + label_format))

    # 获取文件名（不带扩展名）
    img_names = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
    label_names = {os.path.splitext(os.path.basename(f))[0] for f in label_files}

    # 找出不匹配的图像（图片有但标签没有）
    unmatched_images = img_names - label_names

    if verbose:
        print(f"\nProcessing directory: {data_dir}")
        print(f"  Found {len(img_files)} images and {len(label_files)} labels")
        print(f"  Found {len(unmatched_images)} unmatched images")

    # 处理不匹配的图像
    for img_path in img_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if img_name not in unmatched_images:
            continue

        # 原始文件名和扩展名
        img_ext = os.path.splitext(img_path)[1]

        # 在输出目录中保持相同结构
        rel_path = os.path.relpath(img_path, img_dir)
        output_img_dir = os.path.join(output_root, data_dir, img_folder)
        output_img_path = os.path.join(output_img_dir, os.path.basename(img_path))
        output_label_dir = os.path.join(output_root, data_dir, label_folder)
        output_label_path = os.path.join(output_label_dir, f"{img_name}.{label_format}")

        # all_background中的路径
        counter += 1
        total_counter += 1
        all_bg_img_path = os.path.join(all_bg_rgb, f"{total_counter}{img_ext}")
        all_bg_label_path = os.path.join(all_bg_label, f"{total_counter}.{label_format}")

        if verbose:
            print(f"\n  Processing image: {img_name}{img_ext}")
            print(f"    Original image: {img_path}")
            print(f"    Moving to: {output_img_path}")
            print(f"    Creating label: {output_label_path}")
            print(f"    All_background image: {all_bg_img_path}")
            print(f"    All_background label: {all_bg_label_path}")

        if not dry_run:
            # 移动原始图像
            os.makedirs(output_img_dir, exist_ok=True)
            shutil.copy2(img_path, output_img_path)

            # 创建标签文件
            # os.makedirs(output_label_dir, exist_ok=True)
            # with open(output_label_path, 'w') as f:
            #     pass

            # 添加到all_background
            shutil.move(img_path, all_bg_img_path)
            with open(all_bg_label_path, 'w') as f:
                pass

    return total_counter


def sync_multilevel_dirs(
        root_dir: str,
        output_dir: str,
        img_folder: str = 'rgb',
        label_folder: str = 'label',
        img_formats: List[str] = ['.jpg', '.png'],
        label_format: str = 'txt',
        dry_run: bool = False,
        verbose: bool = True
) -> None:
    """
    同步多级目录中的图片和标签

    Args:
        root_dir: 根目录（包含多个子目录，每个子目录下有 img_folder 和 label_folder）
        output_dir: 输出目录（将在此目录下创建结构）
        img_folder: 图片文件夹名
        label_folder: 标签文件夹名
        img_formats: 图片格式列表
        label_format: 标签格式
        dry_run: 只打印变化，不实际修改文件
        verbose: 是否打印详细信息
    """
    # 准备all_background目录
    all_bg_dir = os.path.join(output_dir, 'all_background')
    all_bg_rgb = os.path.join(all_bg_dir, img_folder)
    all_bg_label = os.path.join(all_bg_dir, label_folder)

    if not dry_run:
        os.makedirs(all_bg_rgb, exist_ok=True)
        os.makedirs(all_bg_label, exist_ok=True)

    # 查找所有需要处理的目录
    data_dirs = find_all_data_dirs(root_dir, img_folder, label_folder)

    if verbose:
        print(f"Found {len(data_dirs)} directories to process:")
        for data_dir, _, _ in data_dirs:
            print(f"  {data_dir}")

    total_counter = 0

    # 处理每个目录
    for data_dir, img_dir, label_dir in data_dirs:
        total_counter = process_directory(
            total_counter=total_counter,
            img_folder=img_folder,
            label_folder=label_folder,
            data_dir=data_dir,
            img_dir=img_dir,
            label_dir=label_dir,
            output_root=output_dir,
            all_bg_rgb=all_bg_rgb,
            all_bg_label=all_bg_label,
            img_formats=img_formats,
            label_format=label_format,
            dry_run=dry_run,
            verbose=verbose
        )

    if verbose:
        print(f"\nTotal processed {total_counter} unmatched images")
        print(f"All background files created in: {all_bg_dir}")



if __name__ == '__main__':
    # 示例用法
    sync_multilevel_dirs(
        root_dir='/home/csz_xishu2/data/zbzx_data/version_2/zbzx_orig_data/20260305_jy_itv',  # 原始数据根目录
        output_dir='/home/csz_xishu2/data/zbzx_data/version_2/zbzx_orig_data/20260305_jy_itv_no_label',  # 输出目录
        img_folder='images',  # 图片文件夹名
        label_folder='labels',  # 标签文件夹名
        img_formats=['.jpg', '.png'],  # 图片格式
        label_format='txt',  # 标签格式
        dry_run=False,  # 设为True可以预览变化而不实际修改
        verbose=True
    )
