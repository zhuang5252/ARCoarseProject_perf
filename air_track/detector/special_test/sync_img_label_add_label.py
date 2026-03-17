# -*- coding: utf-8 -*-
# @Author    :
# @File      : sync_img_label_add_label.py
# @Created   : 2025/7/18 下午11:23
# @Desc      : 同步数据与标签是否一一对应，并对没有标签的图像创建新的空标签文件，并对多余的标签文件进行删除操作

import os
import glob
from typing import List, Optional


def find_files(
        root_dir: str,
        folder_name: str,
        extensions: List[str],
        recursive: bool = True
) -> List[str]:
    """
    查找指定目录下符合扩展名的文件

    Args:
        root_dir: 根目录
        folder_name: 要查找的文件夹名（如 'rgb'）
        extensions: 文件扩展名列表（如 ['.jpg', '.png']）
        recursive: 是否递归查找子目录

    Returns:
        文件路径列表（相对路径）
    """
    pattern = "**" if recursive else "*"
    files = []
    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(root_dir, folder_name, pattern + ext),
                recursive=recursive
            )
        )
    # 转换为相对路径（相对于 root_dir/folder_name）
    return [os.path.relpath(f, os.path.join(root_dir, folder_name)) for f in files]


def sync_img_and_label(
        root_dir: str,
        img_folder: str = 'rgb',
        label_folder: str = 'label',
        img_formats: List[str] = ['.jpg', '.png'],
        label_format: str = 'txt',
        dry_run: bool = False,
        verbose: bool = True
) -> None:
    """
    同步图片和标签文件

    Args:
        root_dir: 根目录（包含 img_folder 和 label_folder）
        img_folder: 图片文件夹名
        label_folder: 标签文件夹名
        img_formats: 图片格式列表
        label_format: 标签格式
        dry_run: 只打印变化，不实际修改文件
        verbose: 是否打印详细信息
    """
    # 1. 查找所有图片和标签文件
    img_files = find_files(root_dir, img_folder, img_formats)
    label_files = find_files(root_dir, label_folder, [f'.{label_format}'])

    # 获取文件名（不带扩展名）
    img_names = {os.path.splitext(f)[0] for f in img_files}
    label_names = {os.path.splitext(f)[0] for f in label_files}

    # 2. 找出需要创建的空标签和需要删除的标签
    # 图片有但标签没有 -> 创建空标签
    to_create = img_names - label_names
    # 标签有但图片没有 -> 删除标签（排除 classes.txt）
    to_delete = label_names - img_names - {'classes'}

    if verbose:
        print(f"Found {len(img_files)} images and {len(label_files)} labels")
        print(f"Will create {len(to_create)} empty labels")
        print(f"Will delete {len(to_delete)} redundant labels")

    # 3. 创建缺失的标签文件
    for name in to_create:
        label_path = os.path.join(
            root_dir, label_folder, f"{name}.{label_format}"
        )
        if verbose:
            print(f"Creating empty label: {label_path}")
        if not dry_run:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, 'w') as f:
                pass  # 创建空文件

    # 4. 删除多余的标签文件
    for name in to_delete:
        label_path = os.path.join(
            root_dir, label_folder, f"{name}.{label_format}"
        )
        if verbose:
            print(f"Deleting redundant label: {label_path}")
        if not dry_run:
            os.remove(label_path)

    if verbose:
        print("Sync completed!")


if __name__ == '__main__':
    path = '/home/csz_xishu2/data/zbzx_data/version_2/精可见/20260305_jy_ftv'

    """包含子目录的迭代处理"""
    for sub_dir in os.listdir(path):
        if not os.path.isdir(os.path.join(path, sub_dir)):
            continue
        # 同步图片和标签
        root_dir = os.path.join(path, sub_dir)
        sync_img_and_label(
            root_dir=root_dir,  # 数据根目录
            img_folder='images',  # 图片文件夹名
            label_folder='labels',  # 标签文件夹名
            img_formats=['.jpg', '.png'],  # 图片格式
            label_format='txt',  # 标签格式
            dry_run=False,  # 设为 True 可以预览变化而不实际修改
            verbose=True
        )

    """无子目录的直接处理"""
    # 同步图片和标签
    # root_dir = '/media/linana/2C5821EF5821B88A/yanqing_train_data/0908_second_classifier_dataset/123/JDModel_normal_0917_FC30_second'
    # sync_img_and_label(
    #     root_dir=root_dir,  # 数据根目录
    #     img_folder='img',  # 图片文件夹名
    #     label_folder='npy',  # 标签文件夹名
    #     img_formats=['.jpg'],  # 图片格式
    #     label_format='npy',  # 标签格式
    #     dry_run=False,  # 设为 True 可以预览变化而不实际修改
    #     verbose=True
    # )

    # 同步图片和标签
    # root_dir = '/home/csz_changsha/data/tmg_train_data/20260119_tmg_M300_night_background_error'
    # sync_img_and_label(
    #     root_dir=root_dir,  # 数据根目录
    #     img_folder='images',  # 图片文件夹名
    #     label_folder='labels',  # 标签文件夹名
    #     img_formats=['.jpg', '.png'],  # 图片格式
    #     label_format='txt',  # 标签格式
    #     dry_run=False,  # 设为 True 可以预览变化而不实际修改
    #     verbose=True
    # )
