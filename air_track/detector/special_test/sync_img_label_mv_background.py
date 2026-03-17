# -*- coding: utf-8 -*-
# @Author    :
# @File      : sync_img_label_mv_background.py
# @Created   : 2025/7/18 上午11:23
# @Desc      : 同步数据与标签是否一一对应，并将没有对应标签的图像移除到指定路径下，但不创建标签文件

import os
import shutil


def find_and_copy_missing_images(rgb_folder, label_folder, output_folder, rgb_ext='.jpg', label_ext='.txt'):
    """
    找出rgb中存在但label中不存在的图像，并复制到指定路径

    参数:
        rgb_folder: rgb图像文件夹路径
        label_folder: label图像文件夹路径
        output_folder: 输出文件夹路径
        rgb_ext: rgb图像的扩展名
        label_ext: label图像的扩展名
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取rgb和label文件夹中的文件名（不带扩展名）
    rgb_files = {os.path.splitext(f)[0] for f in os.listdir(rgb_folder) if f.endswith(rgb_ext)}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.endswith(label_ext)}

    # 找出rgb中存在但label中不存在的文件
    missing_files = rgb_files - label_files

    print(f"找到 {len(missing_files)} 个在rgb中但不在label中的文件")

    # 复制这些文件到输出文件夹
    for file_base in missing_files:
        src_file = os.path.join(rgb_folder, file_base + rgb_ext)
        dst_file = os.path.join(output_folder, file_base + rgb_ext)

        shutil.move(src_file, dst_file)
        print(f"已移动: {src_file} -> {dst_file}")

    print("操作完成！")


# 使用示例
if __name__ == "__main__":
    rgb_folder = '/home/csz_changsha/data/tmg_train_data/20260115_tmg_M300_afternoon/2_1/images'  # 替换为你的rgb文件夹路径
    label_folder = '/home/csz_changsha/data/tmg_train_data/20260115_tmg_M300_afternoon/2_1/labels'  # 替换为你的label文件夹路径
    output_folder = '/home/csz_changsha/data/tmg_train_data/20260115_tmg_M300_afternoon/2_1/temp'  # 替换为你想保存的路径

    # 如果扩展名不同，可以在这里修改
    # 例如，rgb是.jpg，label是.png，则改为 rgb_ext='.jpg', label_ext='.png'
    find_and_copy_missing_images(rgb_folder, label_folder, output_folder, rgb_ext='.png', label_ext='.txt')