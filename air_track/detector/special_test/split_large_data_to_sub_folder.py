# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - split_data.py
# @Created   : 2025/08/10 22:07
# @Desc      : 将一个庞大的数据集文件夹，拆分为一个个小文件夹

"""

"""

import os
import shutil

# 配置路径
source_dir = "/media/linana/2C5821EF5821B88A/0727_T18_infrared/0809_tir2/infrared"  # 源图片文件夹
target_root = "/media/linana/2C5821EF5821B88A/0727_T18_infrared/0809_tir2/infrared_split"  # 目标根目录
images_per_folder = 8500  # 每个子文件夹存放的图片数量

# 获取所有图片文件（支持常见格式）
image_files = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
]

# 按数量拆分到子文件夹
for i in range(0, len(image_files), images_per_folder):
    batch = image_files[i:i + images_per_folder]
    folder_num = i // images_per_folder + 1  # 子文件夹编号（从1开始）
    target_folder = os.path.join(target_root, str(folder_num), "infrared")

    # 创建目标文件夹（如 "1/infrared"）
    os.makedirs(target_folder, exist_ok=True)

    # 移动图片
    for img in batch:
        src = os.path.join(source_dir, img)
        dst = os.path.join(target_folder, img)
        shutil.move(src, dst)  # 或 shutil.copy2(src, dst) 复制而非移动

print(f"拆分完成：共 {len(image_files)} 张图片，存放到 {len(image_files) // images_per_folder + 1} 个子文件夹。")
