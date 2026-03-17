# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - pad_img.py
# @Created   : 2025/08/12 17:08
# @Desc      : 在小图尺寸上填充黑边到指定尺寸

"""

"""
import os
import cv2
import numpy as np
from tqdm import tqdm  # 进度条工具（可选）


def center_pad_image(img, target_width=1024, target_height=768):
    """
    将图像中心填充到目标尺寸（四周填充黑色）
    参数:
        img: 输入图像 (H,W,3)
        target_width: 目标宽度
        target_height: 目标高度
    返回:
        填充后的图像 (target_height, target_width, 3)
    """
    h, w = img.shape[:2]

    # 计算需要填充的像素数（上下左右对称填充）
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left

    # 执行填充（BGR格式的黑色= [0,0,0]）
    padded_img = cv2.copyMakeBorder(
        img,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # 黑色填充
    )

    return padded_img


def process_folder(input_dir, output_dir, orig_img_shape=(512, 640, 3), target_width=1024, target_height=768):
    """
    处理整个文件夹中的图片
    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有支持的图片文件
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    # 处理每张图片（带进度条）
    for filename in tqdm(image_files, desc="Processing Images"):
        # 读取图像
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # 检查是否为640x512x3（可选）
        if img.shape != orig_img_shape:
            print(f"警告: {filename} 的尺寸是 {img.shape}，不是预期的 {orig_img_shape}")

        # 中心填充
        padded_img = center_pad_image(img, target_width, target_height)

        # 保存结果
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, padded_img)


if __name__ == "__main__":
    # 配置路径
    input_dir = "/media/linana/2C5821EF5821B88A/0809_tir/infrared_split/test/6/infrared"  # 替换为你的输入文件夹
    output_dir = "/media/linana/2C5821EF5821B88A/0809_tir/infrared_split/test/6_pad/infrared"  # 替换为你的输出文件夹

    # 运行处理
    process_folder(input_dir, output_dir, orig_img_shape=(512, 640, 3), target_width=1024, target_height=768)
    print(f"所有图片已处理完成，保存到: {output_dir}")