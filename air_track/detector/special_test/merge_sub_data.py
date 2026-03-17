# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - mearge_sub_data.py
# @Created   : 2025/08/28 17:13
# @Desc      : 合并子目录数据集并重命名图片和标签

"""
若一个数据集存在多个子目录，将其合并为一个新的文件夹，其内部图片和标签均重新命名
"""
import os
import shutil
from pathlib import Path
import argparse


def get_matching_pairs(img_dir, label_dir):
    """获取严格匹配的图片-标签对"""
    pairs = []

    # 获取不带扩展名的文件名作为匹配依据
    img_files = {f.stem: f for f in img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']}
    label_files = {f.stem: f for f in label_dir.glob('*') if f.suffix.lower() in ['.txt', '.json', '.xml']}

    # 找出共同的文件名（不带扩展名）
    common_stems = set(img_files.keys()) & set(label_files.keys())

    # 创建配对列表
    for stem in sorted(common_stems):
        pairs.append((img_files[stem], label_files[stem]))

    return pairs


def get_img_pairs(img_dir):
    """获取匹配的图片-标签对（用于没有标签的数据，如背景）"""
    pairs = []

    # 获取不带扩展名的文件名作为匹配依据
    img_files = {f.stem: f for f in img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']}

    # 找出共同的文件名（不带扩展名）
    common_stems = set(img_files.keys())

    # 创建配对列表
    for stem in sorted(common_stems):
        pairs.append((img_files[stem], None))

    return pairs


def merge_dataset(src_root, dst_root, img_dir="rgb", label_dir="label", if_background=False):
    """
    改进版：严格匹配图片和标签文件

    Args:
        src_root: 源数据根目录
        dst_root: 目标根目录
        img_dir: 目标RGB图片目录名
        label_dir: 目标标签目录名
        if_background: 是否是背景这类没有标签的数据
    """

    # 创建目标目录
    dst_img = Path(dst_root) / img_dir
    dst_label = Path(dst_root) / label_dir
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_label.mkdir(parents=True, exist_ok=True)

    # 用于保存classes.txt文件内容
    classes_content = None
    counter = 1

    # 遍历所有子文件夹
    for item in Path(src_root).iterdir():
        if item.is_dir():
            print(f"处理文件夹: {item.name}")

            # 检查子目录
            src_img = item / img_dir
            src_label = item / label_dir

            if if_background:
                if not src_img.exists():
                    print(f"警告: {item.name} 缺少img目录，已跳过")
                    continue
            else:
                if not src_img.exists() or not src_label.exists():
                    print(f"警告: {item.name} 缺少img或label目录，已跳过")
                    continue

            if not if_background:
                # 获取严格匹配的文件对
                matched_pairs = get_matching_pairs(src_img, src_label)

                if not matched_pairs:
                    print(f"警告: {item.name} 中没有找到匹配的图片-标签对")
                    continue
            else:
                matched_pairs = get_img_pairs(src_img)

            # 检查classes.txt
            classes_file = src_label / "classes.txt"
            if classes_file.exists():
                with open(classes_file, 'r', encoding='utf-8') as f:
                    current_classes = f.read()
                if classes_content is None:
                    classes_content = current_classes

            # 处理每对匹配文件（若为背景图像，则可以支持没有标签）
            for img_file, label_file in matched_pairs:
                new_name = f"{counter:06d}"  # 6位数字编号

                # 复制图片文件
                new_img_path = dst_img / f"{new_name}{img_file.suffix}"
                shutil.copy2(img_file, new_img_path)

                if label_file:
                    # 复制标签文件
                    new_label_path = dst_label / f"{new_name}{label_file.suffix}"
                    shutil.copy2(label_file, new_label_path)
                    print(f"匹配复制: {img_file.name} ↔ {label_file.name} -> {new_name}")
                else:
                    print(f"复制: {img_file.name} -> {new_name}")
                counter += 1

    # 保存classes.txt
    if classes_content is not None:
        classes_dst = dst_label / "classes.txt"
        with open(classes_dst, 'w', encoding='utf-8') as f:
            f.write(classes_content)
        print(f"\n已保存classes.txt到 {classes_dst}")

    print(f"\n处理完成！共处理 {counter - 1} 对严格匹配的图片-标签文件")
    print(f"img文件保存在: {dst_img}")
    print(f"标签文件保存在: {dst_label}")


def main():
    parser = argparse.ArgumentParser(description='合并数据集文件夹')
    parser.add_argument('--src', type=str, default='/home/csz_xishu2/data/zbzx_data/version_2/fine_visible_train_data/20260305_jy_ftv_640',
                        help='源数据根目录路径')
    parser.add_argument('--dst', type=str, default='/home/csz_xishu2/data/zbzx_data/version_2/fine_visible_train_data/20260305_jy_ftv_640_merge',
                        help='目标根目录路径')
    parser.add_argument('--img', type=str, default='images', help='目标img目录名')
    parser.add_argument('--label', type=str, default='labels', help='目标标签目录名')
    parser.add_argument('--if_background', type=str, default=False, help='是否是背景这类没有标签的数据')

    args = parser.parse_args()

    # 验证源目录是否存在
    if not Path(args.src).exists():
        print(f"错误: 源目录 {args.src} 不存在")
        return

    merge_dataset(args.src, args.dst, args.img, args.label, if_background=args.if_background)


if __name__ == "__main__":
    # 直接运行示例
    # merge_dataset("./", "./merged_dataset")

    # 使用命令行参数运行
    main()
