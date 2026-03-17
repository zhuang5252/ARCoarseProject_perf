# -*- coding: utf-8 -*-
# @Author    : 
# @File      : img_yolo_label_pad_or_crop.py
# @Created   : 2026/3/13 下午5:43
# @Desc      : 图像尺寸统一处理工具 - 支持随机裁剪和随机填充

# !/usr/bin/env python3
"""
图像尺寸统一处理工具 - 支持随机裁剪和随机填充
将不同尺寸的图像统一为目标尺寸，并更新对应的YOLO格式标签
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import argparse
import warnings
import shutil
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ImageSizeProcessor:
    def __init__(self, target_width, target_height, mode='auto',
                 output_root=None, random_seed=42):
        """
        初始化处理器

        Args:
            target_width: 目标图像宽度
            target_height: 目标图像高度
            mode: 'crop', 'pad', 或 'auto'
                  - crop: 强制裁剪
                  - pad: 强制填充
                  - auto: 自动选择，根据尺寸关系自动处理
            output_root: 输出根目录
            random_seed: 随机种子，用于保证可重复性
        """
        self.target_width = target_width
        self.target_height = target_height
        self.mode = mode.lower()
        self.output_root = output_root
        self.random_seed = random_seed
        random.seed(random_seed)

        if mode not in ['crop', 'pad', 'auto']:
            raise ValueError("Mode must be 'crop', 'pad', or 'auto'")

        # 用于存储详细的统计信息
        self.detailed_stats = {
            'crop': defaultdict(int),  # 仅需要裁剪的图像
            'pad': defaultdict(int),  # 仅需要填充的图像
            'both': defaultdict(int),  # 既需要裁剪又需要填充的图像
            'none': defaultdict(int),  # 完全匹配的图像
            'errors': defaultdict(int),  # 处理错误的图像
            'skipped': []  # 跳过的文件夹列表
        }
        self.total_stats = {'crop': 0, 'pad': 0, 'both': 0, 'none': 0, 'errors': 0, 'skipped': 0}

    def process_dataset(self, input_root, output_root=None):
        """处理整个数据集"""
        input_root = Path(input_root)
        if output_root is None:
            output_root = input_root.parent / f"{input_root.name}_resized"
        else:
            output_root = Path(output_root)

        print(f"\n{'=' * 70}")
        print(f"🚀 开始处理数据集")
        print(f"{'=' * 70}")
        print(f"输入目录: {input_root}")
        print(f"输出目录: {output_root}")
        print(f"目标尺寸: {self.target_width} x {self.target_height}")
        print(f"处理模式: {self.mode}")
        print(f"{'=' * 70}\n")

        # 创建输出目录结构
        self._create_output_structure(input_root, output_root)

        # 复制所有classes.txt文件
        self._copy_classes_files(input_root, output_root)

        # 获取所有子目录并排序
        all_subdirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

        if not all_subdirs:
            print("⚠️  警告: 输入目录中没有找到子文件夹")
            return

        print(f"找到 {len(all_subdirs)} 个待处理的文件夹\n")

        # 处理每个子目录
        for sub_dir in all_subdirs:
            folder_name = sub_dir.name
            images_dir = sub_dir / 'images'
            labels_dir = sub_dir / 'labels'

            # 检查目录结构
            if not images_dir.exists() or not labels_dir.exists():
                print(f"⏭️  跳过文件夹 [{folder_name}]: 缺少images或labels目录")
                self.detailed_stats['skipped'].append(folder_name)
                self.total_stats['skipped'] += 1
                continue

            print(f"\n{'=' * 60}")
            print(f"📁 处理文件夹: {folder_name}")
            print(f"{'=' * 60}")

            # 处理该目录下的所有图像
            dir_stats = self._process_directory(images_dir, labels_dir,
                                                output_root / folder_name, folder_name)

            # 更新总统计
            for k, v in dir_stats.items():
                self.total_stats[k] += v

        # 打印详细的统计信息
        self._print_detailed_stats()

    def _create_output_structure(self, input_root, output_root):
        """创建与输入相同的目录结构"""
        for sub_dir in input_root.iterdir():
            if sub_dir.is_dir():
                (output_root / sub_dir.name / 'images').mkdir(parents=True, exist_ok=True)
                (output_root / sub_dir.name / 'labels').mkdir(parents=True, exist_ok=True)

    def _copy_classes_files(self, input_root, output_root):
        """
        复制所有子目录下的classes.txt文件到输出目录的对应位置
        """
        print("📋 正在复制classes.txt文件...")
        copied_count = 0

        for sub_dir in input_root.iterdir():
            if not sub_dir.is_dir():
                continue

            # 检查labels目录下的classes.txt
            src_classes = sub_dir / 'labels' / 'classes.txt'
            if src_classes.exists():
                dst_classes = output_root / sub_dir.name / 'labels' / 'classes.txt'
                shutil.copy2(src_classes, dst_classes)
                copied_count += 1
                print(f"   ✅ 已复制: {sub_dir.name}/labels/classes.txt")

        if copied_count == 0:
            print("   ⚠️  未找到任何classes.txt文件")
        else:
            print(f"   ✅ 共复制 {copied_count} 个classes.txt文件\n")

    def _process_directory(self, images_dir, labels_dir, output_dir, folder_name):
        """处理单个目录下的所有图像"""
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + \
                      list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.bmp')) + \
                      list(images_dir.glob('*.JPG')) + list(images_dir.glob('*.PNG'))

        if not image_files:
            print(f"⚠️  警告: 文件夹 [{folder_name}] 中没有图像文件")
            return {'crop': 0, 'pad': 0, 'both': 0, 'none': 0, 'errors': 0}

        print(f"发现 {len(image_files)} 个图像文件")

        stats = {'crop': 0, 'pad': 0, 'both': 0, 'none': 0, 'errors': 0}

        # 用于记录该文件夹下每种处理方式的文件名
        folder_crop_files = []
        folder_pad_files = []
        folder_both_files = []
        folder_none_files = []
        folder_error_files = []

        for img_path in tqdm(image_files, desc=f"处理中"):
            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"\n❌ 警告: 无法读取图像 {img_path.name}")
                    stats['errors'] += 1
                    self.detailed_stats['errors'][folder_name] += 1
                    folder_error_files.append(img_path.name)
                    continue

                orig_h, orig_w = img.shape[:2]

                # 读取对应的标签文件
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    print(f"\n❌ 警告: 标签文件不存在 {label_path.name}")
                    stats['errors'] += 1
                    self.detailed_stats['errors'][folder_name] += 1
                    folder_error_files.append(img_path.name)
                    continue

                with open(label_path, 'r') as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]

                # 解析标签
                targets = self._parse_labels(labels, orig_w, orig_h)

                # 处理图像和标签
                processed_img, processed_labels = self._process_image_and_labels(
                    img, targets, orig_w, orig_h
                )

                # 统计处理方式
                need_pad = orig_w < self.target_width or orig_h < self.target_height
                need_crop = orig_w > self.target_width or orig_h > self.target_height

                if need_pad and need_crop:
                    stats['both'] += 1
                    self.detailed_stats['both'][folder_name] += 1
                    folder_both_files.append(img_path.name)
                elif need_crop:
                    stats['crop'] += 1
                    self.detailed_stats['crop'][folder_name] += 1
                    folder_crop_files.append(img_path.name)
                elif need_pad:
                    stats['pad'] += 1
                    self.detailed_stats['pad'][folder_name] += 1
                    folder_pad_files.append(img_path.name)
                else:
                    stats['none'] += 1
                    self.detailed_stats['none'][folder_name] += 1
                    folder_none_files.append(img_path.name)

                # 保存处理后的图像
                output_img_path = output_dir / 'images' / img_path.name
                cv2.imwrite(str(output_img_path), processed_img)

                # 保存处理后的标签
                if processed_labels:
                    output_label_path = output_dir / 'labels' / f"{img_path.stem}.txt"
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(processed_labels))

            except Exception as e:
                print(f"\n❌ 处理文件 {img_path.name} 时出错: {e}")
                stats['errors'] += 1
                self.detailed_stats['errors'][folder_name] += 1
                folder_error_files.append(img_path.name)
                continue

        # 打印当前文件夹的统计信息
        self._print_folder_stats(folder_name, stats,
                                 folder_crop_files, folder_pad_files, folder_both_files,
                                 folder_none_files, folder_error_files)

        return stats

    def _parse_labels(self, labels, orig_w, orig_h):
        """解析标签，转换为绝对坐标"""
        targets = []
        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                # 转换为绝对坐标
                x1 = (x_center - width / 2) * orig_w
                y1 = (y_center - height / 2) * orig_h
                x2 = (x_center + width / 2) * orig_w
                y2 = (y_center + height / 2) * orig_h
                targets.append((int(class_id), x1, y1, x2, y2))
        return targets

    def _process_image_and_labels(self, img, targets, orig_w, orig_h):
        """
        统一处理图像和标签（先填充后裁剪）

        Args:
            img: 原始图像
            targets: 目标列表 [(class_id, x1, y1, x2, y2), ...]
            orig_w, orig_h: 原始图像尺寸

        Returns:
            处理后的图像和更新后的标签
        """
        current_img = img.copy()
        current_targets = targets.copy()
        current_w, current_h = orig_w, orig_h

        # 第一步：如果需要填充，先进行随机填充
        if current_w < self.target_width or current_h < self.target_height:
            current_img, current_targets, current_w, current_h = self._random_pad(
                current_img, current_targets, current_w, current_h
            )

        # 第二步：如果需要裁剪，再进行随机裁剪
        # 注意：这里使用填充后的实际尺寸 current_w, current_h 来判断是否需要裁剪
        if current_w > self.target_width or current_h > self.target_height:
            current_img, current_targets = self._random_crop(
                current_img, current_targets, current_w, current_h
            )
            # 裁剪后尺寸变为目标尺寸
            current_w, current_h = self.target_width, self.target_height

        # 转换为YOLO格式标签
        new_labels = self._targets_to_yolo_labels(current_targets, self.target_width, self.target_height)

        return current_img, new_labels

    def _random_pad(self, img, targets, current_w, current_h):
        """
        随机填充图像到目标尺寸

        Args:
            img: 当前图像
            targets: 目标列表
            current_w, current_h: 当前图像尺寸

        Returns:
            填充后的图像、更新后的目标、新尺寸
        """
        # 计算需要填充的总量
        pad_total_w = max(0, self.target_width - current_w)
        pad_total_h = max(0, self.target_height - current_h)

        if pad_total_w == 0 and pad_total_h == 0:
            return img, targets, current_w, current_h

        # 随机分配上下左右的填充量
        pad_left = random.randint(0, pad_total_w) if pad_total_w > 0 else 0
        pad_right = pad_total_w - pad_left

        pad_top = random.randint(0, pad_total_h) if pad_total_h > 0 else 0
        pad_bottom = pad_total_h - pad_top

        # 执行填充
        padded_img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # 更新目标坐标
        updated_targets = []
        for class_id, x1, y1, x2, y2 in targets:
            new_x1 = x1 + pad_left
            new_y1 = y1 + pad_top
            new_x2 = x2 + pad_left
            new_y2 = y2 + pad_top
            updated_targets.append((class_id, new_x1, new_y1, new_x2, new_y2))

        # 返回填充后的实际尺寸（可能等于目标尺寸，也可能小于目标尺寸如果只填充了一维）
        new_w = current_w + pad_total_w
        new_h = current_h + pad_total_h
        return padded_img, updated_targets, new_w, new_h

    def _random_crop(self, img, targets, current_w, current_h):
        """
        随机裁剪图像到目标尺寸

        Args:
            img: 当前图像
            targets: 目标列表
            current_w, current_h: 当前图像尺寸（可能大于目标尺寸）

        Returns:
            裁剪后的图像和更新后的目标
        """
        # 计算需要裁剪的量
        crop_total_w = current_w - self.target_width
        crop_total_h = current_h - self.target_height

        # 随机分配上下左右的裁剪量
        # 注意：裁剪是从左上角开始的，所以只需要确定裁剪起点
        max_crop_x = crop_total_w
        max_crop_y = crop_total_h

        # 如果有目标，需要确保目标在裁剪区域内
        if targets:
            # 计算所有目标的边界框
            all_x1 = min(t[1] for t in targets)
            all_y1 = min(t[2] for t in targets)
            all_x2 = max(t[3] for t in targets)
            all_y2 = max(t[4] for t in targets)

            # 目标区域的宽度和高度
            target_w = all_x2 - all_x1
            target_h = all_y2 - all_y1

            # 检查目标区域是否超过裁剪尺寸
            if target_w > self.target_width or target_h > self.target_height:
                print(f"\n⚠️  警告: 目标区域尺寸 ({target_w:.1f}x{target_h:.1f}) "
                      f"超过裁剪尺寸 ({self.target_width}x{self.target_height})")
                # 使用中心裁剪
                crop_x = max(0, crop_total_w // 2)
                crop_y = max(0, crop_total_h // 2)
            else:
                # 计算裁剪起始点的可移动范围，确保所有目标都在裁剪区域内
                min_crop_x = max(0, all_x2 - self.target_width)
                max_crop_x = min(all_x1, crop_total_w)
                min_crop_y = max(0, all_y2 - self.target_height)
                max_crop_y = min(all_y1, crop_total_h)

                # 如果可移动范围有效，随机选择裁剪点
                if min_crop_x <= max_crop_x and min_crop_y <= max_crop_y:
                    crop_x = random.randint(int(min_crop_x), int(max_crop_x)) if min_crop_x < max_crop_x else int(
                        min_crop_x)
                    crop_y = random.randint(int(min_crop_y), int(max_crop_y)) if min_crop_y < max_crop_y else int(
                        min_crop_y)
                else:
                    # 如果无法完整包含所有目标，使用中心对齐
                    crop_x = max(0, min(all_x1, crop_total_w))
                    crop_y = max(0, min(all_y1, crop_total_h))
        else:
            # 没有目标，完全随机裁剪
            crop_x = random.randint(0, max_crop_x) if max_crop_x > 0 else 0
            crop_y = random.randint(0, max_crop_y) if max_crop_y > 0 else 0

        # 确保裁剪点有效
        crop_x = min(max(0, crop_x), max_crop_x)
        crop_y = min(max(0, crop_y), max_crop_y)

        # 执行裁剪
        cropped_img = img[crop_y:crop_y + self.target_height,
                      crop_x:crop_x + self.target_width]

        # 更新目标坐标
        updated_targets = []
        for class_id, x1, y1, x2, y2 in targets:
            new_x1 = x1 - crop_x
            new_y1 = y1 - crop_y
            new_x2 = x2 - crop_x
            new_y2 = y2 - crop_y

            # 检查目标是否在图像内
            if new_x1 < self.target_width and new_y1 < self.target_height and \
                    new_x2 > 0 and new_y2 > 0 and new_x2 > new_x1 and new_y2 > new_y1:
                # 裁剪到图像边界内
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(self.target_width, new_x2)
                new_y2 = min(self.target_height, new_y2)

                if new_x2 > new_x1 and new_y2 > new_y1:
                    updated_targets.append((class_id, new_x1, new_y1, new_x2, new_y2))

        return cropped_img, updated_targets

    def _targets_to_yolo_labels(self, targets, img_w, img_h):
        """将目标转换为YOLO格式标签"""
        new_labels = []
        for class_id, x1, y1, x2, y2 in targets:
            # 转换为YOLO格式
            new_w = (x2 - x1) / img_w
            new_h = (y2 - y1) / img_h
            new_x_center = (x1 + x2) / 2 / img_w
            new_y_center = (y1 + y2) / 2 / img_h

            # 确保值在(0,1]范围内
            new_x_center = max(min(new_x_center, 1.0), 0.0)
            new_y_center = max(min(new_y_center, 1.0), 0.0)
            new_w = max(min(new_w, 1.0), 0.0)
            new_h = max(min(new_h, 1.0), 0.0)

            if new_w > 0 and new_h > 0:
                new_labels.append(f"{int(class_id)} {new_x_center:.6f} {new_y_center:.6f} "
                                  f"{new_w:.6f} {new_h:.6f}")
        return new_labels

    def _print_folder_stats(self, folder_name, stats, crop_files, pad_files, both_files, none_files, error_files):
        """打印单个文件夹的详细统计信息"""
        total = stats['crop'] + stats['pad'] + stats['both'] + stats['none'] + stats['errors']

        print(f"\n📊 文件夹 [{folder_name}] 处理统计:")
        print(f"   ├─ 总文件数: {total} 张")
        print(f"   ├─ 仅裁剪处理: {stats['crop']} 张 ({stats['crop'] / total * 100:.1f}%)")
        if stats['crop'] > 0:
            if len(crop_files) <= 5:
                for f in crop_files:
                    print(f"   │    └─ ✂️ {f}")
            else:
                print(f"   │    └─ ... 共 {stats['crop']} 个文件")

        print(f"   ├─ 仅填充处理: {stats['pad']} 张 ({stats['pad'] / total * 100:.1f}%)")
        if stats['pad'] > 0:
            if len(pad_files) <= 5:
                for f in pad_files:
                    print(f"   │    └─ ⬛ {f}")
            else:
                print(f"   │    └─ ... 共 {stats['pad']} 个文件")

        print(f"   ├─ 裁剪+填充处理: {stats['both']} 张 ({stats['both'] / total * 100:.1f}%)")
        if stats['both'] > 0:
            if len(both_files) <= 5:
                for f in both_files:
                    print(f"   │    └─ 🔄 {f}")
            else:
                print(f"   │    └─ ... 共 {stats['both']} 个文件")

        print(f"   ├─ 无需处理: {stats['none']} 张 ({stats['none'] / total * 100:.1f}%)")
        if stats['none'] > 0:
            if len(none_files) <= 5:
                for f in none_files:
                    print(f"   │    └─ ✅ {f}")
            else:
                print(f"   │    └─ ... 共 {stats['none']} 个文件")

        print(f"   └─ 处理错误: {stats['errors']} 张 ({stats['errors'] / total * 100:.1f}%)")
        if stats['errors'] > 0 and len(error_files) <= 5:
            for f in error_files:
                print(f"        └─ ❌ {f}")
        elif stats['errors'] > 5:
            print(f"        └─ ... 共 {stats['errors']} 个文件")
        print()

    def _print_detailed_stats(self):
        """打印所有文件夹的详细统计信息"""
        print("\n" + "=" * 80)
        print("📊 最终详细统计报告")
        print("=" * 80)

        # 收集所有有数据的文件夹
        all_folders = set()
        for mode_stats in [self.detailed_stats['crop'], self.detailed_stats['pad'],
                           self.detailed_stats['both'], self.detailed_stats['none'],
                           self.detailed_stats['errors']]:
            all_folders.update(mode_stats.keys())

        skipped_folders = self.detailed_stats['skipped']

        if not all_folders and not skipped_folders:
            print("没有处理任何文件夹")
            return

        # 打印表头
        print(
            f"\n{'文件夹名称':<20} {'仅裁剪':>8} {'仅填充':>8} {'裁剪+填充':>10} {'无需处理':>8} {'错误':>6} {'总计':>8}")
        print("-" * 84)

        # 打印每个文件夹的统计
        grand_total = {'crop': 0, 'pad': 0, 'both': 0, 'none': 0, 'errors': 0}
        crop_folders = []
        pad_folders = []
        both_folders = []
        none_folders = []
        error_folders = []

        for folder in sorted(all_folders):
            crop_cnt = self.detailed_stats['crop'].get(folder, 0)
            pad_cnt = self.detailed_stats['pad'].get(folder, 0)
            both_cnt = self.detailed_stats['both'].get(folder, 0)
            none_cnt = self.detailed_stats['none'].get(folder, 0)
            error_cnt = self.detailed_stats['errors'].get(folder, 0)
            total = crop_cnt + pad_cnt + both_cnt + none_cnt + error_cnt

            grand_total['crop'] += crop_cnt
            grand_total['pad'] += pad_cnt
            grand_total['both'] += both_cnt
            grand_total['none'] += none_cnt
            grand_total['errors'] += error_cnt

            if crop_cnt > 0:
                crop_folders.append(f"{folder}({crop_cnt})")
            if pad_cnt > 0:
                pad_folders.append(f"{folder}({pad_cnt})")
            if both_cnt > 0:
                both_folders.append(f"{folder}({both_cnt})")
            if none_cnt > 0:
                none_folders.append(f"{folder}({none_cnt})")
            if error_cnt > 0:
                error_folders.append(f"{folder}({error_cnt})")

            # 根据主要处理方式添加标记
            if both_cnt > 0:
                marker = "🔄"
            elif crop_cnt > 0 and pad_cnt == 0:
                marker = "✂️"
            elif pad_cnt > 0 and crop_cnt == 0:
                marker = "⬛"
            elif none_cnt > 0:
                marker = "✅"
            else:
                marker = "📁"

            print(
                f"{marker} {folder:<18} {crop_cnt:>8} {pad_cnt:>8} {both_cnt:>10} {none_cnt:>8} {error_cnt:>6} {total:>8}")

        # 打印跳过的文件夹
        if skipped_folders:
            print("-" * 84)
            for folder in sorted(skipped_folders):
                print(f"⏭️  {folder:<18} {'-':>8} {'-':>8} {'-':>10} {'-':>8} {'-':>6} {'-':>8}")

        print("-" * 84)

        total_processed = grand_total['crop'] + grand_total['pad'] + grand_total['both'] + grand_total['none'] + \
                          grand_total['errors']

        print(f"{'总计':<20} {grand_total['crop']:>8} {grand_total['pad']:>8} "
              f"{grand_total['both']:>10} {grand_total['none']:>8} {grand_total['errors']:>6} {total_processed:>8}")

        if skipped_folders:
            print(f"\n⏭️  跳过的文件夹数量: {len(skipped_folders)}")

        # 打印分类统计
        print("\n📌 按处理方式分类的文件夹:")
        if crop_folders:
            print(f"   ├─ ✂️ 仅裁剪处理的文件夹: {', '.join(crop_folders)}")
        else:
            print(f"   ├─ ✂️ 仅裁剪处理的文件夹: 无")

        if pad_folders:
            print(f"   ├─ ⬛ 仅填充处理的文件夹: {', '.join(pad_folders)}")
        else:
            print(f"   ├─ ⬛ 仅填充处理的文件夹: 无")

        if both_folders:
            print(f"   ├─ 🔄 裁剪+填充处理的文件夹: {', '.join(both_folders)}")
        else:
            print(f"   ├─ 🔄 裁剪+填充处理的文件夹: 无")

        if none_folders:
            print(f"   ├─ ✅ 无需处理的文件夹: {', '.join(none_folders)}")
        else:
            print(f"   ├─ ✅ 无需处理的文件夹: 无")

        if error_folders:
            print(f"   ├─ ❌ 有错误的文件夹: {', '.join(error_folders)}")
        else:
            print(f"   ├─ ❌ 有错误的文件夹: 无")

        if skipped_folders:
            print(f"   └─ ⏭️ 跳过的文件夹: {', '.join(sorted(skipped_folders))}")

        # 打印总体统计
        print("\n📈 总体统计:")
        print(f"   ├─ 总文件夹数: {len(all_folders) + len(skipped_folders)}")
        print(f"   ├─ 成功处理文件夹: {len(all_folders)}")
        print(f"   ├─ 跳过文件夹: {len(skipped_folders)}")
        print(f"   ├─ 总处理图像: {total_processed} 张")
        if total_processed > 0:
            print(f"   ├─ 仅裁剪处理: {grand_total['crop']} 张 ({grand_total['crop'] / total_processed * 100:.1f}%)")
            print(f"   ├─ 仅填充处理: {grand_total['pad']} 张 ({grand_total['pad'] / total_processed * 100:.1f}%)")
            print(f"   ├─ 裁剪+填充处理: {grand_total['both']} 张 ({grand_total['both'] / total_processed * 100:.1f}%)")
            print(f"   ├─ 无需处理: {grand_total['none']} 张 ({grand_total['none'] / total_processed * 100:.1f}%)")
            print(f"   └─ 处理错误: {grand_total['errors']} 张 ({grand_total['errors'] / total_processed * 100:.1f}%)")

        print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description='统一图像尺寸并更新YOLO标签')
    parser.add_argument('--input_dir', type=str,
                        default='/home/csz_xishu2/data/zbzx_data/version_2/fine_visible_orig_data/20260305_jy_ftv',
                        help='输入数据集根目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/csz_xishu2/data/zbzx_data/version_2/fine_visible_orig_data/20260305_jy_ftv_640',
                        help='输出数据集根目录（默认在原目录名后加_resized）')
    parser.add_argument('--target_width', type=int, default=640,
                        help='目标图像宽度')
    parser.add_argument('--target_height', type=int, default=512,
                        help='目标图像高度')
    parser.add_argument('--mode', type=str, choices=['crop', 'pad', 'auto'],
                        default='auto', help='处理模式：crop（裁剪）、pad（填充）或auto（自动）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 创建处理器并处理数据集
    processor = ImageSizeProcessor(
        target_width=args.target_width,
        target_height=args.target_height,
        mode=args.mode,
        output_root=args.output_dir,
        random_seed=args.seed
    )

    processor.process_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
