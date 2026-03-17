# -*- coding: utf-8 -*-
# @Author    : 
# @File      : process_folders_AR2511_1.py
# @Created   : 2025/11/26 上午11:02
# @Desc      : 将AR2511数据集文件夹处理为Aot/AR_csv格式

"""
将AR2511数据集文件夹处理为Aot/AR_csv格式
"""

import shutil
import pandas as pd
from pathlib import Path


class DataStructureConverter:
    def __init__(self, data_root_dir, label_root_dir, output_root_dir):
        """
        初始化数据转换器

        Args:
            data_root_dir: 原始数据根目录
            label_root_dir: 原始标签根目录
            output_root_dir: 输出根目录
        """
        self.data_root_dir = Path(data_root_dir)
        self.label_root_dir = Path(label_root_dir)
        self.output_root_dir = Path(output_root_dir)

        # 创建输出目录
        self.output_root_dir.mkdir(parents=True, exist_ok=True)

    def copy_image_files(self, source_dir, target_dir):
        """
        复制图像文件

        Args:
            source_dir: 源目录
            target_dir: 目标目录
        """
        try:
            # 确保目标目录存在
            target_dir.mkdir(parents=True, exist_ok=True)

            # 查找所有图像文件
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            image_files = []

            for ext in image_extensions:
                image_files.extend(source_dir.glob(ext))

            # 复制文件
            copied_count = 0
            for img_file in image_files:
                if img_file.is_file():
                    target_file = target_dir / img_file.name
                    shutil.copy2(img_file, target_file)
                    copied_count += 1

            return copied_count

        except Exception as e:
            print(f"复制图像文件错误: {e}")
            return 0

    def copy_csv_file(self, source_file, target_file):
        """
        复制CSV文件

        Args:
            source_file: 源文件
            target_file: 目标文件
        """
        try:
            # 确保目标目录存在
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(source_file, target_file)
            return True

        except Exception as e:
            print(f"复制CSV文件错误: {e}")
            return False

    def validate_data_structure(self):
        """
        验证原始数据结构是否正确
        """
        print("验证数据结构...")

        # 检查数据目录
        if not self.data_root_dir.exists():
            print(f"错误: 数据目录不存在: {self.data_root_dir}")
            return False

        # 检查标签目录
        if not self.label_root_dir.exists():
            print(f"错误: 标签目录不存在: {self.label_root_dir}")
            return False

        # 查找所有场景
        data_scenes = [d.name for d in self.data_root_dir.iterdir() if d.is_dir()]
        label_scenes = [d.name for d in self.label_root_dir.iterdir() if d.is_dir()]

        print(f"数据目录中的场景: {data_scenes}")
        print(f"标签目录中的场景: {label_scenes}")

        # 检查场景一致性
        common_scenes = set(data_scenes) & set(label_scenes)
        if not common_scenes:
            print("警告: 数据目录和标签目录中没有共同的场景")
            return False

        print(f"共同场景: {common_scenes}")
        return True

    def get_all_channels(self):
        """
        获取所有需要处理的通道信息
        """
        channels_info = []

        # 遍历标签目录结构
        for scene_dir in self.label_root_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            scene_name = scene_dir.name

            # 检查对应的数据目录是否存在
            data_scene_dir = self.data_root_dir / scene_name
            if not data_scene_dir.exists():
                print(f"警告: 数据目录中找不到场景 {scene_name}，跳过")
                continue

            # 遍历通道
            for channel_dir in scene_dir.iterdir():
                if not channel_dir.is_dir():
                    continue

                channel_name = channel_dir.name

                # 检查对应的数据通道目录是否存在
                data_channel_dir = data_scene_dir / channel_name
                if not data_channel_dir.exists():
                    print(f"警告: 数据目录中找不到通道 {scene_name}/{channel_name}，跳过")
                    continue

                # 检查CSV文件是否存在
                csv_file = channel_dir / "groundtruth.csv"
                if not csv_file.exists():
                    print(f"警告: 标签文件不存在 {csv_file}，跳过")
                    continue

                channels_info.append({
                    'scene_name': scene_name,
                    'channel_name': channel_name,
                    'data_dir': data_channel_dir,
                    'csv_file': csv_file
                })

        return channels_info

    def convert_structure(self):
        """
        执行数据结构转换
        """
        print("开始数据结构转换...")

        # 验证数据结构
        if not self.validate_data_structure():
            print("数据结构验证失败，请检查目录结构")
            return False

        # 获取所有通道信息
        channels_info = self.get_all_channels()

        if not channels_info:
            print("没有找到可处理的通道")
            return False

        print(f"找到 {len(channels_info)} 个需要处理的通道")

        total_success = 0
        total_images = 0

        for channel_info in channels_info:
            scene_name = channel_info['scene_name']
            channel_name = channel_info['channel_name']
            data_dir = channel_info['data_dir']
            csv_file = channel_info['csv_file']

            print(f"\n处理场景: {scene_name}, 通道: {channel_name}")

            # 创建目标目录结构
            target_dir_name = f"{scene_name}_{channel_name}"
            target_dir = self.output_root_dir / target_dir_name

            # 创建Images目录
            images_target_dir = target_dir / "Images" / channel_name
            images_target_dir.mkdir(parents=True, exist_ok=True)

            # 创建ImageSets目录
            imagesets_target_dir = target_dir / "ImageSets"
            imagesets_target_dir.mkdir(parents=True, exist_ok=True)

            # 复制图像文件
            print(f"  复制图像文件从: {data_dir}")
            image_count = self.copy_image_files(data_dir, images_target_dir)
            total_images += image_count
            print(f"  复制了 {image_count} 个图像文件")

            # 复制CSV文件
            csv_target_file = imagesets_target_dir / "groundtruth.csv"
            print(f"  复制标签文件从: {csv_file}")
            csv_success = self.copy_csv_file(csv_file, csv_target_file)

            if csv_success and image_count > 0:
                total_success += 1
                print(f"  ✓ 成功处理: {target_dir_name}")
            else:
                print(f"  ✗ 处理失败: {target_dir_name}")

        # 生成转换报告
        self.generate_conversion_report(channels_info, total_success, total_images)

        print(f"\n转换完成! 成功处理 {total_success}/{len(channels_info)} 个通道")
        return True

    def generate_conversion_report(self, channels_info, total_success, total_images):
        """
        生成转换报告
        """
        report_file = self.output_root_dir / "conversion_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 数据结构转换报告 ===\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

            f.write(f"1. 转换统计\n")
            f.write(f"   总通道数: {len(channels_info)}\n")
            f.write(f"   成功处理: {total_success}\n")
            f.write(f"   失败数量: {len(channels_info) - total_success}\n")
            f.write(f"   总图像数: {total_images}\n\n")

            f.write(f"2. 原始目录结构\n")
            f.write(f"   数据根目录: {self.data_root_dir}\n")
            f.write(f"   标签根目录: {self.label_root_dir}\n")
            f.write(f"   输出目录: {self.output_root_dir}\n\n")

            f.write(f"3. 处理的通道详情\n")
            f.write(f"{'场景':<15} {'通道':<10} {'状态':<10} {'图像数量':<10}\n")
            f.write("-" * 50 + "\n")

            for channel_info in channels_info:
                scene_name = channel_info['scene_name']
                channel_name = channel_info['channel_name']
                data_dir = channel_info['data_dir']

                # 统计图像数量
                image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
                image_count = 0
                for ext in image_extensions:
                    image_count += len(list(data_dir.glob(ext)))

                # 检查是否成功处理
                target_dir_name = f"{scene_name}_{channel_name}"
                target_dir = self.output_root_dir / target_dir_name
                success = target_dir.exists() and (target_dir / "Images").exists()

                status = "成功" if success else "失败"

                f.write(f"{scene_name:<15} {channel_name:<10} {status:<10} {image_count:<10}\n")

            f.write(f"\n4. 新目录结构示例\n")
            if channels_info:
                example = channels_info[0]
                scene_name = example['scene_name']
                channel_name = example['channel_name']
                f.write(f"   {scene_name}_{channel_name}/\n")
                f.write(f"   ├── Images/\n")
                f.write(f"   │   └── {channel_name}/\n")
                f.write(f"   │       ├── image1.png\n")
                f.write(f"   │       └── image2.png\n")
                f.write(f"   └── ImageSets/\n")
                f.write(f"       └── {channel_name}/\n")
                f.write(f"           └── groundtruth.csv\n")

        print(f"转换报告已生成: {report_file}")

    def verify_conversion(self):
        """
        验证转换结果
        """
        print("\n验证转换结果...")

        target_dirs = [d for d in self.output_root_dir.iterdir() if d.is_dir()]

        if not target_dirs:
            print("错误: 输出目录中没有找到任何转换后的数据")
            return False

        verified_count = 0

        for target_dir in target_dirs:
            scene_channel = target_dir.name

            # 检查目录结构
            images_dir = target_dir / "Images"
            imagesets_dir = target_dir / "ImageSets"

            if not images_dir.exists():
                print(f"警告: {scene_channel} 缺少 Images 目录")
                continue

            if not imagesets_dir.exists():
                print(f"警告: {scene_channel} 缺少 ImageSets 目录")
                continue

            # 检查是否有图像文件
            image_files = list(images_dir.rglob("*.*"))
            if not image_files:
                print(f"警告: {scene_channel} 中没有图像文件")
                continue

            # 检查CSV文件
            csv_files = list(imagesets_dir.rglob("groundtruth.csv"))
            if not csv_files:
                print(f"警告: {scene_channel} 中没有groundtruth.csv文件")
                continue

            verified_count += 1
            print(f"  ✓ 验证通过: {scene_channel} (图像: {len(image_files)}, CSV: {len(csv_files)})")

        print(f"验证完成: {verified_count}/{len(target_dirs)} 个目录通过验证")
        return verified_count > 0


if __name__ == "__main__":
    # 使用实际路径运行
    data_root_dir = "/home/csz_changsha/data/AR_2511/data_filter"  # 相对路径或绝对路径
    label_root_dir = "/home/csz_changsha/data/AR_2511/labels"  # 相对路径或绝对路径
    output_root_dir = "/home/csz_changsha/data/AR_2511/converted_data"  # 输出目录

    # 创建转换器
    converter = DataStructureConverter(data_root_dir, label_root_dir, output_root_dir)

    # 执行转换
    success = converter.convert_structure()

    if success:
        converter.verify_conversion()
        print(f"\n转换完成! 结果保存在: {output_root_dir}")
    else:
        print("\n转换失败!")
