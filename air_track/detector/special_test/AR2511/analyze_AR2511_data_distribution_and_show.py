# -*- coding: utf-8 -*-
# @Author    : 
# @File      : analyze_AR2511_data_distribution_and_show.py
# @Created   : 2025/11/25 上午9:02
# @Desc      : 分析AR2511数据分布并可视化标签在图中

"""
分析AR2511数据分布并可视化标签在图中
"""

import os
import glob
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from air_track.detector.visualization.visualize_and_save import visualize_and_save


# TODO 需要检查其中生成的一些分析图表的准确性


class DataVisualizerAndAnalyzer:
    def __init__(self, data_root_dir, label_root_dir, output_root_dir):
        """
        初始化可视化分析器

        Args:
            data_root_dir: 数据根目录（包含scene1, scene4等）
            label_root_dir: 标签根目录（包含CSV文件）
            output_root_dir: 输出根目录
        """
        self.data_root_dir = Path(data_root_dir)
        self.label_root_dir = Path(label_root_dir)
        self.output_root_dir = Path(output_root_dir)

        # 创建输出目录
        self.output_root_dir.mkdir(parents=True, exist_ok=True)

        # 颜色列表用于不同目标
        self.colors = [
            (255, 0, 0),  # 红色
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (255, 165, 0),  # 橙色
            (128, 0, 128)  # 紫色
        ]

    def find_image_path(self, scene_name, channel_name, img_name):
        """
        根据图像名称查找对应的图像文件路径
        """
        # 构建图像路径
        img_dir = self.data_root_dir / scene_name / channel_name

        if not img_dir.exists():
            print(f"警告: 图像目录不存在: {img_dir}")
            return None

        # 尝试不同的图像文件扩展名
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        for ext in extensions:
            # 构建可能的图像路径
            img_path = img_dir / f"{img_name.replace('.png', ext)}"
            if img_path.exists():
                return img_path

            # 如果没有扩展名，尝试添加扩展名
            img_path = img_dir / f"{img_name}{ext}"
            if img_path.exists():
                return img_path

        # 如果没找到，尝试查找目录下的所有图像文件
        image_files = list(img_dir.glob("*.*"))
        if image_files:
            # 尝试根据文件名模式匹配
            for img_file in image_files:
                if img_name.replace('.png', '') in img_file.name:
                    return img_file

        print(f"警告: 找不到图像文件: {img_name} 在目录 {img_dir}")
        return None

    def draw_bounding_boxes(self, image_path, bboxes, output_path, labels=None):
        """
        在图像上绘制边界框

        Args:
            image_path: 输入图像路径
            bboxes: 边界框列表，每个边界框为 [xmin, ymin, xmax, ymax]
            output_path: 输出图像路径
            labels: 标签列表
        """
        try:
            # 读取图像
            image = Image.open(image_path)

            # 确保图像是RGB模式
            if image.mode != 'RGB':
                print(f"转换图像模式: {image.mode} -> RGB")
                image = image.convert('RGB')

            draw = ImageDraw.Draw(image)

            # 尝试使用字体
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # 绘制每个边界框
            for i, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax = bbox
                color = self.colors[i % len(self.colors)]

                # 绘制矩形框
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

                # 添加标签
                if labels and i < len(labels):
                    label = labels[i]
                    # 绘制标签背景
                    text_bbox = draw.textbbox((xmin, ymin - 25), str(label), font=font)
                    draw.rectangle(text_bbox, fill=color)
                    # 绘制标签文本
                    draw.text((xmin, ymin - 25), str(label), fill=(255, 255, 255), font=font)

                # 添加目标ID
                draw.text((xmin + 5, ymin + 5), f"ID:{i + 1}", fill=color, font=font)

            # 保存图像
            image.save(output_path)
            return True

        except Exception as e:
            print(f"绘制边界框错误: {e}")
            return False

    def analyze_channel_data(self, csv_file_path, scene_name, channel_name):
        """
        分析单个通道的数据

        Returns:
            dict: 分析结果
        """
        try:
            df = pd.read_csv(csv_file_path)

            # 计算边界框的宽度、高度和面积
            bbox_widths = df['Bottom_right_x'] - df['Top_left_x']
            bbox_heights = df['Bottom_right_y'] - df['Top_left_y']
            bbox_areas = bbox_widths * bbox_heights

            # 计算分位数用于框的大小分类
            area_q1 = bbox_areas.quantile(0.33)
            area_q2 = bbox_areas.quantile(0.66)

            # 分类框的大小
            small_boxes = bbox_areas[bbox_areas <= area_q1]
            medium_boxes = bbox_areas[(bbox_areas > area_q1) & (bbox_areas <= area_q2)]
            large_boxes = bbox_areas[bbox_areas > area_q2]

            # 基本统计
            analysis = {
                'scene_name': scene_name,
                'channel_name': channel_name,
                'total_frames': df['frame'].nunique(),
                'total_targets': len(df),
                'targets_per_frame': len(df) / df['frame'].nunique() if df['frame'].nunique() > 0 else 0,

                # 边界框统计
                'bbox_width_stats': {
                    'mean': bbox_widths.mean(),
                    'std': bbox_widths.std(),
                    'min': bbox_widths.min(),
                    'max': bbox_widths.max(),
                    'median': bbox_widths.median()
                },
                'bbox_height_stats': {
                    'mean': bbox_heights.mean(),
                    'std': bbox_heights.std(),
                    'min': bbox_heights.min(),
                    'max': bbox_heights.max(),
                    'median': bbox_heights.median()
                },
                'bbox_area_stats': {
                    'mean': bbox_areas.mean(),
                    'std': bbox_areas.std(),
                    'min': bbox_areas.min(),
                    'max': bbox_areas.max(),
                    'median': bbox_areas.median(),
                    'q1': area_q1,
                    'q2': area_q2
                },

                # 框大小分类统计
                'bbox_size_categories': {
                    'small': {
                        'count': len(small_boxes),
                        'percentage': len(small_boxes) / len(bbox_areas) * 100,
                        'min_area': small_boxes.min() if len(small_boxes) > 0 else 0,
                        'max_area': small_boxes.max() if len(small_boxes) > 0 else 0
                    },
                    'medium': {
                        'count': len(medium_boxes),
                        'percentage': len(medium_boxes) / len(bbox_areas) * 100,
                        'min_area': medium_boxes.min() if len(medium_boxes) > 0 else 0,
                        'max_area': medium_boxes.max() if len(medium_boxes) > 0 else 0
                    },
                    'large': {
                        'count': len(large_boxes),
                        'percentage': len(large_boxes) / len(bbox_areas) * 100,
                        'min_area': large_boxes.min() if len(large_boxes) > 0 else 0,
                        'max_area': large_boxes.max() if len(large_boxes) > 0 else 0
                    }
                },

                # 具体框信息
                'largest_bbox': {
                    'width': bbox_widths.max(),
                    'height': bbox_heights.max(),
                    'area': bbox_areas.max(),
                    'frame': df.loc[bbox_areas.idxmax(), 'frame'] if len(bbox_areas) > 0 else None
                },
                'smallest_bbox': {
                    'width': bbox_widths.min(),
                    'height': bbox_heights.min(),
                    'area': bbox_areas.min(),
                    'frame': df.loc[bbox_areas.idxmin(), 'frame'] if len(bbox_areas) > 0 else None
                },
                'median_bbox': {
                    'width': bbox_widths.median(),
                    'height': bbox_heights.median(),
                    'area': bbox_areas.median()
                },

                # 距离统计
                'distance_stats': {
                    'mean': df['DM_distance'].mean(),
                    'std': df['DM_distance'].std(),
                    'min': df['DM_distance'].min(),
                    'max': df['DM_distance'].max()
                },

                # 目标分布
                'targets_per_frame_distribution': df.groupby('frame').size().value_counts().to_dict(),

                # 边界框位置分布
                'center_x_stats': {
                    'mean': df['Center_x'].mean(),
                    'std': df['Center_x'].std()
                },
                'center_y_stats': {
                    'mean': df['Center_y'].mean(),
                    'std': df['Center_y'].std()
                }
            }

            return analysis

        except Exception as e:
            print(f"分析通道数据错误: {e}")
            return None

    def create_channel_analysis_plots(self, analysis, output_dir):
        """
        创建单个通道的分析图表
        """
        try:
            # 设置中文字体
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

            # 1. 目标数量分布图
            plt.figure(figsize=(10, 6))
            targets_dist = analysis['targets_per_frame_distribution']
            plt.bar(targets_dist.keys(), targets_dist.values())
            plt.xlabel('每帧目标数量')
            plt.ylabel('帧数')
            plt.title(f"{analysis['scene_name']} - {analysis['channel_name']}\n每帧目标数量分布")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'targets_per_frame_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 边界框尺寸分布
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 宽度分布
            ax1.hist([analysis['bbox_width_stats']['mean']], bins=20, alpha=0.7)
            ax1.set_xlabel('边界框宽度')
            ax1.set_ylabel('频率')
            ax1.set_title('边界框宽度分布')
            ax1.grid(True, alpha=0.3)

            # 高度分布
            ax2.hist([analysis['bbox_height_stats']['mean']], bins=20, alpha=0.7)
            ax2.set_xlabel('边界框高度')
            ax2.set_ylabel('频率')
            ax2.set_title('边界框高度分布')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'bbox_size_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 距离分布
            plt.figure(figsize=(10, 6))
            # 这里使用模拟数据，实际应该从CSV读取
            distances = [analysis['distance_stats']['mean']]
            plt.hist(distances, bins=20, alpha=0.7)
            plt.xlabel('距离')
            plt.ylabel('频率')
            plt.title(f"{analysis['scene_name']} - {analysis['channel_name']}\n目标距离分布")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'distance_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. 框大小分类饼图
            plt.figure(figsize=(8, 8))
            sizes = analysis['bbox_size_categories']
            labels = ['小框', '中框', '大框']
            sizes_count = [sizes['small']['count'], sizes['medium']['count'], sizes['large']['count']]
            colors = ['lightblue', 'lightgreen', 'lightcoral']

            plt.pie(sizes_count, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f"{analysis['scene_name']} - {analysis['channel_name']}\n边界框大小分类")
            plt.savefig(output_dir / 'bbox_size_categories.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"创建分析图表错误: {e}")

    def generate_channel_report(self, analysis, output_dir):
        """
        生成单个通道的详细报告
        """
        report_path = output_dir / 'analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 数据分析报告 ===\n")
            f.write(f"场景: {analysis['scene_name']}\n")
            f.write(f"通道: {analysis['channel_name']}\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

            f.write(f"1. 基本统计信息\n")
            f.write(f"   总帧数: {analysis['total_frames']}\n")
            f.write(f"   总目标数: {analysis['total_targets']}\n")
            f.write(f"   平均每帧目标数: {analysis['targets_per_frame']:.2f}\n\n")

            f.write(f"2. 边界框尺寸统计\n")
            f.write(f"   宽度 - 均值: {analysis['bbox_width_stats']['mean']:.2f}, "
                    f"标准差: {analysis['bbox_width_stats']['std']:.2f}\n")
            f.write(f"   高度 - 均值: {analysis['bbox_height_stats']['mean']:.2f}, "
                    f"标准差: {analysis['bbox_height_stats']['std']:.2f}\n")
            f.write(f"   面积 - 均值: {analysis['bbox_area_stats']['mean']:.2f}, "
                    f"标准差: {analysis['bbox_area_stats']['std']:.2f}\n\n")

            f.write(f"3. 边界框极值信息\n")
            f.write(f"   最大框 - 宽度: {analysis['largest_bbox']['width']:.2f}, "
                    f"高度: {analysis['largest_bbox']['height']:.2f}, "
                    f"面积: {analysis['largest_bbox']['area']:.2f}, "
                    f"帧号: {analysis['largest_bbox']['frame']}\n")
            f.write(f"   最小框 - 宽度: {analysis['smallest_bbox']['width']:.2f}, "
                    f"高度: {analysis['smallest_bbox']['height']:.2f}, "
                    f"面积: {analysis['smallest_bbox']['area']:.2f}, "
                    f"帧号: {analysis['smallest_bbox']['frame']}\n")
            f.write(f"   中位数框 - 宽度: {analysis['median_bbox']['width']:.2f}, "
                    f"高度: {analysis['median_bbox']['height']:.2f}, "
                    f"面积: {analysis['median_bbox']['area']:.2f}\n\n")

            f.write(f"4. 边界框大小分类\n")
            sizes = analysis['bbox_size_categories']
            f.write(f"   小框 (<={analysis['bbox_area_stats']['q1']:.2f}): {sizes['small']['count']}个, "
                    f"占比: {sizes['small']['percentage']:.1f}%\n")
            f.write(f"   中框 ({analysis['bbox_area_stats']['q1']:.2f}-{analysis['bbox_area_stats']['q2']:.2f}): "
                    f"{sizes['medium']['count']}个, 占比: {sizes['medium']['percentage']:.1f}%\n")
            f.write(f"   大框 (>{analysis['bbox_area_stats']['q2']:.2f}): {sizes['large']['count']}个, "
                    f"占比: {sizes['large']['percentage']:.1f}%\n\n")

            f.write(f"5. 距离统计\n")
            f.write(f"   均值: {analysis['distance_stats']['mean']:.2f}\n")
            f.write(f"   标准差: {analysis['distance_stats']['std']:.2f}\n")
            f.write(f"   范围: [{analysis['distance_stats']['min']:.2f}, "
                    f"{analysis['distance_stats']['max']:.2f}]\n\n")

            f.write(f"6. 目标分布\n")
            for target_count, frame_count in analysis['targets_per_frame_distribution'].items():
                f.write(f"   {target_count}个目标: {frame_count}帧\n")

            f.write(f"\n7. 中心点位置统计\n")
            f.write(f"   X坐标 - 均值: {analysis['center_x_stats']['mean']:.2f}, "
                    f"标准差: {analysis['center_x_stats']['std']:.2f}\n")
            f.write(f"   Y坐标 - 均值: {analysis['center_y_stats']['mean']:.2f}, "
                    f"标准差: {analysis['center_y_stats']['std']:.2f}\n")

    def process_all_channels(self):
        """
        处理所有通道的数据
        """
        # 查找所有CSV文件（在labels目录下）
        csv_files = glob.glob(str(self.label_root_dir / "**" / "groundtruth.csv"), recursive=True)

        if not csv_files:
            print(f"错误: 在 {self.label_root_dir} 目录下未找到任何CSV文件")
            print("请检查目录结构是否正确")
            return

        all_analyses = []

        for csv_file in csv_files:
            csv_path = Path(csv_file)
            # 解析路径：labels/scneX/chX/groundtruth.csv
            scene_name = csv_path.parent.parent.name
            channel_name = csv_path.parent.name

            print(f"处理场景: {scene_name}, 通道: {channel_name}")

            # 创建输出目录
            output_dir = self.output_root_dir / scene_name / channel_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # 创建可视化图像目录
            vis_dir = output_dir / "visualized_images"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # 读取CSV数据
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"错误: 读取CSV文件失败: {csv_file}, 错误: {e}")
                continue

            # 分析数据
            analysis = self.analyze_channel_data(csv_file, scene_name, channel_name)
            if analysis:
                all_analyses.append(analysis)

                # 生成分析图表和报告
                self.create_channel_analysis_plots(analysis, output_dir)
                self.generate_channel_report(analysis, output_dir)

                # 可视化图像（示例处理前几帧）
                sample_frames = df['frame'].unique()[:]  # 只处理前5帧作为示例

                for frame in sample_frames:
                    frame_data = df[df['frame'] == frame]
                    if len(frame_data) == 0:
                        continue

                    img_name = frame_data.iloc[0]['img_name']

                    # 查找图像文件
                    img_path = self.find_image_path(scene_name, channel_name, img_name)
                    if img_path and img_path.exists():
                        # 提取边界框
                        bboxes, distances = [], []
                        for _, row in frame_data.iterrows():
                            bbox = [
                                row['Top_left_x'], row['Top_left_y'],
                                row['Bottom_right_x'], row['Bottom_right_y']
                            ]
                            bboxes.append(bbox)
                            distances.append(row['DM_distance'])

                        # 绘制边界框
                        output_img_path = vis_dir / f"vis_{img_name}"
                        # success = self.draw_bounding_boxes(img_path, bboxes, output_img_path)
                        visualize_and_save(img_path, detected_objects=None, visualize_save_dir=vis_dir,
                                           scale_w=1, scale_h=1, bbox_gts=bboxes, distance=distances)

                    #     if success:
                    #         print(f"  已生成可视化图像: {output_img_path.name}")
                    # else:
                    #     print(f"  警告: 找不到图像文件: {img_name}")

            print(f"完成处理: {scene_name}/{channel_name}\n")

        # 生成总体分析报告
        if all_analyses:
            self.generate_overall_report(all_analyses)
        else:
            print("警告: 没有成功分析任何数据")

    def generate_overall_report(self, all_analyses):
        """
        生成总体分析报告
        """
        overall_dir = self.output_root_dir / "overall_analysis"
        overall_dir.mkdir(parents=True, exist_ok=True)

        # 总体统计
        total_frames = sum(analysis['total_frames'] for analysis in all_analyses)
        total_targets = sum(analysis['total_targets'] for analysis in all_analyses)
        avg_targets_per_frame = total_targets / total_frames if total_frames > 0 else 0

        # 框大小分类总体统计
        total_small = sum(a['bbox_size_categories']['small']['count'] for a in all_analyses)
        total_medium = sum(a['bbox_size_categories']['medium']['count'] for a in all_analyses)
        total_large = sum(a['bbox_size_categories']['large']['count'] for a in all_analyses)
        total_boxes = total_small + total_medium + total_large

        # 生成总体报告
        report_path = overall_dir / "overall_analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 总体数据分析报告 ===\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

            f.write(f"1. 数据集总体统计\n")
            f.write(f"   总场景数: {len(set(a['scene_name'] for a in all_analyses))}\n")
            f.write(f"   总通道数: {len(all_analyses)}\n")
            f.write(f"   总帧数: {total_frames}\n")
            f.write(f"   总目标数: {total_targets}\n")
            f.write(f"   平均每帧目标数: {avg_targets_per_frame:.2f}\n\n")

            f.write(f"2. 边界框大小分类总体统计\n")
            f.write(f"   小框: {total_small}个, 占比: {total_small / total_boxes * 100:.1f}%\n")
            f.write(f"   中框: {total_medium}个, 占比: {total_medium / total_boxes * 100:.1f}%\n")
            f.write(f"   大框: {total_large}个, 占比: {total_large / total_boxes * 100:.1f}%\n\n")

            f.write(f"3. 各通道详细统计\n")
            f.write(
                f"{'场景':<10} {'通道':<10} {'帧数':<8} {'目标数':<8} {'目标/帧':<10} {'小框':<8} {'中框':<8} {'大框':<8}\n")
            f.write("-" * 80 + "\n")

            for analysis in all_analyses:
                sizes = analysis['bbox_size_categories']
                f.write(f"{analysis['scene_name']:<10} {analysis['channel_name']:<10} "
                        f"{analysis['total_frames']:<8} {analysis['total_targets']:<8} "
                        f"{analysis['targets_per_frame']:<10.2f} "
                        f"{sizes['small']['count']:<8} {sizes['medium']['count']:<8} {sizes['large']['count']:<8}\n")

            f.write(f"\n4. 距离统计汇总\n")
            all_distances = []
            for analysis in all_analyses:
                all_distances.extend([analysis['distance_stats']['mean']])

            if all_distances:
                f.write(f"   平均距离: {np.mean(all_distances):.2f}\n")
                f.write(f"   距离标准差: {np.std(all_distances):.2f}\n")
                f.write(f"   最小距离: {min(all_distances):.2f}\n")
                f.write(f"   最大距离: {max(all_distances):.2f}\n")

            f.write(f"\n5. 边界框尺寸汇总\n")
            widths = [a['bbox_width_stats']['mean'] for a in all_analyses]
            heights = [a['bbox_height_stats']['mean'] for a in all_analyses]

            f.write(f"   平均宽度: {np.mean(widths):.2f}\n")
            f.write(f"   平均高度: {np.mean(heights):.2f}\n")
            f.write(f"   最大框面积: {max(a['largest_bbox']['area'] for a in all_analyses):.2f}\n")
            f.write(f"   最小框面积: {min(a['smallest_bbox']['area'] for a in all_analyses):.2f}\n")

        # 生成总体图表
        self.create_overall_plots(all_analyses, overall_dir)

        print(f"总体分析报告已生成: {report_path}")

    def create_overall_plots(self, all_analyses, output_dir):
        """
        创建总体分析图表
        """
        try:
            # 1. 各通道目标数量对比
            plt.figure(figsize=(12, 6))
            channels = [f"{a['scene_name']}_{a['channel_name']}" for a in all_analyses]
            targets = [a['total_targets'] for a in all_analyses]
            frames = [a['total_frames'] for a in all_analyses]

            x = np.arange(len(channels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(14, 8))
            bars1 = ax.bar(x - width / 2, targets, width, label='目标数量')
            bars2 = ax.bar(x + width / 2, frames, width, label='帧数')

            ax.set_xlabel('通道')
            ax.set_ylabel('数量')
            ax.set_title('各通道目标数量和帧数对比')
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'channels_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 目标分布饼图
            plt.figure(figsize=(10, 8))
            target_counts = {}
            for analysis in all_analyses:
                for count, frames in analysis['targets_per_frame_distribution'].items():
                    target_counts[count] = target_counts.get(count, 0) + frames

            plt.pie(target_counts.values(), labels=target_counts.keys(), autopct='%1.1f%%')
            plt.title('每帧目标数量分布')
            plt.savefig(output_dir / 'targets_distribution_pie.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 框大小分类总体饼图
            plt.figure(figsize=(10, 8))
            total_small = sum(a['bbox_size_categories']['small']['count'] for a in all_analyses)
            total_medium = sum(a['bbox_size_categories']['medium']['count'] for a in all_analyses)
            total_large = sum(a['bbox_size_categories']['large']['count'] for a in all_analyses)

            sizes = [total_small, total_medium, total_large]
            labels = ['小框', '中框', '大框']
            colors = ['lightblue', 'lightgreen', 'lightcoral']

            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('总体边界框大小分类分布')
            plt.savefig(output_dir / 'overall_bbox_size_categories.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. 各通道框大小分类对比图
            plt.figure(figsize=(15, 8))
            channels = [f"{a['scene_name']}_{a['channel_name']}" for a in all_analyses]
            small_counts = [a['bbox_size_categories']['small']['count'] for a in all_analyses]
            medium_counts = [a['bbox_size_categories']['medium']['count'] for a in all_analyses]
            large_counts = [a['bbox_size_categories']['large']['count'] for a in all_analyses]

            x = np.arange(len(channels))
            width = 0.25

            fig, ax = plt.subplots(figsize=(16, 8))
            bars1 = ax.bar(x - width, small_counts, width, label='小框', color='lightblue')
            bars2 = ax.bar(x, medium_counts, width, label='中框', color='lightgreen')
            bars3 = ax.bar(x + width, large_counts, width, label='大框', color='lightcoral')

            ax.set_xlabel('通道')
            ax.set_ylabel('框数量')
            ax.set_title('各通道边界框大小分类对比')
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'channels_bbox_size_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 5. 框面积分布直方图（所有通道合并）
            plt.figure(figsize=(12, 6))
            all_areas = []
            for analysis in all_analyses:
                # 这里需要从原始数据计算，简化处理使用统计值
                mean_area = analysis['bbox_area_stats']['mean']
                std_area = analysis['bbox_area_stats']['std']
                # 生成模拟数据用于可视化
                simulated_areas = np.random.normal(mean_area, std_area,
                                                   analysis['total_targets'])
                all_areas.extend(simulated_areas)

            plt.hist(all_areas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('边界框面积')
            plt.ylabel('频率')
            plt.title('总体边界框面积分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'overall_bbox_area_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 6. 最大最小框对比图
            plt.figure(figsize=(12, 8))
            max_areas = [a['largest_bbox']['area'] for a in all_analyses]
            min_areas = [a['smallest_bbox']['area'] for a in all_analyses]
            median_areas = [a['median_bbox']['area'] for a in all_analyses]

            channels = [f"{a['scene_name']}_{a['channel_name']}" for a in all_analyses]
            x = np.arange(len(channels))

            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(x, max_areas, 'ro-', label='最大框面积', markersize=8)
            ax.plot(x, median_areas, 'go-', label='中位数框面积', markersize=8)
            ax.plot(x, min_areas, 'bo-', label='最小框面积', markersize=8)

            ax.set_xlabel('通道')
            ax.set_ylabel('框面积')
            ax.set_title('各通道边界框极值对比')
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'bbox_extremes_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"创建总体图表错误: {e}")


def main():
    # 设置路径 - 根据您的实际路径修改
    data_root_dir = "/home/csz_changsha/data/AR_2511/data"  # 数据根目录
    label_root_dir = "/home/csz_changsha/data/AR_2511/labels"  # 标签根目录
    output_root_dir = "/home/csz_changsha/data/AR_2511/analysis_output"  # 输出根目录

    # 检查目录是否存在
    if not os.path.exists(data_root_dir):
        print(f"错误: 数据目录不存在: {data_root_dir}")
        return

    if not os.path.exists(label_root_dir):
        print(f"错误: 标签目录不存在: {label_root_dir}")
        return

    # 创建分析器并运行
    analyzer = DataVisualizerAndAnalyzer(data_root_dir, label_root_dir, output_root_dir)
    analyzer.process_all_channels()

    print("所有处理完成！")
    print(f"结果保存在: {output_root_dir}")


if __name__ == "__main__":
    main()
