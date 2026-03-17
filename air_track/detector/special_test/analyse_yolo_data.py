import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict
from air_track.utils import combine_load_cfg_yaml


def get_labels_paths(base_path, label_folder='labels'):
    # 初始化一个空列表来存储 labels 文件夹的完整路径
    labels_paths = []
    # 遍历目录结构
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == label_folder:
                # 构建完整的 labels 文件夹路径
                labels_folder_path = os.path.join(root, dir_name)
                # 将路径添加到列表中
                labels_paths.append(labels_folder_path)

    labels_paths.sort()

    return labels_paths


def parse_yolo_annotations(txt_dir, img_width, img_height):
    """解析YOLO标注文件并返回包含实际尺寸的边界框数据"""
    bbox_data = []
    txt_files = glob(os.path.join(txt_dir, "**/*.txt"), recursive=True)
    txt_files.sort()

    for txt_file in txt_files:
        if 'classes.txt' in txt_file:
            print('跳过文件：', txt_file)
            continue
        # 从txt文件名获取图像名称（去除扩展名）
        image_name = os.path.basename(txt_file)

        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(f"跳过无效行：{txt_file}:{line_num}")
                    continue

                # 提取归一化的宽高并转换为实际像素尺寸
                class_id, _, _, w_norm, h_norm = parts[:5]
                try:
                    w_px = float(w_norm) * img_width  # 转换为实际宽度
                    h_px = float(h_norm) * img_height  # 转换为实际高度
                except ValueError:
                    continue

                bbox_data.append({
                    "image_name": image_name,
                    "class_id": class_id,
                    "width_px": w_px,
                    "height_px": h_px
                })

    return bbox_data


def calculate_statistics(bbox_data):
    """计算边界框的统计指标"""
    widths = np.array([b["width_px"] for b in bbox_data])
    heights = np.array([b["height_px"] for b in bbox_data])

    return {
        "width": {
            "max": np.max(widths),
            "min": np.min(widths),
            "mean": np.mean(widths),
            "std": np.std(widths)
        },
        "height": {
            "max": np.max(heights),
            "min": np.min(heights),
            "mean": np.mean(heights),
            "std": np.std(heights)
        }
    }


def save_statistics(stats, img_size, output_dir, class_stats=None):
    """保存统计结果到文本文件"""
    with open(os.path.join(output_dir, "statistics.txt"), 'w') as f:
        f.write(f"Image Size: {img_size}x{img_size}px\n\n")
        f.write("Width Statistics:\n")
        f.write(f"Max: {stats['width']['max']:.2f}px\n")
        f.write(f"Min: {stats['width']['min']:.2f}px\n")
        f.write(f"Mean: {stats['width']['mean']:.2f}px\n")
        f.write(f"Std: {stats['width']['std']:.2f}px\n\n")
        f.write("Height Statistics:\n")
        f.write(f"Max: {stats['height']['max']:.2f}px\n")
        f.write(f"Min: {stats['height']['min']:.2f}px\n")
        f.write(f"Mean: {stats['height']['mean']:.2f}px\n")
        f.write(f"Std: {stats['height']['std']:.2f}px\n")

        if class_stats:
            f.write("\nClass Statistics:\n")
            for cls, count in class_stats.items():
                f.write(f"{cls}: {count}\n")
            f.write(f"\nTotal Objects: {sum(class_stats.values())}\n")


def save_bbox_csv(bbox_data, output_dir):
    """保存边界框原始数据到CSV文件"""
    csv_path = os.path.join(output_dir, "bbox_details.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'class_id', 'width_px', 'height_px']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bbox_data)
    print(f"边界框明细已保存至 {csv_path}")


def plot_distributions(bbox_data, img_size, output_dir):
    """绘制分布图表"""
    plt.figure(figsize=(15, 6))

    # 宽度分布
    plt.subplot(1, 2, 1)
    sns.histplot([b["width_px"] for b in bbox_data],
                 bins=50, color='blue', kde=True)
    plt.title(f'Width Distribution {img_size} px)')
    plt.xlabel('Width (pixels)')

    # 高度分布
    plt.subplot(1, 2, 2)
    sns.histplot([b["height_px"] for b in bbox_data],
                 bins=50, color='red', kde=True)
    plt.title(f'Height Distribution {img_size} px)')
    plt.xlabel('Height (pixels)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_distribution.png'), dpi=300)
    plt.close()


def calculate_class_statistics(bbox_data, cfg_data):
    """计算类别统计信息"""
    class_stats = defaultdict(int)

    for bbox in bbox_data:
        try:
            class_id = int(bbox["class_id"])
            class_name = cfg_data['classes'][class_id]
            class_stats[class_name] += 1
        except (ValueError, KeyError, IndexError):
            continue

    return dict(class_stats)


def analyze_yolo_dataset(cfg_data, txt_dir, img_width, img_height, output_dir="output"):
    """主分析函数"""
    os.makedirs(output_dir, exist_ok=True)

    # 解析数据
    bbox_data = parse_yolo_annotations(txt_dir, img_width, img_height)
    if not bbox_data:
        print("未找到有效的边界框数据！")
        return

    # 计算统计
    stats = calculate_statistics(bbox_data)
    class_stats = calculate_class_statistics(bbox_data, cfg_data)

    # 控制台输出
    print(f"\n分析结果（图像尺寸 {img_width}x{img_height}px）：")
    print(f"宽度范围：{stats['width']['min']:.2f}px - {stats['width']['max']:.2f}px")
    print(f"平均宽度±标准差：{stats['width']['mean']:.2f}px ± {stats['width']['std']:.2f}px")
    print(f"高度范围：{stats['height']['min']:.2f}px - {stats['height']['max']:.2f}px")
    print(f"平均高度±标准差：{stats['height']['mean']:.2f}px ± {stats['height']['std']:.2f}px")
    print("\n类别统计：")
    for cls, count in class_stats.items():
        print(f"{cls}: {count}")
    print(f"总目标数: {sum(class_stats.values())}")

    # 保存结果
    save_statistics(stats, (img_width, img_height), output_dir, class_stats)
    save_bbox_csv(bbox_data, output_dir)
    plot_distributions(bbox_data, (img_width, img_height), output_dir)


if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(script_dir, 'config/dataset_dalachi.yaml')
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=[data_yaml])

    data_path = '/media/linana/2C5821EF5821B88A/yanqing_train_data/筛选数据/20250828/merge_all'
    labels_paths = get_labels_paths(data_path, label_folder='label')

    for label_path in labels_paths:
        analyze_yolo_dataset(
            cfg_data=cfg_data,
            txt_dir=label_path,
            img_width=1920,  # 实际图像宽度
            img_height=1080,  # 实际图像高度
            output_dir=os.path.dirname(label_path)
        )
