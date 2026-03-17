import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm


def parse_xml_info(xml_path, root_dir):
    """解析XML文件获取图像尺寸、类别信息和所属目录"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 获取所有类别
    classes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text.strip().lower()
        classes.append(cls)

    # 获取所属的最外层子目录
    rel_path = os.path.relpath(xml_path, start=root_dir)
    top_dir = rel_path.split(os.sep)[0]

    return (width, height), classes, top_dir


def process_dataset(input_dir):
    """处理整个数据集，返回图像尺寸、类别统计、XML文件数量和目录映射"""
    # 初始化统计变量
    img_sizes = set()
    class_counts = defaultdict(int)
    class_dirs = defaultdict(set)  # 存储每个类别对应的目录
    xml_files = []
    xml_count = 0

    print("正在扫描XML文件并统计信息...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                try:
                    (width, height), classes, top_dir = parse_xml_info(xml_path, input_dir)
                    img_sizes.add((width, height))
                    for cls in classes:
                        class_counts[cls] += 1
                        class_dirs[cls].add(top_dir)
                    xml_files.append(xml_path)
                    xml_count += 1
                except Exception as e:
                    print(f"解析 {xml_path} 失败: {str(e)}")

    # 检查图像尺寸是否一致
    if len(img_sizes) > 1:
        print(f"警告: 发现多种图像尺寸: {img_sizes}，将使用第一种")
    img_size = next(iter(img_sizes)) if img_sizes else (0, 0)

    # 将目录集合转换为排序后的列表
    class_dirs = {k: sorted(list(v)) for k, v in class_dirs.items()}

    return img_size, dict(class_counts), xml_count, xml_files, class_dirs


def write_classes_file(output_dir, img_size, class_counts, xml_count, class_dirs):
    """生成包含完整统计信息的classes.txt文件"""
    classes_file = os.path.join(output_dir, 'classes.txt')
    total_samples = sum(class_counts.values())

    with open(classes_file, 'w') as f:
        # 基础信息部分
        f.write("=== 数据集统计信息 ===\n\n")
        f.write(f"image_size: {img_size[0]}x{img_size[1]}\n")
        f.write(f"xml_file_count: {xml_count}\n")
        f.write(f"total_samples: {total_samples}\n")
        f.write(f"unique_classes: {len(class_counts)}\n\n")

        # 类别统计部分
        f.write("=== 类别分布 ===\n")

        # 按样本数降序排列
        # sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        # 按类别名称排序
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[0])

        # 计算百分比
        max_len = max(len(cls) for cls in class_counts.keys())
        for cls, count in sorted_classes:
            percentage = (count / total_samples) * 100
            f.write(f"{cls.ljust(max_len)} : {str(count).rjust(6)} ({percentage:.2f}%)\n")

        # 新增：类别-目录映射部分
        f.write("\n=== 类别来源目录 ===\n")
        # 按样本数降序排列
        # for cls, dirs in sorted(class_dirs.items(), key=lambda x: len(x[1]), reverse=True):
        # 按类别名称排序
        for cls, dirs in sorted(class_dirs.items(), key=lambda x: x[0]):
            f.write(f"\n{cls} (共 {len(dirs)} 个目录):\n")
            # 每行显示5个目录，整齐排列
            for i in range(0, len(dirs), 5):
                line_dirs = dirs[i:i + 5]
                f.write("    " + ", ".join(f"{d:<15}" for d in line_dirs) + "\n")

    print(f"\n已生成详细的统计文件: {classes_file}")
    return classes_file


def xml_to_yolo(xml_path, img_size, class_dict, output_dir, input_dir, keep_classes=None):
    """将单个XML文件转换为YOLO格式
    Args:
        xml_path: XML文件路径
        img_size: 图像尺寸 (width, height)
        class_dict: 类别到ID的映射字典
        output_dir: 输出目录
        input_dir: 输入目录
        keep_classes: 要保留的类别集合，None表示保留所有
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取相对路径保持原结构
    rel_path = os.path.relpath(xml_path, start=input_dir)
    yolo_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(yolo_path), exist_ok=True)

    img_width, img_height = img_size

    yolo_lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text.strip().lower()

        # 如果设置了保留类别且当前类别不在保留列表中，则跳过
        if keep_classes is not None and cls not in keep_classes:
            continue

        cls_id = class_dict[cls]

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # 转换为YOLO格式
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 确保坐标在[0,1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 写入YOLO格式文件（即使没有目标也要创建空文件）
    yolo_file = os.path.splitext(yolo_path)[0] + '.txt'
    with open(yolo_file, 'w') as f:
        f.write('\n'.join(yolo_lines))


def main(input_dir, output_dir, keep_classes=None):
    """
    Args:
        input_dir: 输入目录，包含VOC格式的XML文件
        output_dir: 输出目录，将保存YOLO格式的txt文件
        keep_classes: 要保留的类别列表，None表示保留所有
    """
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1：扫描数据集获取信息
    img_size, class_counts, xml_count, xml_files, class_dirs = process_dataset(input_dir)

    if not img_size[0] or not class_counts:
        print("错误: 未找到有效的XML文件或标注信息！")
        return

    # 打印简要统计信息
    print(f"\n发现 {xml_count} 个XML文件")
    print(f"共 {sum(class_counts.values())} 个标注样本")
    print(f"包含 {len(class_counts)} 个类别")

    # 生成类别映射字典
    sorted_classes = sorted(class_counts.keys())
    class_dict = {cls: idx for idx, cls in enumerate(sorted_classes)}

    # 步骤2：生成详细的classes.txt文件
    classes_file = write_classes_file(output_dir, img_size, class_counts, xml_count, class_dirs)

    # 如果指定了保留类别，检查这些类别是否存在
    if keep_classes is not None:
        keep_classes = set(cls.lower() for cls in keep_classes)
        missing_classes = keep_classes - set(class_dict.keys())
        if missing_classes:
            print(f"警告: 以下指定类别在数据集中不存在: {missing_classes}")
        # 只保留数据集中存在的类别
        keep_classes = keep_classes & set(class_dict.keys())
        print(f"将只保留以下类别: {sorted(keep_classes)}")

    # 步骤3：转换所有XML为YOLO格式
    print("\n开始转换XML到YOLO格式...")
    for xml_path in tqdm(xml_files, desc="转换进度"):
        try:
            xml_to_yolo(xml_path, img_size, class_dict, output_dir, input_dir, keep_classes)
        except Exception as e:
            print(f"转换 {xml_path} 失败: {str(e)}")

    print(f"\n转换完成！YOLO格式标签已保存到: {output_dir}")


if __name__ == "__main__":
    input_dir = '/home/csz_changsha/data/RGBT_Tiny_orig/annotations_voc'
    output_dir = '/home/csz_changsha/data/RGBT_Tiny_orig/labels'

    # 只保留"person"和"car"类别
    main(input_dir, output_dir, keep_classes=['bus', 'car'])

    # 保留所有类别
    # main(input_dir, output_dir)
