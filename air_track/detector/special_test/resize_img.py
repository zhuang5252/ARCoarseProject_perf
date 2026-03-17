# -*- coding: utf-8 -*-
# @Author    : 
# @File      : resize_img.py
# @Created   : 2025/10/23 下午4:46
# @Desc      : 


import os
import glob
from PIL import Image
import matplotlib.pyplot as plt


def resize_images(input_folder, output_folder, target_size=(64, 64)):
    """
    将文件夹中的所有图片resize到指定尺寸

    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        target_size: 目标尺寸 (width, height)
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

    # 统计处理的图片数量
    processed_count = 0
    failed_files = []

    print(f"开始处理图片...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"目标尺寸: {target_size[0]}x{target_size[1]}")
    print("-" * 50)

    # 遍历所有支持的图片格式
    for extension in extensions:
        pattern = os.path.join(input_folder, extension)
        files = glob.glob(pattern)

        for file_path in files:
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 获取原始尺寸
                    original_size = img.size
                    print(f"处理图片: {os.path.basename(file_path)} ({original_size[0]}x{original_size[1]})")

                    # Resize图片 (使用高质量的重采样算法)
                    resized_img = img.resize(target_size, Image.LANCZOS)

                    # 生成输出文件路径
                    filename = os.path.basename(file_path)
                    output_path = os.path.join(output_folder, filename)

                    # 保存图片（保持原始格式和质量）
                    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                        resized_img.save(output_path, 'JPEG', quality=95)
                    else:
                        resized_img.save(output_path)

                    processed_count += 1
                    print(f"  ✓ 已保存: {output_path} ({target_size[0]}x{target_size[1]})")

            except Exception as e:
                error_msg = f"处理 {os.path.basename(file_path)} 时出错: {e}"
                print(f"  ✗ {error_msg}")
                failed_files.append((os.path.basename(file_path), str(e)))

    # 输出处理结果统计
    print("-" * 50)
    print(f"处理完成！")
    print(f"成功处理: {processed_count} 张图片")

    if failed_files:
        print(f"处理失败: {len(failed_files)} 张图片")
        print("\n失败文件列表:")
        for filename, error in failed_files:
            print(f"  {filename}: {error}")
    else:
        print("所有图片处理成功！")

    return processed_count, failed_files


def create_sample_images(input_folder, num_images=5):
    """
    创建示例图片用于测试（可选功能）
    """
    os.makedirs(input_folder, exist_ok=True)

    print(f"创建 {num_images} 个16x16示例图片...")

    # 创建不同颜色的示例图片
    colors_and_shapes = [
        ("red_square.png", (255, 0, 0), "square"),
        ("green_circle.png", (0, 255, 0), "circle"),
        ("blue_triangle.png", (0, 0, 255), "triangle"),
        ("yellow_star.png", (255, 255, 0), "star"),
        ("purple_diamond.png", (128, 0, 128), "diamond"),
    ]

    created_count = 0
    for filename, color, shape in colors_and_shapes[:num_images]:
        try:
            # 创建16x16的图片
            img = Image.new('RGB', (16, 16), (255, 255, 255))
            pixels = img.load()

            # 根据形状类型绘制不同的图案
            if shape == "square":
                # 绘制彩色方块
                for i in range(4, 12):
                    for j in range(4, 12):
                        pixels[i, j] = color

            elif shape == "circle":
                # 绘制圆形
                center_x, center_y, radius = 8, 8, 4
                for i in range(16):
                    for j in range(16):
                        if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
                            pixels[i, j] = color

            elif shape == "triangle":
                # 绘制三角形
                for i in range(16):
                    for j in range(16):
                        if i >= j and i + j >= 15:
                            pixels[i, j] = color

            elif shape == "star":
                # 绘制星形
                for i in range(16):
                    for j in range(16):
                        if i == 8 or j == 8 or abs(i - 8) == abs(j - 8):
                            if 3 <= i <= 12 and 3 <= j <= 12:
                                pixels[i, j] = color

            elif shape == "diamond":
                # 绘制菱形
                center = 8
                for i in range(16):
                    for j in range(16):
                        if abs(i - center) + abs(j - center) <= 5:
                            pixels[i, j] = color

            # 保存图片
            img.save(os.path.join(input_folder, filename))
            created_count += 1
            print(f"  ✓ 创建: {filename}")

        except Exception as e:
            print(f"  ✗ 创建 {filename} 失败: {e}")

    print(f"示例图片创建完成！共创建 {created_count} 张图片")
    return created_count


def show_comparison(input_folder, output_folder, example_filename):
    """
    显示处理前后的图片对比（可选功能）
    """
    try:
        input_path = os.path.join(input_folder, example_filename)
        output_path = os.path.join(output_folder, example_filename)

        if not os.path.exists(input_path) or not os.path.exists(output_path):
            print("对比图片不存在")
            return

        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        with Image.open(input_path) as input_img:
            ax1.imshow(input_img)
            ax1.set_title(f'原始图片 - {input_img.size[0]}×{input_img.size[1]}', fontsize=14)
            ax1.axis('off')

        with Image.open(output_path) as output_img:
            ax2.imshow(output_img)
            ax2.set_title(f'Resize后 - {output_img.size[0]}×{output_img.size[1]}', fontsize=14)
            ax2.axis('off')

        plt.tight_layout()
        comparison_file = "resize_comparison.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"图片对比已保存为: {comparison_file}")
        plt.show()

    except Exception as e:
        print(f"生成对比图时出错: {e}")


def main():
    """
    主函数 - 直接运行此程序即可使用
    """
    # 设置文件夹路径
    input_folder = "/home/csz_changsha/ATP_DELIVER/ATP_example/secondary_classifier_2/input"  # 输入图片文件夹（存放16x16图片）
    output_folder = "/home/csz_changsha/ATP_DELIVER/ATP_example/secondary_classifier_2/input_64_64"  # 输出图片文件夹

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"输入文件夹 '{input_folder}' 不存在")
        create_sample = input("是否创建示例图片？(y/n): ").lower().strip()
        if create_sample == 'y':
            create_sample_images(input_folder)
        else:
            print("请创建输入文件夹并放入16x16的图片")
            return

    # 检查输入文件夹中是否有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"输入文件夹 '{input_folder}' 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片待处理")

    # 执行resize操作
    processed_count, failed_files = resize_images(input_folder, output_folder, (64, 64))

    # 显示处理结果摘要
    if processed_count > 0:
        print("\n" + "=" * 60)
        print("处理结果摘要:")
        print("=" * 60)

        # 显示几个处理后的文件信息
        print("\n处理后的文件列表:")
        output_files = os.listdir(output_folder)
        for i, filename in enumerate(output_files[:5]):  # 只显示前5个
            file_path = os.path.join(output_folder, filename)
            with Image.open(file_path) as img:
                size = img.size
                print(f"  {i + 1}. {filename} - {size[0]}x{size[1]}")

        if len(output_files) > 5:
            print(f"  ... 还有 {len(output_files) - 5} 个文件")

        # 询问是否显示对比图
        show_compare = input("\n是否显示处理前后对比图？(y/n): ").lower().strip()
        if show_compare == 'y' and output_files:
            show_comparison(input_folder, output_folder, output_files[0])

    print("\n程序执行完毕！")


if __name__ == "__main__":
    main()
