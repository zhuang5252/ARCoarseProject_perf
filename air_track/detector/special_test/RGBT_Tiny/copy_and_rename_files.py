import os
import shutil
from glob import glob
import math


# TODO 脚本copy的数据存在问题


def copy_and_rename_files(source_root, target_root, parts, img_folder="rgb", label_folder="label", scale=0.1):
    # 创建目标文件夹
    os.makedirs(os.path.join(target_root, img_folder), exist_ok=True)
    os.makedirs(os.path.join(target_root, label_folder), exist_ok=True)

    # 获取所有大文件夹并按名称排序
    # big_folders = sorted(glob(os.path.join(source_root, "*/")))
    big_folders = [os.path.join(source_root, part) for part in parts]

    # 初始化计数器
    global_counter = 1

    for folder_idx, big_folder in enumerate(big_folders):
        print(f"Processing folder {folder_idx + 1}/{len(big_folders)}: {big_folder}")

        # 获取当前大文件夹中的图像和标签文件
        image_files = sorted(glob(os.path.join(big_folder, "images", "*.jpg")))
        label_files = sorted(glob(os.path.join(big_folder, "labels", "*.txt")))

        # 计算需要取的数量 (10%)
        num_files = len(image_files)
        num_to_take = math.ceil(num_files * scale)

        # 确保图像和标签文件数量一致且匹配
        assert len(image_files) == len(label_files), f"图像和标签数量不匹配: {big_folder}"

        # 提取前10%的文件
        selected_images = image_files[:num_to_take]
        selected_labels = label_files[:num_to_take]

        # 复制并重命名文件
        for i, (img_path, label_path) in enumerate(zip(selected_images, selected_labels)):
            if i == 28:
                print()
            # 新文件名
            new_img_name = f"{global_counter}.jpg"
            new_label_name = f"{global_counter}.txt"

            # 目标路径
            dest_img_path = os.path.join(target_root, img_folder, new_img_name)
            dest_label_path = os.path.join(target_root, label_folder, new_label_name)

            # 复制文件
            shutil.copy2(img_path, dest_img_path)
            shutil.copy2(label_path, dest_label_path)

            # 更新计数器
            global_counter += 1

            # 每处理100个文件打印一次进度
            if global_counter % 100 == 0:
                print(f"已处理 {global_counter} 个文件...")

    print(f"\n处理完成！共处理了 {global_counter - 1} 个文件。")


# 使用示例
if __name__ == "__main__":
    source_root = "/home/csz_changsha/data/RGBT-Tiny/ir"  # 替换为你的源文件夹路径
    target_root_1 = "/home/csz_changsha/data/RGBT-Tiny/val1"  # 替换为你的目标文件夹路径
    target_root_2 = "/home/csz_changsha/data/RGBT-Tiny/test1"  # 替换为你的目标文件夹路径

    part_val = [
        'DJI_0032_3', 'DJI_0032_5', 'DJI_0043_2', 'DJI_0047_3',
        'DJI_0061_2', 'DJI_0075_3', 'DJI_0077_2', 'DJI_0081_2',
        'DJI_0083_2', 'DJI_0085_2', 'DJI_0089_2', 'DJI_0095_3',
        'DJI_0135_1', 'DJI_0137_3', 'DJI_0141_2', 'DJI_0167_1',
        'DJI_0169_3', 'DJI_0171_2', 'DJI_0181_3', 'DJI_0183_2',
        'DJI_0225_1', 'DJI_0229_2', 'DJI_0237_2', 'DJI_0265_1',
        'DJI_0275_3', 'DJI_0279_2', 'DJI_0279_4', 'DJI_0283_1',
        'DJI_0287_1', 'DJI_0291_2', 'DJI_0303_2', 'DJI_0305_3',
        'DJI_0305_7', 'DJI_0309_4', 'DJI_0343_5'
    ]
    part_test = [
        'DJI_0067_2', 'DJI_0073_3', 'DJI_0133_2', 'DJI_0179_2', 'DJI_0227_1',
        'DJI_0271_2', 'DJI_0295_3', 'DJI_0327_3', 'DJI_0331_3', 'DJI_0351_2'
    ]

    copy_and_rename_files(source_root, target_root_1, part_val, img_folder="infrared", label_folder="label", scale=0.1)
    # copy_and_rename_files(source_root, target_root_2, part_test, img_folder="infrared", label_folder="label", scale=1)
