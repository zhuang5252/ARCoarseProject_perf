import os
import shutil

"""按照Aot数据集处理数据集文件结构"""


def rename_to_folder_names(directory, prefix):
    """重命名文件夹"""
    # 获取给定目录下的所有项
    for item in os.listdir(directory):
        # 构建完整的路径
        item_path = os.path.join(directory, item)
        # 检查这个路径是否是文件夹
        if os.path.isdir(item_path):
            # 构建新的文件夹名称
            new_name = f"{prefix}{item}"
            # 构建新的文件夹路径
            new_path = os.path.join(directory, new_name)
            # 重命名文件夹
            os.rename(item_path, new_path)


def process_images_folder(directory):
    """移动文件"""
    path = os.path.join(directory, 'Images')
    os.makedirs(path, exist_ok=True)

    source_folder = os.path.join(directory, os.path.basename(directory) + '_result')
    destination_folder = path

    # 移动文件夹
    try:
        shutil.move(source_folder, destination_folder)
        print(f"文件夹 {source_folder} 已成功移动到 {destination_folder}")
    except shutil.Error as e:
        print(f"移动文件夹时出错: {e}")


if __name__ == '__main__':
    # 使用示例：
    path = '/media/csz_changsha/2C5821EF5821B88A/data/AR_2410/normal'
    # rename_to_folder_names(path, 'part')

    for item in os.listdir(path):
        orig_path = os.path.join(path, item)
        process_images_folder(orig_path)
