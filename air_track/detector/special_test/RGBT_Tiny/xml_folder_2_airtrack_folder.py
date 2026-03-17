import os
import shutil
from tqdm import tqdm


def reorganize_dataset_structure(original_root, new_root):
    """
    重新组织数据集结构
    :param original_root: 原始数据集根目录
    :param new_root: 新数据集根目录
    """
    # 确保新目录存在
    os.makedirs(new_root, exist_ok=True)

    # 创建rgb和ir目录
    rgb_dir = os.path.join(new_root, 'rgb')
    ir_dir = os.path.join(new_root, 'ir')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    # 处理原始images和labels目录
    original_images = os.path.join(original_root, 'images')
    original_labels = os.path.join(original_root, 'labels')

    # 正确获取最外层目录（DJI_xxxx_x）
    top_dirs = set()
    for entry in os.listdir(original_images):
        if os.path.isdir(os.path.join(original_images, entry)) and entry.startswith('DJI_'):
            top_dirs.add(entry)

    print(f"发现 {len(top_dirs)} 个最外层目录: {', '.join(top_dirs)}")

    # 处理每个最外层目录
    for top_dir in tqdm(top_dirs, desc="重组数据结构"):
        # 处理images目录
        original_img_top = os.path.join(original_images, top_dir)

        # 检查00和01子目录
        for sub_dir in ['00', '01']:
            original_img_sub = os.path.join(original_img_top, sub_dir)
            if not os.path.exists(original_img_sub):
                print(f"警告: 目录不存在 - {original_img_sub}")
                continue

            # 确定目标目录（rgb或ir）
            target_parent = rgb_dir if sub_dir == '00' else ir_dir
            target_top = os.path.join(target_parent, top_dir)

            # 创建目标目录结构
            target_images = os.path.join(target_top, 'images')
            os.makedirs(target_images, exist_ok=True)

            # 移动图像文件
            img_files = [f for f in os.listdir(original_img_sub)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not img_files:
                print(f"警告: 无图像文件 - {original_img_sub}")
                continue

            for img_file in img_files:
                src = os.path.join(original_img_sub, img_file)
                dst = os.path.join(target_images, img_file)
                shutil.move(src, dst)

        # 处理labels目录
        original_label_top = os.path.join(original_labels, top_dir)
        if not os.path.exists(original_label_top):
            print(f"警告: 标签目录不存在 - {original_label_top}")
            continue

        # 检查00和01子目录
        for sub_dir in ['00', '01']:
            original_label_sub = os.path.join(original_label_top, sub_dir)
            if not os.path.exists(original_label_sub):
                print(f"警告: 标签子目录不存在 - {original_label_sub}")
                continue

            # 确定目标目录（rgb或ir）
            target_parent = rgb_dir if sub_dir == '00' else ir_dir
            target_top = os.path.join(target_parent, top_dir)

            # 创建目标目录结构
            target_labels = os.path.join(target_top, 'labels')
            os.makedirs(target_labels, exist_ok=True)

            # 移动标签文件
            label_files = [f for f in os.listdir(original_label_sub)
                           if f.lower().endswith('.txt')]

            if not label_files:
                print(f"警告: 无标签文件 - {original_label_sub}")
                continue

            for label_file in label_files:
                src = os.path.join(original_label_sub, label_file)
                dst = os.path.join(target_labels, label_file)
                shutil.move(src, dst)

    # 移动classes.txt文件
    original_classes = os.path.join(original_labels, 'classes.txt')
    if os.path.exists(original_classes):
        shutil.copy2(original_classes, new_root)  # 使用copy2保留元数据
        print(f"已复制 classes.txt 到 {new_root}")
    else:
        print("警告: 未找到 classes.txt 文件")


def main(original_root, new_root):
    # 验证原始目录结构
    required_dirs = ['images', 'labels']
    for req_dir in required_dirs:
        if not os.path.exists(os.path.join(original_root, req_dir)):
            print(f"错误: 原始目录中缺少必需的'{req_dir}'目录！")
            return

    # 执行重组操作
    reorganize_dataset_structure(original_root, new_root)


if __name__ == "__main__":
    orig_root = '/home/csz_changsha/data/RGBT-Tiny'
    new_root = '/home/csz_changsha/data/RGBT-Tiny_new'
    main(orig_root, new_root)
