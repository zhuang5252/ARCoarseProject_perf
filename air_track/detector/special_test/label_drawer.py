import shutil

import cv2
import os
from pathlib import Path


def draw_bboxes(image_path, label_path, output_dir):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return

    h, w = image.shape[:2]

    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 绘制每个边界框
    for line in lines:
        data = list(map(float, line.strip().split()))
        if len(data) != 5:
            continue

        class_id, x_center, y_center, width, height = data

        # 转换为绝对坐标
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # 计算矩形坐标
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 绘制矩形和类别
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(image, f'Class {int(class_id)}', (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存结果
    rel_path = Path(image_path).relative_to(base_dir)
    output_path = output_dir / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


# 配置路径
base_dir = Path("/home/csz_changsha/data/yanqing_data/hongniu_0526_1")  # 修改为你的父文件夹路径
output_dir = Path("/home/csz_changsha/data/yanqing_data/hongniu_0526_out")  # 修改为输出目录

# 遍历所有子文件夹
for subfolder in base_dir.iterdir():
    if subfolder.is_dir():
        rgb_dir = subfolder / "rgb"
        label_dir = subfolder / "labels"

        # 遍历所有图像文件
        for img_path in rgb_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                # 匹配对应的标签文件
                label_path = label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    draw_bboxes(str(img_path), str(label_path), output_dir)
                else:
                    rel_path = Path(img_path).relative_to(base_dir)
                    output_path = output_dir / rel_path
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(img_path, output_path)
                    print(f"Label file missing for: {img_path}")

print("处理完成！输出目录:", output_dir)
