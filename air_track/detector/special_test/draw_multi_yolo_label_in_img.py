import os
import cv2
from pathlib import Path
from air_track.detector.utils.detect_utils import cxcywhn2xyxy
from air_track.detector.visualization.visualize_and_save import visualize_and_save


def get_all_folders(path, target_depth, current_depth=1, file_end=None):
    """
    递归找出path路径下的target_depth深度的文件完整路径，并保存列表。
    如果file_end为None，返回所有文件路径，否则只返回以file_end中字符串结尾的文件路径。
    """
    folders_paths = []

    if current_depth == target_depth:
        for f in os.listdir(path):
            temp = os.path.join(path, f)
            if os.path.isdir(temp):
                if file_end is None or any(f.endswith(end) for end in file_end):
                    folders_paths.append(temp)
    else:
        for f in os.listdir(path):
            temp = os.path.join(path, f)
            if os.path.isdir(temp):
                folders_paths.extend(get_all_folders(temp, target_depth, current_depth + 1, file_end))

    return folders_paths


def get_all_images(path, img_folder='rgb', file_end=None):
    # 遍历所有子文件夹
    path = Path(path)
    img_paths = []
    for sub_folder in path.iterdir():
        if sub_folder.is_dir():
            img_dir = os.path.join(sub_folder, img_folder)
            if not os.path.exists(img_dir):
                continue
            for f in os.listdir(img_dir):
                temp = os.path.join(img_dir, f)
                if os.path.isfile(temp):
                    if file_end is None or any(f.endswith(end) for end in file_end):
                        img_paths.append(temp)

    return img_paths


def transform_target(row, shape):
    # 判断是否存在目标
    if int(row['Target_conut']):
        top_left = (int(float(row['Top_left_x']) + 0.5), int(float(row['Top_left_y']) + 0.5))
        bottom_right = (int(float(row['Bottom_right_x']) + 0.5), int(float(row['Bottom_right_y']) + 0.5))
        center = (float(row['Center_x']), float(row['Center_y']))
        w = (bottom_right[0] - top_left[0])
        h = (bottom_right[1] - top_left[1])

        return top_left, bottom_right
    else:
        return None, None


def draw_label_in_img(img, top_left, bottom_right, save_path, max_pixel=255):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if top_left and bottom_right:
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        # 图像上绘制框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, max_pixel, 0), 1)  # 在图像上绘制矩形

    # 保存带有标签框的图像
    cv2.imwrite(save_path, img)
    print('saved: ', save_path)


def process(img_paths, save_path,
            img_folder='images', label_folder='labels',
            img_format='.jpg', gt_format='.txt'):
    # 遍历每张图片
    for i, img_path in enumerate(img_paths):
        print(f'当前处理第{i + 1}/{len(img_paths)}帧')
        # 读取图片
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
        save_img_path = img_path.replace(base_path, save_path)
        img = cv2.imread(img_path)
        shape = img.shape[:2]

        label_path = img_path.replace(f'/{img_folder}/', f'/{label_folder}/')
        label_path = label_path.replace(img_format, gt_format)

        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                labels = file.readlines()
            labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的
            # 将标签由字符串转换为浮点数
            labels = [[float(item) for item in label] for label in labels]
        else:
            labels = []

        # 将标签从cxcywhn转换为xyxy
        bbox_xyxy = []
        for label in labels:
            bbox_gt = cxcywhn2xyxy(label[1:], img_w=shape[1], img_h=shape[0])
            bbox_xyxy.append([int(label[0])] + bbox_gt)

        visualize_and_save(img_path, visualize_save_dir=os.path.dirname(save_img_path), bbox_gts=bbox_xyxy)


if __name__ == '__main__':
    # path = '/home/csz_changsha/data/AR_2410/normal_airtrack_test_1'
    # save_path = '/home/csz_changsha/data/AR_2410/output'
    # img_paths = get_all_files(path, target_depth=4, file_end=['.png'])
    # label_paths = get_all_files(path, target_depth=2, file_end=['.csv'])
    #
    # process(img_paths, label_paths, save_path, max_pixel=65535)
    print()

    path = '/home/csz_changsha/data/FlySubjectAirToGround_merge/dataset_have_target'
    save_path = '/home/csz_changsha/data/FlySubjectAirToGround_merge/dataset_have_target1111'
    img_paths = get_all_images(path, img_folder='images', file_end=['.jpg'])

    process(img_paths, save_path, img_folder='images', label_folder='labels', img_format='.jpg', gt_format='.txt')
