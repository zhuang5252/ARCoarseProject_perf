import os
import random
import shutil
import numpy as np
from air_track.utils import reprod_init


def calc_iou(i1, i2):
    x1 = i1['cx']
    y1 = i1['cy']
    w1 = i1['w']
    h1 = i1['h']

    x2 = i2['cx']
    y2 = i2['cy']
    w2 = i2['w']
    h2 = i2['h']

    ix_min = max(x1 - w1 / 2, x2 - w2 / 2)
    iy_min = max(y1 - h1 / 2, y2 - h2 / 2)
    ix_max = min(x1 + w1 / 2, x2 + w2 / 2)
    iy_max = min(y1 + h1 / 2, y2 + h2 / 2)

    iw = max(ix_max - ix_min, 0.)
    ih = max(iy_max - iy_min, 0.)

    intersections = iw * ih
    unions = (i1['w'] * i1['h'] + i2['w'] * i2['h'] - intersections)

    iou = intersections / unions
    return iou


def get_labels_paths(base_path):
    # 初始化一个空列表来存储 labels 文件夹的完整路径
    labels_paths = []
    # 遍历目录结构
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == 'labels':
                # 构建完整的 labels 文件夹路径
                labels_folder_path = os.path.join(root, dir_name)
                # 将路径添加到列表中
                labels_paths.append(labels_folder_path)

    labels_paths.sort()

    return labels_paths


def analyse_and_filter_labels(paths, orig_w=1280, orig_h=720, scale=2, center_range=10,
                              range_min=16, range_max=32, iou=False, iou_threshold=0.25):
    center_x = orig_w / 2
    center_y = orig_h / 2

    center_bbox = {'cx': center_x, 'cy': center_y, 'w': center_range, 'h': center_range}

    frame_count = 0
    obj_count = 0
    center_count = 0
    no_center_count = 0
    obj_16_32_count = 0
    less_than_32_count = 0
    filter_16_32_count = 0
    filter_less_than_32_count = 0
    no_obj_count = 0
    img_train_val_paths = []
    img_test_paths = []
    train_folder_len = len(paths) - 2
    for i, path in enumerate(paths):
        label_names = os.listdir(path)
        label_names.sort()
        for j, label_name in enumerate(label_names):
            label_path = os.path.join(path, label_name)
            frame_count += 1

            # 读取标签
            with open(label_path, 'r') as file:
                labels = file.readlines()
            labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的

            if np.array(labels).shape[-1]:
                # 统计每个类别的数量
                for label in labels:
                    obj_count += 1
                    cx = float(label[1]) * orig_w
                    cy = float(label[2]) * orig_h
                    orig_label_w = float(label[3]) * orig_w
                    orig_label_h = float(label[4]) * orig_h
                    w = float(label[3]) * orig_w / scale
                    h = float(label[4]) * orig_h / scale

                    label_bbox = {'cx': cx, 'cy': cy, 'w': orig_label_w, 'h': orig_label_h}

                    if range_min <= w <= range_max and range_min <= h <= range_max:
                        obj_16_32_count += 1
                    elif w <= range_max and h <= range_max:
                        less_than_32_count += 1

                    if iou:
                        flag = calc_iou(center_bbox, label_bbox) >= iou_threshold
                    else:
                        flag = center_x - center_range <= cx <= center_x + center_range and center_y - center_range <= cy <= center_y + center_range

                    if flag:
                        center_count += 1
                    else:
                        no_center_count += 1
                        if range_min <= w <= range_max and range_min <= h <= range_max:
                            filter_16_32_count += 1
                        elif w <= range_max and h <= range_max:
                            filter_less_than_32_count += 1
                            temp = label_path.replace('labels', 'images')
                            temp = temp.replace('.txt', '.jpg')
                            if i + 1 < train_folder_len:
                                img_train_val_paths.append(temp)
                            else:
                                img_test_paths.append(temp)
            else:
                no_obj_count += 1

    print(f"总帧数: {frame_count}")
    print(f"总目标数: {obj_count}")
    print(f"目标在中心{center_range}*{center_range}范围内: {center_count}")
    print(f"目标不在中心{center_range}*{center_range}范围内: {no_center_count}")
    print(f"目标尺寸在{range_min}-{range_max}范围内: {obj_16_32_count}")
    print(f"目标尺寸小于{range_max}范围内: {less_than_32_count}")
    print(f"目标不在中心{center_range}*{center_range}范围内且尺寸在{range_min}-{range_max}范围内: {filter_16_32_count}")
    print(f"目标不在中心{center_range}*{center_range}范围内且尺寸小于{range_max}范围内: {filter_less_than_32_count}")
    print(f"图片无目标的数量: {no_obj_count}")

    return img_train_val_paths, img_test_paths


def copy_images_and_labels(image_paths, target_dir, file):
    """准备拷贝图片和标签到对应的目录"""
    with open(file, 'w') as f:
        for i, img_path in enumerate(image_paths):
            print(f'当前处理第{i + 1}帧/{len(image_paths)}帧')
            # 确定图片和标签的目录与文件名
            img_dir = os.path.dirname(img_path)
            img_filename = os.path.basename(img_path)
            label_dir = os.path.join(os.path.dirname(img_dir), 'labels')
            label_filename = img_filename.replace('.jpg', '.txt')  # 假设标签文件与图片同名，但扩展名不同

            # 拷贝图片
            target_img_dir = os.path.join(target_dir, 'images')
            os.makedirs(target_img_dir, exist_ok=True)
            img_path_new = os.path.join(target_img_dir, f'{i:05d}.jpg')
            shutil.copy(img_path, img_path_new)

            # 拷贝标签
            target_label_dir = os.path.join(target_dir, 'labels')
            os.makedirs(target_label_dir, exist_ok=True)
            label_path_new = os.path.join(target_label_dir, f'{i:05d}.txt')
            shutil.copy(os.path.join(label_dir, label_filename), label_path_new)

            f.write(img_path + '\n')


def save_data(image_train_val_paths, image_test_paths, data_path, train_val_scale=0.95):
    # 按照8:2的比例拆分训练集和测试集
    random.shuffle(image_train_val_paths)
    split_index = int(len(image_train_val_paths) * train_val_scale)
    train_images = image_train_val_paths[:split_index]
    val_images = image_train_val_paths[split_index:]
    test_images = image_test_paths

    # 按照每个文件夹的格式存入字典
    temp_list_train = {}
    for i, img_path in enumerate(train_images):
        temp = os.path.dirname(img_path)
        if temp not in temp_list_train:
            temp_list_train[temp] = 1
        else:
            temp_list_train[temp] += 1
    temp_list_val = {}
    for i, img_path in enumerate(val_images):
        temp = os.path.dirname(img_path)
        if temp not in temp_list_val:
            temp_list_val[temp] = 1
        else:
            temp_list_val[temp] += 1
    temp_list_test = {}
    for i, img_path in enumerate(test_images):
        temp = os.path.dirname(os.path.dirname(img_path))
        if temp not in temp_list_test:
            temp_list_test[temp] = [img_path]
        else:
            temp_list_test[temp].append(img_path)

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 拷贝训练集和测试集，并保存训练集和测试集的图片路径到文件
    copy_images_and_labels(train_images, train_dir, file=data_path + '/train.txt')
    copy_images_and_labels(val_images, val_dir, file=data_path + '/val.txt')
    copy_images_and_labels(test_images, test_dir, file=data_path + '/test.txt')

    # 将单个文件夹的数据拷贝到对应的文件夹
    for k in temp_list_test:
        base_name = os.path.basename(k)
        test_dir = os.path.join(data_path, base_name)
        os.makedirs(test_dir, exist_ok=True)
        copy_images_and_labels(temp_list_test[k], test_dir, file=data_path + f'/{base_name}.txt')


if __name__ == '__main__':
    base_path = '/home/csz_changsha/data/malan/carrier'
    data_path = '/home/csz_changsha/data/malan/carrier_AirTrack_dataset'

    reprod_init(128)

    labels_paths = get_labels_paths(base_path)

    # img_train_val_paths, img_test_paths = analyse_and_filter_labels(labels_paths, orig_w=1280, orig_h=720,
    #                                                                 scale=2, center_range=16,
    #                                                                 range_min=16, range_max=32,
    #                                                                 iou=True, iou_threshold=0.25)
    img_train_val_paths, img_test_paths = analyse_and_filter_labels(labels_paths, orig_w=1280, orig_h=720,
                                                                    scale=2, center_range=10,
                                                                    range_min=16, range_max=32,
                                                                    iou=False, iou_threshold=0.25)

    save_data(img_train_val_paths, img_test_paths, data_path, train_val_scale=0.95)
