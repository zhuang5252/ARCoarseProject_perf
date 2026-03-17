import os
import re
import shutil
import numpy as np
import pandas as pd
from air_track.detector.utils.detect_utils import xyxy2cxcywhn
from air_track.utils import get_all_files


cls = ['1.0']


def save_txt(datas: list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 将字典中的每个键值对写入不同的txt文件中
    # 写入文件
    with open(save_path, 'w') as file:
        for data in datas:
            # 将列表转换为字符串，并用换行符分隔
            line = ' '.join(map(str, data)) + '\n'
            file.write(line)


if __name__ == '__main__':
    orig_path = '/home/csz_changsha/data/FlySubjectAirToGround'
    save_path = '/home/csz_changsha/data/FlySubjectAirToGround/AR_2410/sah_trj_2_1330'
    parts_train = ['sah_trj_2_1330']
    parts_val = []

    save_train_img_path = os.path.join(save_path, 'images/train')
    save_val_img_path = os.path.join(save_path, 'images/val')
    save_test_img_path = os.path.join(save_path, 'images/test')
    save_train_label_path = os.path.join(save_path, 'labels/train')
    save_val_label_path = os.path.join(save_path, 'labels/val')
    save_test_label_path = os.path.join(save_path, 'labels/test')
    os.makedirs(save_train_img_path, exist_ok=True)
    os.makedirs(save_val_img_path, exist_ok=True)
    os.makedirs(save_test_img_path, exist_ok=True)
    os.makedirs(save_train_label_path, exist_ok=True)
    os.makedirs(save_val_label_path, exist_ok=True)
    os.makedirs(save_test_label_path, exist_ok=True)

    save_data = {}
    cls_names_tmp = []
    rx = re.compile(r'(\D+)(\d+)')  # 这个正则表达式模式用于匹配由非数字字符开头，后面跟着一个或多个数字字符的字符串序列

    for part in parts_train + parts_val:
        tmp_path = os.path.join(orig_path, part)
        csv_files = get_all_files(tmp_path, 2, file_end='.csv')

        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            flight_ids = data.flight_id.values
            img_names = data.img_name.values
            cls_names = data.id.values
            gt_lefts = data.gt_left.values
            gt_tops = data.gt_top.values
            gt_rights = data.gt_right.values
            gt_bottoms = data.gt_bottom.values
            size_widths = data.size_width.values
            size_heights = data.size_height.values

            img_paths = [os.path.join(orig_path, part, 'Images',  x, y) for x, y in zip(flight_ids, img_names)]
            for i, img_path in enumerate(img_paths):

                cls_name = str(cls_names[i])
                # if not (not isinstance(cls_name, str) and np.isnan(cls_name)):  # 不是字符串类型且且为 NaN（无目标的空图像）
                #     m = rx.search(cls_name)  # 使用正则表达式对象 rx 来搜索匹配 item_id 的字符串
                #     cls_name = m.group(1)  # 获取非数字字符序列
                #     item_id_num = int(m.group(2))  # 获取数字字符序列并转为int型
                #
                #     if cls_name == 'BIrd':
                #         cls_name = 'Bird'

                if cls_name not in cls_names_tmp:
                    cls_names_tmp.append(cls_name)
                w = size_widths[-6]
                h = size_heights[-5]
                gt_left = gt_lefts[i]
                gt_top = gt_tops[i]
                gt_right = gt_rights[i]
                gt_bottom = gt_bottoms[i]

                bbox = [gt_left, gt_top, gt_right, gt_bottom]

                if not np.isnan(gt_left):
                    cls_id = cls.index(cls_name)
                    bbox = xyxy2cxcywhn(bbox, w=w, h=h)
                    tmp = [cls_id] + bbox
                    if img_path in save_data:
                        save_data[img_path].append([tmp])
                    save_data[img_path] = [tmp]
                else:
                    save_data[img_path] = []

    # save_txt(save_data, os.path.join(save_path, 'txt'))
    l = len(save_data)
    train_len = [0, int(l * 0.7)]
    val_len = [train_len[1], train_len[1] + int(l * 0.15)]
    test_len = [val_len[1], l]

    # 获取字典的keys并打乱顺序
    # save_data_keys_copy = copy.deepcopy(list(save_data.keys()))
    shuffled_keys = list(save_data.keys())
    # random.shuffle(shuffled_keys)

    for i, key in enumerate(shuffled_keys):
        value_lists = save_data[key]
        tmp_part = key.split('/')[-4]
        base_name = os.path.basename(key)
        save_label_file = base_name.replace('.png', '.txt')
        # if train_len[0] <= i < train_len[1]:
        # if tmp_part in ['part1', 'part3']:
        #     tmp_img_path = os.path.join(save_train_img_path, base_name)
        #     tmp_label_path = os.path.join(save_train_label_path, save_label_file)
        # else:
        #     tmp_img_path = os.path.join(save_val_img_path, base_name)
        #     tmp_label_path = os.path.join(save_val_label_path, save_label_file)
        # elif val_len[0] <= i < val_len[1]:
        #     tmp_img_path = os.path.join(save_val_img_path, base_name)
        #     tmp_label_path = os.path.join(save_val_label_path, save_label_file)
        # else:
        #     tmp_img_path = os.path.join(save_test_img_path, base_name)
        #     tmp_label_path = os.path.join(save_test_label_path, save_label_file)
        tmp_img_path = os.path.join(save_test_img_path, base_name)
        tmp_label_path = os.path.join(save_test_label_path, save_label_file)

        if os.path.exists(key):
            shutil.copy(key, tmp_img_path)
            save_txt(value_lists, tmp_label_path)

    print()
