# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - rm_second_data.py
# @Created   : 2025/09/08 22:26
# @Desc      : 按照指定数量删除二级分类器数据

"""
按照指定数量删除二级分类器数据
"""
import os
import random
import shutil
import pandas as pd
from air_track.utils import extract_number


# 待删除负样本数量
rm_part_data = {
    'JDModel_normal_0827_M300_background_1_merge_second': 0,
    'JDModel_normal_0906_dlc_filter_no_benchang_second': 10109,
    'JDModel_normal_0907_dlk_filter_merge_second': 924,
    'JDModel_paste_0823_0827_M300_background_1_merge_second': 19582,
    'JDModel_paste_0823_0906_dlc_filter_no_benchang_second': 114876,
    'JDModel_paste_0823_0907_dlk_filter_merge_second': 42406
}


if __name__ == '__main__':
    for part in rm_part_data:
        npy_format = '.npy'
        npy_folder = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0908_second_classifier_dataset/{part}/npy'
        img_folder = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0908_second_classifier_dataset/{part}/img'
        label_file = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0908_second_classifier_dataset/{part}/label.csv'

        npy_paths = [f for f in os.listdir(npy_folder) if f.endswith(npy_format)]
        npy_paths.sort(key=extract_number)
        npy_paths = [os.path.join(npy_folder, f) for f in npy_paths]
        labels = pd.read_csv(label_file).values

        random.seed(128)
        random.shuffle(npy_paths)

        count = 0
        for npy_path in npy_paths:
            num = int(os.path.basename(npy_path).split('.')[0])
            img_path = os.path.join(img_folder, f'{num}.jpg')
            item = labels[num - 1]
            if count >= rm_part_data[part]:
                break
            if int(item[0]) == num and int(item[2]) == 1:
                npy_move_path = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0908_second_classifier_dataset/负样本/{part}/npy'
                img_move_path = f'/media/linana/2C5821EF5821B88A/yanqing_train_data/0908_second_classifier_dataset/负样本/{part}/img'
                os.makedirs(npy_move_path, exist_ok=True)
                os.makedirs(img_move_path, exist_ok=True)
                shutil.move(npy_path, f'{npy_move_path}/{num}.npy')
                shutil.move(img_path, f'{img_move_path}/{num}.jpg')
                count += 1

