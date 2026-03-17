# -*- coding: utf-8 -*-
# @Author    : 
# @File      : pick_picture.py
# @Created   : 2025/8/16 下午5:45
# @Desc      : 对数据集进行抽帧，可设置几帧抽一帧

import os
import shutil

def pick_picture(picture_path, save_path, pick_num):
    num = 0
    folder_name="rgb"
    label_name="label"
    for i in os.listdir(picture_path):
        num_path = os.path.join(picture_path, i)
        rgb_path = os.path.join(num_path, folder_name)
        label_path=os.path.join(num_path, label_name)
        # 使用 makedirs 创建多级目录
        os.makedirs(os.path.join(save_path, i, folder_name), exist_ok=True)
        os.makedirs(os.path.join(save_path, i, label_name), exist_ok=True)

        rgb_paths = os.listdir(rgb_path)
        # random.shuffle(rgb_paths)
        rgb_paths.sort()

        for k in rgb_paths:
            l=k.replace('.jpg','.txt')
            if k.endswith('.jpg'):
                if num % pick_num == 0:
                    shutil.copy(os.path.join(rgb_path, k), os.path.join(save_path, i, folder_name, k))
                    if os.path.exists(os.path.join(label_path, l)):
                        shutil.copy(os.path.join(label_path, l), os.path.join(save_path, i, label_name, l))
                num += 1

if __name__ == '__main__':
    picture_path="/media/linana/2C5821EF5821B88A/待标注/0917_JD"
    save_path="/media/linana/2C5821EF5821B88A/待标注/0917_JD_out"
    pick_num=10
    pick_picture(picture_path, save_path, pick_num)