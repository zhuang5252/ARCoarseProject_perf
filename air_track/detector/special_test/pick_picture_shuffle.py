# -*- coding: utf-8 -*-
# @Author    : 
# @File      : pick_picture.py
# @Created   : 2025/8/16 下午5:45
# @Desc      : 对数据集进行随机抽帧，可设置几帧抽一帧，也可指定比例(scale)，保留打乱后的前scale * len数据

import os
import shutil
import random

def pick_picture(picture_path, save_path, pick_num, scale=1.0):
    num = 0
    foldername="rgb"
    labelname="label"

    for i in os.listdir(picture_path):
        num_path = os.path.join(picture_path, i)
        rgb_path = os.path.join(num_path, foldername)
        label_path=os.path.join(num_path, labelname)
        # 使用 makedirs 创建多级目录
        os.makedirs(os.path.join(save_path, i, foldername), exist_ok=True)
        os.makedirs(os.path.join(save_path, i, labelname), exist_ok=True)

        random.seed(11)
        rgb_paths = os.listdir(rgb_path)
        random.shuffle(rgb_paths)
        # rgb_paths.sort()

        rgb_paths = rgb_paths[: int(len(rgb_paths) * scale)]

        for k in rgb_paths:
            l=k.replace('.jpg','.txt')
            if k.endswith('.jpg'):
                if num % pick_num == 0:
                    shutil.copy(os.path.join(rgb_path, k), os.path.join(save_path, i, foldername, k))
                    if os.path.exists(os.path.join(label_path, l)):
                        shutil.copy(os.path.join(label_path, l), os.path.join(save_path, i, labelname, l))
                num += 1

if __name__ == '__main__':
    picture_path="/media/linana/2C5821EF5821B88A/待标注/0917_JD"
    save_path="/media/linana/2C5821EF5821B88A/待标注/0917_JD_out"
    pick_num=1
    pick_picture(picture_path, save_path, pick_num, scale=0.37)