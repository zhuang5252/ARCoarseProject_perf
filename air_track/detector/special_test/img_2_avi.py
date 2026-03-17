import cv2
import os
import numpy as np

# 指定图片文件夹路径和输出视频路径
image_folder = '/home/csz_changsha/data/Aot_small_640_512/part1/Images/0001ba865c8e410e88609541b8f55ffc'
video_name = '/home/csz_changsha/data/Aot_small_640_512/part1/Images/0001ba865c8e410e88609541b8f55ffc.avi'

# 获取图片列表并排序
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort()  # 确保图片按顺序排序

# 获取第一张图片来确定视频的尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 定义视频编码器和创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

# 遍历图片并写入视频
for i, image in enumerate(images):
    if i < 223:
        continue
    img = cv2.imread(os.path.join(image_folder, image))
    video.write(img)

# 释放视频写入对象
video.release()

print(f"视频已保存到 {video_name}")
