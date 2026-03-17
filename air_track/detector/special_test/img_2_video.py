import cv2
import os

# 设置要读取图片的文件夹路径
image_folder = '/media/csz_changsha/2C5821EF5821B88A/AR_DeepSort_torch/output'
video_path = '/media/csz_changsha/2C5821EF5821B88A/AR_DeepSort_torch'

# 读取文件夹下所有图片的文件名
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.bmp'))]

# 按文件名排序，确保视频帧的顺序是正确的
images.sort(key=lambda x: int(x.split('.')[0]))

# 图片路径列表
img_paths = []  # 你的图片路径列表
for i in images:
    item = os.path.join(image_folder, i)
    img_paths.append(item)

os.makedirs(video_path, exist_ok=True)

# 视频文件名和帧率
video_name = os.path.join(video_path, 'output_video.mp4')
fps = 20  # 设置帧率，这里设置为 20 帧每秒

# 获取第一张图片的尺寸
img = cv2.imread(img_paths[0])
height, width, layers = img.shape

# 设置视频的编解码方式和帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 逐个将图片写入视频
for file in img_paths:
    video.write(cv2.imread(file))

# 释放资源
cv2.destroyAllWindows()
video.release()
