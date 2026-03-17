import os
import random
import shutil
from air_track.utils import reprod_init


reprod_init(128)

# 假设这些文件夹存储在一个列表中
discovery_paths = '/home/csz_changsha/data/0104/cc/18-txt'
folders = os.listdir(discovery_paths)
data_path = '/home/csz_changsha/data/0104/cc/AirTrack_dataset'
train_dir = os.path.join(data_path, 'train')
test_dir = os.path.join(data_path, 'test')

# 准备存储图片路径的列表
all_images = []

# 遍历所有文件夹，读取discovery.txt文件中的图片路径
for folder in folders:
    discovery_path = os.path.join(discovery_paths, folder, 'discovery.txt')
    if os.path.exists(discovery_path):
        with open(discovery_path, 'r') as file:
            all_images.extend([line.strip() for line in file.readlines()])

# 按照8:2的比例拆分训练集和测试集
random.shuffle(all_images)
split_index = int(len(all_images) * 0.8)
train_images = all_images[:split_index]
test_images = all_images[split_index:]


# 准备拷贝图片和标签到对应的目录
def copy_images_and_labels(image_paths, target_dir, file):
    with open(file, 'w') as f:
        for i, img_path in enumerate(image_paths):
            print(f'当前处理第{i + 1}帧/{len(image_paths)}帧')
            if 'VOC070_BUS' in img_path:
                continue
            # 确定图片和标签的目录与文件名
            img_path = img_path.replace('/home/lrf/1225', '/home/csz_changsha/data/0104/cc')
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

            f.write(img_path_new + '\n')


# 拷贝训练集和测试集，并保存训练集和测试集的图片路径到文件
copy_images_and_labels(train_images, train_dir, file=data_path + '/train.txt')
copy_images_and_labels(test_images, test_dir, file=data_path + '/test.txt')
