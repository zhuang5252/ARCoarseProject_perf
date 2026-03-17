import os
from air_track.detector.special_test.analyse_yolo_data import get_labels_paths
from air_track.utils import combine_load_cfg_yaml


def get_label_count(cfg, label_folders):
    label_count = {}
    for label_folder in label_folders:
        label_names = os.listdir(label_folder)

        for i, label_name in enumerate(label_names):
            if label_name == 'classes.txt':
                continue
            # print(f"当前处理第{i}/{len(label_names)}帧")

            label_path = os.path.join(label_folder, label_name)
            # 读取标签
            with open(label_path, 'r') as file:
                labels = file.readlines()
            labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的

            # 统计每个类别的数量
            for label in labels:
                class_id = int(label[0])
                try:
                    cls = cfg['classes'][class_id]
                    if cls not in label_count:
                        label_count[cls] = 1
                    else:
                        label_count[cls] += 1
                except:
                    print('跳过：', label_path)

    return label_count


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(script_dir, 'config/dataset_dalachi.yaml')
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=[data_yaml])

    data_path = '/media/linana/2C5821EF5821B88A/yanqing_train_data/筛选数据/20250828/merge_all'
    labels_paths = get_labels_paths(data_path, label_folder='label')

    label_count = get_label_count(cfg_data, labels_paths)
    print(f"标签数量统计：{label_count}")
    print(f"标签目标总数量：{sum(label_count.values())}")
    print()
