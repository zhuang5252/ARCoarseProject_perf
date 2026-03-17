import cv2
import numpy as np
from matplotlib import pyplot as plt
from air_track.detector.utils.detect_utils import read_flight_id_csv
from air_track.utils import common_utils


def create_image_with_bboxes(width, height, bbox_gts, save_path):
    """
    创建一个指定大小的全黑图片，并在其上画出指定的边界框。

    参数:
    width (int): 图片的宽度。
    height (int): 图片的高度。
    bbox_gts (list of tuples): 边界框列表，每个边界框为一个四元组 (left, top, right, bottom)。
    save_path (str): 图片保存的路径。

    返回:
    None
    """
    # 创建一个全黑图片
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 在图片上画出每个bbox
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(image)
    for bbox_gt in bbox_gts:
        gt_left, gt_top, gt_right, gt_bottom = bbox_gt
        rect = plt.Rectangle((gt_left, gt_top), gt_right - gt_left, gt_bottom - gt_top, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')  # 不显示坐标轴

    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def is_all_nan(lst):
    # 将列表转换为numpy数组
    array = np.array(lst)
    # 使用numpy的isnan函数检查每个元素是否为nan
    return np.isnan(array).all()


def visualize_and_save(width, height, bbox_gts, save_path):
    """检测结果可视化"""
    # 创建一个全黑图片
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for bbox in bbox_gts[0]:
        if not is_all_nan(bbox):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
    for bbox in bbox_gts[1]:
        if not is_all_nan(bbox):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
    for bbox in bbox_gts[2]:
        if not is_all_nan(bbox):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

    cv2.imwrite(save_path, image)


if __name__ == '__main__':

    """有标签预测"""
    dataset_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/config/dataset.yaml'
    train_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/detector/config/hrnet_train.yaml'
    pred_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/air_track/detector/config/predict.yaml'

    # 合并若干个yaml的配置文件内容
    yaml_list = [dataset_yaml, train_yaml, pred_yaml]
    cfg = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    part_list = ['0730', 'lz_trj_1_0300', 'lz_trj_2_0300']

    bbox_gts = []
    for part in part_list:
        flight_id = part + '_result'
        csv_file = f'/home/csz_changsha/data/AR_2410/normal_airtrack_test/{part}/ImageSets/groundtruth.csv'
        # 从 groundtruth.csv 文件中提取所需信息
        df_flight, frame_numbers, file_names = read_flight_id_csv(cfg, cfg['data_dir'], part=part, flight_id=flight_id)

        part_bbox_gts = []
        for i, (frame_num, file_name) in enumerate(zip(frame_numbers, file_names)):
            # 取出列表中标题的index行的数据
            gt_left = df_flight['gt_left'].iloc[i]
            gt_right = df_flight['gt_right'].iloc[i]
            gt_top = df_flight['gt_top'].iloc[i]
            gt_bottom = df_flight['gt_bottom'].iloc[i]

            bbox_gt = (gt_left, gt_top, gt_right, gt_bottom)
            part_bbox_gts.append(bbox_gt)

        bbox_gts.append(part_bbox_gts)

    visualize_and_save(640, 512, bbox_gts, 'show.png')

