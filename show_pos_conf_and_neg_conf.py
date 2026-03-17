import os
import cv2
import pickle
from air_track.detector.utils.calculate_metrics import calculate_metrics, is_all_nan
from air_track.detector.visualization.visualize_and_save import show_confidence, show_confidence_sorted
from air_track.utils import check_and_change_img_size


def detected_objs_2_bbox(data, scale_w, scale_h):
    conf = float(data['conf'])
    # cx = (data['cx'] + data['offset'][0]) / scale_w
    # cy = (data['cy'] + data['offset'][1]) / scale_h
    # w = data['w'] / scale_w
    # h = data['h'] / scale_h
    cx = data['cx']
    cy = data['cy']
    w = data['w']
    h = data['h']
    # 转换为左上角的坐标xy，右下角的坐标xy
    bbox = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2 + 0.5), int(cy + h / 2 + 0.5)]

    return conf, bbox


def visualize_and_save(cur_file, visualize_save_dir,
                       pos_candidate_targets, neg_candidate_targets,
                       scale_w=1, scale_h=1, bbox_gt=None, distance=None,
                       ):
    """检测结果可视化"""
    image = cv2.imread(cur_file)
    image = check_and_change_img_size(image, 640, 512)

    save_file_name = os.path.basename(cur_file)
    os.makedirs(visualize_save_dir, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if bbox_gt:
        if not is_all_nan(bbox_gt):
            cv2.rectangle(image, (int(bbox_gt[0]), int(bbox_gt[1])), (int(bbox_gt[2]), int(bbox_gt[3])), (0, 255, 0), 1)
    if distance:
        cv2.putText(image, f'distance: {distance}', (15, 25), font, 1, (0, 255, 0), 1)

    for pos_candidate_target in pos_candidate_targets:
        if pos_candidate_target:
            conf, bbox = detected_objs_2_bbox(pos_candidate_target, scale_w, scale_h)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
            cv2.putText(image, f'{round(conf, 4)}', (bbox[0], bbox[1]), font, 1, (0, 255, 255), 1)
    for neg_candidate_target in neg_candidate_targets:
        if neg_candidate_target:
            conf, bbox = detected_objs_2_bbox(neg_candidate_target, scale_w, scale_h)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            cv2.putText(image, f'{round(conf, 4)}', (bbox[0], bbox[1]), font, 1, (0, 0, 255), 1)

    cv2.imwrite(f'{visualize_save_dir}/{save_file_name}', image)


if __name__ == '__main__':
    """可视化正样本置信度 > 负样本置信度、正样本置信度 < 负样本置信度、漏检的检测结果"""

    cache_file = '/home/csz_changsha/PycharmProjects/pythonProject/AirTrack_aot_local/output/cache_metrics.pkl'
    save_path = '/home/csz_changsha/PycharmProjects/pythonProject/AirTrack_aot_local/output'

    # 使用 'rb' 模式打开文件进行读取（'rb' 代表 'read binary'）
    with open(cache_file, 'rb') as file:
        # 使用 pickle.load 来读取文件中的数据
        data = pickle.load(file)

    cur_files = data['cur_files']
    bbox_gts = data['bbox_gts']
    candidate_targets = data['candidate_targets']
    frame_nums = data['frame_nums']
    scale_w = data['scale_w']
    scale_h = data['scale_h']
    detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
    pos_candidate_targets, neg_candidate_targets = \
        calculate_metrics(bbox_gts, candidate_targets, frame_nums, scale_w=1, scale_h=1,
                                        threshold=0.25, target_min_size=10, iou=True)

    for i, (cur_file, bbox_gt, pos_candidate_target, neg_candidate_target) in enumerate(
            zip(cur_files, bbox_gts, pos_candidate_targets, neg_candidate_targets)):
        print(f"当前处理第{i + 1}/{len(cur_files)}帧")
        if pos_candidate_target and neg_candidate_target:
            conf_pos = max([conf['conf'] for conf in pos_candidate_target])
            conf_neg = max([conf['conf'] for conf in neg_candidate_target])
            if conf_pos < conf_neg:
                visualize_save_dir = os.path.join(save_path, 'pos_conf_less_than_neg_conf')
                visualize_and_save(cur_file, visualize_save_dir,
                                   pos_candidate_target, neg_candidate_target,
                                   scale_w=scale_w, scale_h=scale_h, bbox_gt=bbox_gt)
            else:
                visualize_save_dir = os.path.join(save_path, 'pos_conf_more_than_neg_conf')
                visualize_and_save(cur_file, visualize_save_dir,
                                   pos_candidate_target, neg_candidate_target,
                                   scale_w=scale_w, scale_h=scale_h, bbox_gt=bbox_gt)
        if not pos_candidate_target:
            visualize_save_dir = os.path.join(save_path, 'without_pos_conf')
            visualize_and_save(cur_file, visualize_save_dir,
                               pos_candidate_target, neg_candidate_target,
                               scale_w=scale_w, scale_h=scale_h, bbox_gt=bbox_gt)

    show_confidence(pos_conf_list, neg_conf_list, save_path, file_name='conf.png')

    show_confidence_sorted(pos_conf_list, neg_conf_list,
                           save_path=save_path,
                           file_name='conf_sorted.png')
