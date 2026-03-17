import copy
import pickle
import numpy as np
from air_track.detector.utils import detect_utils


def is_all_nan(lst):
    # 将列表转换为numpy数组
    array = np.array(lst)
    # 使用numpy的isnan函数检查每个元素是否为nan
    return np.isnan(array).all()


def calc_iou(i1, i2):
    x1 = i1['cx']
    y1 = i1['cy']
    w1 = i1['w']
    h1 = i1['h']

    x2 = i2['cx']
    y2 = i2['cy']
    w2 = i2['w']
    h2 = i2['h']

    ix_min = max(x1 - w1 / 2, x2 - w2 / 2)
    iy_min = max(y1 - h1 / 2, y2 - h2 / 2)
    ix_max = min(x1 + w1 / 2, x2 + w2 / 2)
    iy_max = min(y1 + h1 / 2, y2 + h2 / 2)

    iw = max(ix_max - ix_min, 0.)
    ih = max(iy_max - iy_min, 0.)

    intersections = iw * ih
    unions = (i1['w'] * i1['h'] + i2['w'] * i2['h'] - intersections)

    iou = intersections / unions
    return iou


def calculate_metrics(classes, bbox_gts, candidate_targets, frame_nums, scale_w_list, scale_h_list,
                      threshold=0.25, target_min_size=2, iou=False, cls_flag=False):
    if len(bbox_gts) != len(candidate_targets):
        assert 'len(bbox_gts) != len(candidate_targets).'

    # 总标签数量
    total_gts = 0
    total_frames = len(bbox_gts)
    # 总检测数量
    total_detections = sum(len(ct) for ct in candidate_targets)
    # tp数量
    true_positives = 0

    # 保存真目标的distance值
    pos_pred_x, distance_pred_y = [], []
    pos_conf_list, neg_conf_list = [], []
    pos_candidate_targets, neg_candidate_targets = [], []
    for idx, gts, candidates, scale_w, scale_h in zip(frame_nums, bbox_gts,
                                                      candidate_targets, scale_w_list, scale_h_list):
        candidates_orig = copy.deepcopy(candidates)
        if not gts:  # 如果这个标签为空，跳过
            continue

        for gt_item in gts:
            if not np.isnan(gt_item[1:]).all():  # 如果gt_item[1:]中所有元素都不是nan，则将其添加到total_gts中
                total_gts += 1
        # total_gts += len(gts)

        pos_conf, neg_conf = [], []
        pos_candidate_target, neg_candidate_target = [], []
        for gt_box in gts:
            gt = detect_utils.xyxy2xywh(gt_box[1:])
            gt_cx, gt_cy, gt_w, gt_h = gt
            gt_w = max(target_min_size, gt_w)
            gt_h = max(target_min_size, gt_h)
            target_gt = {'cx': gt_cx, 'cy': gt_cy, 'w': gt_w, 'h': gt_h}

            for i, candidate in enumerate(candidates):
                candidate = copy.deepcopy(candidate)
                if 'offset' in candidate:
                    candidate['cx'] = (candidate['cx'] + candidate['offset'][0]) / scale_w
                    candidate['cy'] = (candidate['cy'] + candidate['offset'][1]) / scale_h
                else:
                    candidate['cx'] = candidate['cx'] / scale_w
                    candidate['cy'] = candidate['cy'] / scale_h
                candidate['w'] = candidate['w'] / scale_w
                candidate['h'] = candidate['h'] / scale_h

                if cls_flag and candidate['cls'] != classes[int(gt_box[0])]:
                    continue

                if iou:
                    if calc_iou(target_gt, candidate) >= threshold:
                        true_positives += 1
                        pos_pred_x.append(idx)
                        if 'distance' in candidate:
                            distance_pred_y.append(candidate['distance'])
                        pos_conf.append(candidate['conf'])
                        pos_candidate_target.append(candidate)
                        candidates.pop(i)  # 移除已匹配的候选框
                        break  # 只有一个与当前gt相匹配的目标
                else:
                    if ((candidate['cx'] - gt_cx) ** 2 + (candidate['cy'] - gt_cy) ** 2) ** 0.5 <= threshold:
                        true_positives += 1
                        pos_pred_x.append(idx)
                        if 'distance' in candidate:
                            distance_pred_y.append(candidate['distance'])
                        pos_conf.append(candidate['conf'])
                        pos_candidate_target.append(candidate)
                        candidates.pop(i)  # 移除已匹配的候选框
                        break  # 只有一个与当前gt相匹配的目标

        # 将未匹配的候选框添加到neg_conf中
        for candidate in candidates:
            neg_conf.append(candidate['conf'])
            neg_candidate_target.append(candidate)

        pos_conf_list.append(pos_conf)
        neg_conf_list.append(neg_conf)
        pos_candidate_targets.append(pos_candidate_target)
        neg_candidate_targets.append(neg_candidate_target)

    # 虚警数量
    false_detections = total_detections - true_positives
    miss_detections = total_gts - true_positives

    # 检出率
    detection_rate = true_positives / total_gts if total_gts else 0
    # 漏检率
    miss_rate = 1 - detection_rate
    # miss_rate = miss_detections / total_gts if total_gts else 0

    # 准确率
    precision_rate = true_positives / total_detections if total_detections else 0
    # 虚警率
    false_alarm_rate = 1 - precision_rate
    # false_rate = false_detections / total_detections if total_detections else 0

    print(f'总gt数：{total_gts}, 总帧数：{total_frames},\n'
          f'模型总检出数量：{total_detections}, 真阳性目标：{true_positives},\n'
          f'漏检数量：{miss_detections}, 虚警数量：{false_detections},\n'
          f'召回率:{detection_rate}, 准确率:{precision_rate},\n'
          f'漏检率：{miss_rate}, 虚警率：{false_alarm_rate}'
          )

    return detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
           pos_candidate_targets, neg_candidate_targets


if __name__ == '__main__':
    path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/output/cache_lz_trj_1_0300_lz_trj_1_0300_result.pkl'

    classes = [
        'other',
        'uav'
    ]

    with open(path, 'rb') as file:
        data = pickle.load(file)

    bbox_gts, candidate_targets = data['bbox_xyxy'], data['candidate_targets']
    frame_nums, scale_w_list, scale_h_list, img_paths = data['frame_nums'], data['scale_w_list'], data['scale_h_list'], data['img_paths']

    detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
    pos_candidate_targets, neg_candidate_targets = calculate_metrics(classes, bbox_gts, candidate_targets,
                                                                     frame_nums, scale_w_list, scale_h_list,
                                                                     threshold=0.25,
                                                                     target_min_size=10,
                                                                     iou=True)
