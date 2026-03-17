import os
import cv2
import torch
import pickle
import random
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
from matplotlib import pyplot as plt
from air_track.detector.engine.predictor import Predictor
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import cxcywhn2xyxy, argmax2d
from air_track.detector.visualization.visualize_and_save import visualize_and_save, show_confidence, \
    draw_feature_img_orig_data
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size, natural_sort_key


def match_and_extract(classes, detected_objects, yolo_labels, img_width=1280, img_height=720, min_w=8, min_h=8):
    # 1. 找出所有检测框的最小w和h
    # min_w = min(obj['w'] for obj in detected_objects)
    # min_h = min(obj['h'] for obj in detected_objects)

    print(f"最小宽度: {min_w}, 最小高度: {min_h}")

    # 2. 将检测框从像素坐标转换为归一化坐标
    detections_norm = []
    for obj in detected_objects:
        detections_norm.append({
            'cx': obj['cx'] / img_width,
            'cy': obj['cy'] / img_height,
            'w': obj['w'] / img_width,
            'h': obj['h'] / img_height,
            'conf': obj['conf'],
            'cls': obj['cls']
        })

    # 3. 按类别存储结果的字典
    results = {}

    # 4. 匹配检测框和标签框
    for label in yolo_labels:
        class_id, label_cx, label_cy, label_w, label_h = label
        class_name = classes[int(class_id)]

        if class_name not in results:
            results[class_name] = []

        # 寻找与当前标签框最匹配的检测框
        best_match = None
        best_iou = 0
        best_conf = 0

        for det in detections_norm:
            iou = calculate_iou(
                label_cx, label_cy, label_w, label_h,
                det['cx'], det['cy'], det['w'], det['h']
            )
            # if iou > 0.5 and det['conf'] > best_conf:
            if iou > 0.3:
                best_iou = iou
                best_conf = det['conf']
                best_match = det

                norm_min_w = min_w / img_width
                norm_min_h = min_h / img_height

                # 初始计算边界框（可能超出边界）
                x1 = best_match['cx'] - norm_min_w / 2
                y1 = best_match['cy'] - norm_min_h / 2
                x2 = best_match['cx'] + norm_min_w / 2
                y2 = best_match['cy'] + norm_min_h / 2

                # 平移处理（保持尺寸不变）
                if x1 < 0:
                    offset = -x1
                    x1 += offset
                    x2 += offset
                elif x2 > 1:
                    offset = x2 - 1
                    x1 -= offset
                    x2 -= offset

                if y1 < 0:
                    offset = -y1
                    y1 += offset
                    y2 += offset
                elif y2 > 1:
                    offset = y2 - 1
                    y1 -= offset
                    y2 -= offset

                # 重新计算中心点（宽高保持不变）
                final_cx = (x1 + x2) / 2
                final_cy = (y1 + y2) / 2
                final_w = norm_min_w  # 保持最小宽度
                final_h = norm_min_h  # 保持最小高度

                results[class_name].append({
                    'center_x': final_cx,
                    'center_y': final_cy,
                    'width': final_w,
                    'height': final_h,
                    'confidence': det['conf'],
                    'original_iou': iou,
                })

    return results


def calculate_iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    """计算两个归一化边界框的IoU"""
    # 转换为x1,y1,x2,y2格式
    box1 = [cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2]
    box2 = [cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2]

    # 计算交集区域
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集区域
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def extract_feature_patches_by_class(feature_map, result_dict, img_width=480, img_height=270, min_w=8, min_h=8):
    """
    从特征图上提取匹配目标对应区域的小特征图，并按类别存放
    （保持最小尺寸，通过平移处理超出边界的情况）

    参数:
        feature_map: 原始特征图 [1, C, H, W]
        result_dict: 匹配结果字典 {class_id: [boxes...]}
        img_width: 原始图像宽度
        img_height: 原始图像高度

    返回:
        patches_dict: 按类别存放的小特征图字典 {class_id: [tensor1, tensor2,...]}
        patch_info_dict: 按类别存放的元信息字典 {class_id: [info1, info2,...]}
    """
    # 获取特征图尺寸
    _, C, feat_h, feat_w = feature_map.shape

    # 计算特征图与原始图像的比例
    w_ratio = feat_w / img_width
    h_ratio = feat_h / img_height

    # 找出所有框的最小宽高（以特征图像素为单位）
    # min_w = min(box['width'] * img_width * w_ratio
    #             for boxes in result_dict.values() for box in boxes)
    # min_h = min(box['height'] * img_height * h_ratio
    #             for boxes in result_dict.values() for box in boxes)

    print(f"特征图上的最小尺寸: {min_w:.1f}x{min_h:.1f} 像素")

    patches_dict = defaultdict(list)
    patch_info_dict = defaultdict(list)

    for class_id, boxes in result_dict.items():
        for box in boxes:
            # 转换为特征图上的坐标
            cx = box['center_x'] * img_width * w_ratio
            cy = box['center_y'] * img_height * h_ratio
            w = max(box['width'] * img_width * w_ratio, min_w)
            h = max(box['height'] * img_height * h_ratio, min_h)

            # 初始边界框坐标（可能超出特征图范围）
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # 平移处理（保持尺寸不变）
            if x1 < 0:
                offset = -x1
                x1 += offset
                x2 += offset
            elif x2 > feat_w:
                offset = x2 - feat_w
                x1 -= offset
                x2 -= offset

            if y1 < 0:
                offset = -y1
                y1 += offset
                y2 += offset
            elif y2 > feat_h:
                offset = y2 - feat_h
                y1 -= offset
                y2 -= offset

            # 转换为整数坐标
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if x2 <= x1 or y2 <= y1:
                continue

            # 提取特征图区域
            patch = feature_map[0, :, y1:y2, x1:x2].clone()

            # 记录调整信息
            adjusted = (x1 != int(cx - w / 2)) or (y1 != int(cy - h / 2))

            patches_dict[class_id].append(patch)
            patch_info_dict[class_id].append({
                'feature_coords': (x1, y1, x2, y2),
                'original_coords': (cx / w_ratio / img_width, cy / h_ratio / img_height,
                                    w / w_ratio / img_width, h / h_ratio / img_height),
                'adjusted': adjusted,
                'confidence': box['confidence']
            })

    return dict(patches_dict), dict(patch_info_dict)


def safe_tsne_visualization(features, labels, analysis_dir, part, perplexity=15):
    """
    安全的t-SNE可视化函数，自动调整perplexity

    参数:
        features: 特征矩阵 [N, D]
        labels: 对应标签 [N]
        perplexity: 初始尝试的perplexity值
    """
    # 动态调整perplexity
    n_samples = len(features)
    perplexity = min(perplexity, n_samples - 1)
    if perplexity < 1:
        print("警告：样本数过少，无法进行t-SNE分析")
        return

    print(f"使用perplexity={perplexity} (样本数={n_samples})")

    # 执行t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    for class_id in np.unique(labels):
        mask = labels == class_id
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                    label=f'Class {class_id}', alpha=0.6)

    plt.title(f't-SNE Visualization (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.savefig(f'{analysis_dir}/tsne_{part}.png')


# 修改prepare_tsne_data函数，添加最小样本检查
def prepare_tsne_data(patches_dict, min_samples=1):
    """准备t-SNE数据，并检查样本数量"""
    features = []
    labels = []

    for class_id, patch_list in patches_dict.items():
        if len(patch_list) < min_samples:
            print(f"警告：类别 {class_id} 只有 {len(patch_list)} 个样本，建议收集更多数据")
            # continue

        for patch in patch_list:
            flattened = patch.flatten().unsqueeze(0)
            features.append(flattened)
            labels.append(class_id)

    if not features:
        print("错误：没有足够的样本进行t-SNE分析")
        return None, None

    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels)
    return features, labels


def predict_with_gt(predictor, part, img_folder, label_folder, show_feature=False):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg

    cluster_min_size = 16
    max_cluster_samples = 120000
    classes = cfg['classes']
    down_scale = predictor.down_scale
    img_read_method = predictor.dataset_params['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']
    train_val_test_scale = predictor.dataset_params.get('train_val_test_scale', 'None')

    img_names = os.listdir(img_folder)
    random.shuffle(img_names)
    # img_names.sort(key=natural_sort_key)

    # 按比例截取数据集
    if train_val_test_scale != 'None':
        l = len(img_names)
        train_scale, val_scale, test_scale = train_val_test_scale
        img_names = img_names[int(l * (train_scale + val_scale)): int(l * (train_scale + val_scale + test_scale))]

    inference_times = []
    bbox_xyxy = []
    bbox_cxcywhn = []
    frame_nums = []
    img_paths = []
    candidate_targets = []
    scale_w, scale_h = 0, 0

    all_patches = {}
    for cls in classes:
        all_patches[cls] = []
    for i, img_name in enumerate(img_names):
        print(f"当前处理第{i}/{len(img_names)}帧")

        img_path = os.path.join(img_folder, img_name)
        label_path = os.path.join(label_folder, img_name.replace(cfg['img_format'], cfg['gt_format']))

        if not os.path.exists(img_path):
            continue

        if img_read_method == 'gray':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(img_path)

        # 图像存在才进行后续操作
        orig_h, orig_w = img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        input_img = check_and_change_img_size(img, img_size_w, img_size_h)

        # 数据前处理
        input_data = predictor.process_input(input_img)

        # 进行检测，返回检测结果和推理耗时
        if predictor.return_feature_map:
            features, prediction, detected_objects, inference_time = predictor.predict(input_data)
        else:
            prediction, detected_objects, inference_time = predictor.predict(input_data)
            features = None
        inference_times.append(inference_time)

        if show_feature:
            temp = prediction[0].float().cpu().detach().numpy()
            save_path = '/home/csz_changsha/temp/'
            os.makedirs(save_path, exist_ok=True)
            max_iloc_y, max_iloc_x = argmax2d(temp[0][0])

            draw_feature_img_orig_data(temp[0], save_path + f'{img_name}_feature.png',
                                       is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)

        frame_bbox_xyxy, frame_bbox_cxcywhn = [], []
        if cfg['calculate_metrics_flag']:
            if not os.path.exists(label_path):
                continue
            # 读取标签
            with open(label_path, 'r') as file:
                labels = file.readlines()
            labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的
            # 将标签由字符串转换为浮点数
            labels = [[float(item) for item in label] for label in labels]

            if cfg['analysis_show'] and features is not None and i < max_cluster_samples:
                features = features.cpu().detach()
                # 方法1：全局平均池化（保留空间信息）
                features_mean = features.mean(dim=1, keepdim=True)  # [1, 1, 270, 480]
                # 方法2：全局最大池化
                features_max = features.max(dim=1, keepdim=True)[0]
                # 方法3：求和
                features_sum = features.sum(dim=1, keepdim=True)
                feature = features_mean

                # 执行匹配和提取
                result_dict = match_and_extract(classes, detected_objects, labels,
                                                img_width=img_size_w, img_height=img_size_h,
                                                min_w=cluster_min_size, min_h=cluster_min_size)
                # 提取特征图块
                patches_dict, info_dict = extract_feature_patches_by_class(
                    feature, result_dict, img_width=img_size_w // down_scale, img_height=img_size_h // down_scale,
                    min_w=cluster_min_size // down_scale, min_h=cluster_min_size // down_scale)

                for k in patches_dict.keys():
                    all_patches[k] += patches_dict[k]

            # 将标签从cxcywhn转换为xyxy
            for label in labels:
                bbox_gt = cxcywhn2xyxy(label[1:], img_w=orig_w, img_h=orig_h)
                frame_bbox_xyxy.append([int(label[0])] + bbox_gt)
                frame_bbox_cxcywhn.append(label)

        frame_nums.append(i)
        img_paths.append(img_path)
        candidate_targets.append(detected_objects)
        if cfg['calculate_metrics_flag']:
            bbox_xyxy.append(frame_bbox_xyxy)
            bbox_cxcywhn.append(frame_bbox_cxcywhn)

        # 检测结果可视化
        if cfg['visualize_flag']:
            # 可视化结果保存
            visualize_and_save(img_path, detected_objects, os.path.join((cfg['visualize_save_dir']), part),
                               scale_w, scale_h, frame_bbox_xyxy)

    if '/' in part:
        part = part.split('/')[0]

    if cfg['calculate_metrics_flag']:
        save_file_dir = cfg['metrics_cache_save_file']
        cache_file = f'{save_file_dir}/cache_{part}.pkl'
        with open(cache_file, 'wb') as file:
            pickle.dump({'bbox_cxcywhn': bbox_cxcywhn, 'bbox_xyxy': bbox_xyxy, 'candidate_targets': candidate_targets,
                         'frame_nums': frame_nums, 'scale_w': scale_w, 'scale_h': scale_h, 'img_paths': img_paths},
                        file)

        detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
        pos_candidate_targets, neg_candidate_targets = \
            calculate_metrics(cfg['classes'], bbox_xyxy, candidate_targets, frame_nums, scale_w, scale_h,
                              threshold=cfg['threshold'],
                              target_min_size=cfg['target_min_size'],
                              iou=cfg['iou_flag'], cls_flag=cfg['cls_flag'])
        if cfg['analysis_show']:
            show_confidence(pos_conf_list, neg_conf_list, save_path=cfg['visualize_save_dir'],
                            file_name=f'conf_{part}.png')
            # 创建特征分析保存目录
            analysis_dir = os.path.join(cfg['visualize_save_dir'], 'feature_analysis')
            os.makedirs(analysis_dir, exist_ok=True)

            # 准备t-SNE数据
            tsne_features, tsne_labels = prepare_tsne_data(all_patches)

            if tsne_features is not None:
                # 3. 动态调整perplexity的可视化
                safe_tsne_visualization(tsne_features, tsne_labels, analysis_dir, part)

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == "__main__":
    """有标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict_sitch_tiny.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    # 创建预测器
    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']

    # 加载模型
    predictor.set_model(model_path=None)

    # 进行测试
    data_dir = cfg_predict['data_dir']
    # for part in cfg_predict['part_val']:
    for part in cfg_predict['part_test']:
        print('当前测试数据集 part：', part)
        data_path = os.path.join(data_dir, part, cfg_predict['img_folder'])
        label_path = os.path.join(data_dir, part, cfg_predict['label_folder'])

        predict_with_gt(predictor, part, data_path, label_path, show_feature=False)
