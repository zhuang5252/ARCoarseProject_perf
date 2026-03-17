import os
import cv2
import torch
import pickle
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from air_track.detector.engine.predictor import Predictor
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.utils.detect_utils import cxcywhn2xyxy, argmax2d, check_boundary_norm, calc_iou_single_img, \
    check_boundary
from air_track.detector.visualization.visualize_and_save import (
    visualize_and_save, show_confidence, draw_feature_img_orig_data
)
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size, natural_sort_key


def match_and_extract(classes: List[str], detected_objects: List[dict], yolo_labels: List[list],
                      img_width: int = 960, img_height: int = 540,
                      min_w: int = 16, min_h: int = 16, threshold: float = 0.25) -> Dict[str, List[dict]]:
    """
    匹配检测框和标签框，并提取调整后的边界框信息

    参数:
        classes: 类别名称列表
        detected_objects: 检测结果列表，每个元素包含cx,cy,w,h,conf,cls
        yolo_labels: YOLO格式标签列表，每个元素为[class_id, cx, cy, w, h]
        img_width: 图像宽度
        img_height: 图像高度
        min_w: 最小宽度
        min_h: 最小高度

    返回:
        按类别组织的边界框信息字典
    """
    # print(f"最小宽度: {min_w}, 最小高度: {min_h}")

    # 归一化最小尺寸
    norm_min_w = min_w / img_width
    norm_min_h = min_h / img_height

    # 归一化检测框
    detections_norm = []
    for obj in detected_objects:
        detections_norm.append({
            'cxn': obj['cx'] / img_width,
            'cyn': obj['cy'] / img_height,
            'wn': obj['w'] / img_width,
            'hn': obj['h'] / img_height,
            'conf': obj['conf'],
            'cls': obj['cls']
        })

    # 按类别存储结果的字典
    results = defaultdict(list)

    # 匹配检测框和标签框
    for label in yolo_labels:
        class_id, label_cx, label_cy, label_w, label_h = label
        class_name = classes[int(class_id)]

        for det in detections_norm:
            bbox_1 = {'cx': det['cxn'], 'cy': det['cyn'], 'w': det['wn'], 'h': det['hn']}
            bbox_2 = {'cx': label_cx, 'cy': label_cy, 'w': label_w, 'h': label_h}
            iou = calc_iou_single_img(bbox_1, bbox_2)

            if iou > threshold:
                # 调整边界框
                x1 = det['cxn'] - norm_min_w / 2
                y1 = det['cyn'] - norm_min_h / 2
                x2 = det['cxn'] + norm_min_w / 2
                y2 = det['cyn'] + norm_min_h / 2

                bbox_norm_xyxy = (x1, y1, x2, y2)
                bbox_norm_xyxy_new = check_boundary_norm(bbox_norm_xyxy)
                x1, y1, x2, y2 = bbox_norm_xyxy_new

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

    return dict(results)


def extract_feature_patches_by_class(feature_map: torch.Tensor, result_dict: Dict[str, List[dict]],
                                     img_width: int = 480, img_height: int = 270,
                                     min_w: int = 8, min_h: int = 8) -> Tuple[Dict, Dict]:
    """
    从特征图上提取匹配目标对应区域的小特征图，并按类别存放

    参数:
        feature_map: 原始特征图 [1, C, H, W]
        result_dict: 匹配结果字典 {class_id: [boxes...]}
        img_width: 原始图像宽度
        img_height: 原始图像高度
        min_w: 最小宽度（像素）
        min_h: 最小高度（像素）

    返回:
        patches_dict: 按类别存放的小特征图字典 {class_id: [tensor1, tensor2,...]}
        patch_info_dict: 按类别存放的元信息字典 {class_id: [info1, info2,...]}
    """
    # 获取特征图尺寸
    _, C, feat_h, feat_w = feature_map.shape

    # 计算特征图与原始图像的比例
    w_ratio = feat_w / img_width
    h_ratio = feat_h / img_height

    # print(f"特征图上的最小尺寸: {min_w:.1f}x{min_h:.1f} 像素")

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

            bbox_xyxy = (x1, y1, x2, y2)
            bbox_xyxy_new = check_boundary(bbox_xyxy, min_w, min_h, img_width, img_height)
            x1, y1, x2, y2 = bbox_xyxy_new

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


def safe_tsne_visualization(features: np.ndarray, labels: np.ndarray,
                            save_dir: str, file_prefix: str,
                            perplexity: int = 15) -> bool:
    """
    安全的t-SNE可视化，自动调整perplexity

    参数:
        features: 特征矩阵 [N, D]
        labels: 标签数组 [N]
        save_dir: 保存目录
        file_prefix: 文件名前缀
        perplexity: t-SNE参数

    返回:
        是否成功执行
    """
    if len(features) < 1:
        print("警告：样本数过少，无法进行t-SNE分析")
        return False

    # 动态调整perplexity
    n_samples = len(features)
    perplexity = min(perplexity, n_samples - 1)
    if perplexity < 1:
        print("警告：perplexity值过小")
        return False

    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        for class_id in np.unique(labels):
            mask = labels == class_id
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                        label=f'Class {class_id}', alpha=0.6)

        plt.title(f't-SNE Visualization (perplexity={perplexity})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/tsne_{file_prefix}_{n_samples}.png')
        plt.close()
        return True
    except Exception as e:
        print(f"t-SNE可视化失败: {str(e)}")
        return False


def prepare_tsne_data(patches_dict: Dict[str, List[torch.Tensor]],
                      min_samples: int = 1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    准备t-SNE分析数据

    参数:
        patches_dict: 特征图块字典
        min_samples: 最小样本数要求

    返回:
        (features, labels) 或 (None, None)
    """
    features = []
    labels = []

    for class_id, patch_list in patches_dict.items():
        if len(patch_list) < min_samples:
            print(f"警告：类别 {class_id} 只有 {len(patch_list)} 个样本")
            # continue

        for patch in patch_list:
            flattened = patch.flatten().unsqueeze(0)
            features.append(flattened)
            labels.append(class_id)

    if not features:
        print("没有足够的样本进行t-SNE分析，跳过T-SNE可视化分析")
        return None, None

    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels)

    return features, labels


def process_single_image(predictor: Predictor, img_path: str, label_path: str,
                         cfg: dict, show_feature: bool = False) -> Optional[dict]:
    """
    处理单张图像

    返回包含处理结果的字典，如果处理失败返回None
    """
    if not os.path.exists(img_path):
        return None

    try:
        img_read_method = predictor.dataset_params['img_read_method'].lower()
        img_size_w, img_size_h = predictor.dataset_params['img_size']
        # 读取图像
        if img_read_method == 'gray':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_path}")
            return None

        # 调整图像尺寸
        orig_h, orig_w = img.shape[:2]
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        input_img = check_and_change_img_size(img, img_size_w, img_size_h)

        # 数据预处理
        input_data = predictor.process_input(input_img)

        # 执行预测
        if predictor.return_feature_map:
            features, prediction, detected_objects, inference_time = predictor.predict(input_data)
        else:
            prediction, detected_objects, inference_time = predictor.predict(input_data)
            features = None

        # 特征可视化
        if show_feature:
            if isinstance(prediction[0], np.ndarray):
                temp = prediction[0].astype(np.float32)
            else:
                temp = prediction[0].float().cpu().detach().numpy()
            save_path = '/home/csz_changsha/temp/'
            os.makedirs(save_path, exist_ok=True)
            max_iloc_y, max_iloc_x = argmax2d(temp[0][0])
            draw_feature_img_orig_data(
                temp[0], f'{save_path}{os.path.basename(img_path)}_feature.png',
                is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)

        # 读取标签
        frame_bbox_xyxy, frame_bbox_cxcywhn = [], []
        if cfg['calculate_metrics_flag'] and os.path.exists(label_path):
            with open(label_path, 'r') as file:
                labels = [list(map(float, line.strip().split())) for line in file if line.strip()]

            for label in labels:
                bbox_gt = cxcywhn2xyxy(label[1:], img_w=orig_w, img_h=orig_h)
                frame_bbox_xyxy.append([int(label[0])] + bbox_gt)
                frame_bbox_cxcywhn.append(label)

        return {
            'features': features,
            'detected_objects': detected_objects,
            'inference_time': inference_time,
            'bbox_xyxy': frame_bbox_xyxy,
            'bbox_cxcywhn': frame_bbox_cxcywhn,
            'labels': labels if 'labels' in locals() else None,
            'scale_w': scale_w,
            'scale_h': scale_h,
            'img_path': img_path
        }

    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        return None


def predict_with_gt(predictor: Predictor, part: str, img_folder: str,
                    label_folder: str, show_feature: bool = False,
                    cluster_min_size: int = 16, max_cluster_samples: int = 30000) -> None:
    """
    主预测函数，处理整个文件夹的图像
    """
    cfg = predictor.cfg
    classes = cfg['classes']
    down_scale = predictor.down_scale
    cluster_min_size = cfg.get('cluster_min_size', cluster_min_size if cluster_min_size else 16)  # 聚类特征图块尺寸
    max_cluster_samples = cfg.get('max_cluster_samples', max_cluster_samples if max_cluster_samples else 30000)  # 聚类分析的最大样本数
    img_size_w, img_size_h = predictor.dataset_params['img_size']
    train_val_test_scale = predictor.dataset_params.get('train_val_test_scale', 'None')

    # 获取图像列表
    img_names = [f for f in os.listdir(img_folder) if f.endswith(cfg['img_format'])]
    random.shuffle(img_names)
    # img_names.sort(key=natural_sort_key)

    # 按数据集比例截取
    if train_val_test_scale != 'None':
        l = len(img_names)
        train_scale, val_scale, test_scale = train_val_test_scale
        img_names = img_names[int(l * (train_scale + val_scale)): int(l * (train_scale + val_scale + test_scale))]

    # 初始化结果容器
    results = {
        'inference_times': [],
        'frame_nums': [],
        'bbox_xyxy': [],
        'bbox_cxcywhn': [],
        'candidate_targets': [],
        'all_patches': defaultdict(list),
        'scale_w': 1,
        'scale_h': 1,
        'img_paths': []
    }

    # 处理每张图像
    for i, img_name in enumerate(img_names):
        if (i + 1) % 10 == 0 or i + 1 == len(img_names):
            print(f"处理进度: {i + 1}/{len(img_names)}", end='\r', flush=True)
            print(f"处理进度: {i + 1}/{len(img_names)}", flush=True)

        img_path = os.path.join(img_folder, img_name)
        label_path = os.path.join(label_folder, img_name.replace(cfg['img_format'], cfg['gt_format']))

        # 处理单张图像
        result = process_single_image(predictor, img_path, label_path, cfg, show_feature)
        if result is None:
            continue

        # 收集结果
        results['frame_nums'].append(i)
        results['scale_w'] = result['scale_w']
        results['scale_h'] = result['scale_h']
        results['img_paths'].append(img_path)
        results['inference_times'].append(result['inference_time'])
        results['candidate_targets'].append(result['detected_objects'])

        # 用于计算指标和特征分析
        if cfg['calculate_metrics_flag'] and result['labels']:
            results['bbox_xyxy'].append(result['bbox_xyxy'])
            results['bbox_cxcywhn'].append(result['bbox_cxcywhn'])

            # 特征提取和分析
            if cfg['analysis_show'] and result['features'] is not None and i < max_cluster_samples:
                features = result['features'].cpu().detach()
                feature = features.mean(dim=1, keepdim=True)  # 使用平均池化

                # 匹配和提取特征
                result_dict = match_and_extract(
                    classes, result['detected_objects'], result['labels'],
                    img_width=img_size_w, img_height=img_size_h,
                    min_w=cluster_min_size, min_h=cluster_min_size, threshold=cfg['threshold'])

                # 按照类别提取特征
                patches_dict, info_dict = extract_feature_patches_by_class(
                    feature, result_dict,
                    img_width=img_size_w // down_scale,
                    img_height=img_size_h // down_scale,
                    min_w=cluster_min_size // down_scale,
                    min_h=cluster_min_size // down_scale)

                for cls, patches in patches_dict.items():
                    results['all_patches'][cls].extend(patches)

        # 单帧可视化
        if cfg['visualize_flag']:
            visualize_and_save(
                img_path, result['detected_objects'],
                os.path.join(cfg['visualize_save_dir'], part),
                result['scale_w'], result['scale_h'],
                result.get('bbox_xyxy', []))

    # 后处理和分析
    if cfg['calculate_metrics_flag']:
        # 保存缓存
        save_file_dir = cfg['metrics_cache_save_file']
        os.makedirs(save_file_dir, exist_ok=True)
        cache_file = f'{save_file_dir}/cache_{part}.pkl'

        with open(cache_file, 'wb') as file:
            pickle.dump({
                'bbox_cxcywhn': results['bbox_cxcywhn'],
                'bbox_xyxy': results['bbox_xyxy'],
                'candidate_targets': results['candidate_targets'],
                'frame_nums': results['frame_nums'],
                'scale_w': results['scale_w'],
                'scale_h': results['scale_h'],
                'img_paths': results['img_paths']
            }, file)

        # 计算指标
        detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
        pos_candidate_targets, neg_candidate_targets = calculate_metrics(
            classes, results['bbox_xyxy'], results['candidate_targets'],
            results['frame_nums'],
            results['scale_w'], results['scale_h'],
            threshold=cfg['threshold'],
            target_min_size=cfg['target_min_size'],
            iou=cfg['iou_flag'], cls_flag=cfg['cls_flag'])

        # 可视化置信度
        if cfg['analysis_show']:
            show_confidence(
                pos_conf_list, neg_conf_list,
                save_path=cfg['visualize_save_dir'],
                file_name=f'conf_{part}.png')

            # t-SNE分析
            analysis_dir = os.path.join(cfg['visualize_save_dir'], 'feature_analysis')
            tsne_features, tsne_labels = prepare_tsne_data(results['all_patches'])

            if tsne_features is not None:
                # 动态调整perplexity的可视化
                safe_tsne_visualization(tsne_features, tsne_labels, analysis_dir, part)

    # 输出推理速度
    if results['inference_times']:
        inference_times = sorted(results['inference_times'])[2:-2]  # 去掉极端值
        avg_time = sum(inference_times) / len(inference_times)
        print(f'\n平均推理速度: {avg_time:.2f} ms')


if __name__ == "__main__":
    """有标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict_dalachi_0615.yaml')
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

        predict_with_gt(predictor, part, data_path, label_path, show_feature=True,
                        cluster_min_size=16, max_cluster_samples=30000)
