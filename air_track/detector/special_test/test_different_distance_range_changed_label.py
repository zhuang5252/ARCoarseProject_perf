import math
import os
import pickle
import timeit
import cv2
import torch
import numpy as np
from air_track.detector.data.base_dataset.gaussian_render import gaussian2D
from air_track.detector.model import hrnet
from air_track.detector.utils.calculate_metrics import calculate_metrics_without_distance
from air_track.detector.utils.detect_utils import argmax2d, calc_iou, combine_images, read_flight_id_csv
from air_track.detector.visualization.visualize_and_save import visualize_and_save
from air_track.utils import common_utils, load_model, reprod_init, check_and_change_img_size, numpy_2_tensor


def build_model(cfg):
    model_params = cfg['model_params']
    model = hrnet.__dict__[model_params['model_cls']](cfg=model_params, pretrained=False)
    return model


def pred_to_detections(
        comb_pred,
        offset,
        size,
        tracking,
        distance,
        offset_scale: float = 256.0,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.025,
        down_scale: int = 2,
        target_min_size: int = 2,
        x_pad: int = 0,
        y_pad: int = 0
):
    comb_pred = comb_pred.copy()
    res_detections = []

    # comb_pred = comb_pred[0]
    h4, w4 = comb_pred.shape

    while True:
        y, x = argmax2d(comb_pred)
        # conf = comb_pred[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].sum()
        conf = comb_pred[y, x]
        # print('conf', conf)
        if conf < conf_threshold or np.isnan(conf):
            break

        w = 2 ** size[0, y, x]
        h = 2 ** size[1, y, x]
        d = 2 ** distance[y, x]

        # 按照输出图的大小确定最大值（值为图像尺寸 / pred_scale），后续如果尺寸多变，需进一步优化
        w = min(w4, max(target_min_size, w))
        h = min(h4, max(target_min_size, h))
        # item_cls = cls[:, y, x]
        item_tracking = tracking[:, y, x] * offset_scale

        cx_img = x
        cy_img = y
        cx = (x + 0.5) * down_scale
        cy = (y + 0.5) * down_scale

        new_item = dict(
            conf=conf,
            cx=cx + x_pad,
            cy=cy + y_pad,
            w=w,
            h=h,
            distance=d,
            tracking=list(item_tracking),
            offset=list(offset[:, y, x] * down_scale),
        )

        overlaps = False
        for prev_item in res_detections:
            if calc_iou(prev_item, new_item) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            res_detections.append(new_item)

        w = math.ceil(w * 2 / down_scale)
        h = math.ceil(h * 2 / down_scale)
        w = max(1, w // 2 * 2 + 1)
        h = max(1, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 2, sigma_y=h / 2)

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        comb_pred[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w] -= item_mask

    return res_detections


def process_input(cur_image, prev_images, max_pixel, device):
    """输入数据前处理"""
    cur_tensor_img = numpy_2_tensor(cur_image, max_pixel).unsqueeze(0)

    # prev_images可能不止一帧
    prev_tensor_images = []
    for prev_img in prev_images:
        prev_tensor_img = numpy_2_tensor(prev_img, max_pixel).unsqueeze(0)
        prev_tensor_images.append(prev_tensor_img)

    # 将cur数据与prev数据合并
    input_data = combine_images(prev_tensor_images, cur_tensor_img)

    input_data = input_data.float().to(device)

    return input_data


def process_output(pred, max_nb_objects=4, conf_threshold=0.25, iou_threshold=0.25,
                   down_scale=2, target_min_size=2, x_pad=0, y_pad=0, offset_scale=256.0):
    # pred = [size, offset, distance, tracking, mask]
    pred = {
        'size': pred[0],
        'offset': pred[1],
        'distance': pred[2],
        'tracking': pred[3],
        'mask': pred[4],  # 按照当前模型设置，始终存在mask的输出
    }

    # pred结果包含5部分输出：size、offset、distance、tracking、mask
    pred['mask'] = torch.sigmoid(pred['mask'][0].float())  # torch.Size([1, 256, 320])

    for k in list(pred.keys()):
        pred[k] = pred[k][0].float().cpu().detach().numpy()

    # 对模型输出的pred转为坐标
    detected_objects = pred_to_detections(
        comb_pred=pred['mask'],
        offset=pred['offset'],
        size=pred['size'],
        tracking=pred['tracking'],
        distance=pred['distance'][0],
        offset_scale=offset_scale,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        down_scale=down_scale,  # 根据模型要修改
        target_min_size=target_min_size,
        x_pad=x_pad,
        y_pad=y_pad,
    )

    # 保留最大的max_nb_objects个目标
    detected_objects = detected_objects[:max_nb_objects]

    return detected_objects


def detect(model, cfg, cur_image, prev_images, device='cpu'):
    """模型检测"""
    model.eval()

    # False禁用梯度跟踪
    torch.set_grad_enabled(False)

    # 输入数据前处理
    input_data = process_input(cur_image, prev_images, cfg['max_pixel'], device)

    # 模型预测
    start_time = timeit.default_timer()
    if cfg['half']:
        model = model.half()
        input_data = input_data.half()
        pred = model(input_data)
    else:
        # 自动混合精度计算
        # with torch.cuda.amp.autocast():
        pred = model(input_data)

    end_time = timeit.default_timer()
    # 推理耗时
    inference_time = (end_time - start_time) * 1000

    # 输出数据后处理
    detected_objects = process_output(pred, max_nb_objects=cfg['max_nb_objects'],
                                      conf_threshold=cfg['conf_threshold'], iou_threshold=cfg['iou_threshold'],
                                      down_scale=cfg['down_scale'], target_min_size=cfg['target_min_size'],
                                      x_pad=cfg['x_pad'], y_pad=cfg['y_pad'], offset_scale=cfg['offset_scale'])

    return detected_objects, inference_time


def get_prev_step_frame(path, file_names, index, frame_step, input_frames, img_size_w, img_size_h):
    """
    从file_names中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    prev_images = []

    for i in range(frame_step, input_frames, frame_step):
        interval_index = i * frame_step

        file_name = file_names[index - interval_index]
        file = os.path.join(path, file_name)

        prev_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        prev_images.append(prev_img)

    return prev_images


def predict_with_gt(cfg, data_dir, part, flight_id, range_distance=None):
    """主函数读取一个文件夹下的图像进行检测"""
    reprod_init(seed=cfg['seed'])

    frame_step = cfg['frame_step']
    input_frames = cfg['input_frames']
    model_path = cfg['model_path']
    img_size_w, img_size_h = cfg['dataset_params']['img_size']
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    # 超参数设置为小于2帧或者步长不大于0，则返回空列表
    if input_frames < 2 or frame_step <= 0:
        assert ValueError('input_frames < 2 or frame_step <= 0. ')

    img_folder = cfg['img_folder']
    img_path = f'{data_dir}/{part}/{img_folder}/{flight_id}'

    # 定义及load模型
    model = build_model(cfg)
    model = load_model(model, model_path, device)

    # 从 groundtruth.csv 文件中提取所需信息
    df_flight, frame_numbers, file_names = read_flight_id_csv(cfg, data_dir, flight_id, part)

    break_frame_num = 0
    flag = np.zeros(len(df_flight))
    for (min_distance, max_distance) in range_distance:
        print('Min distance is: ', min_distance, 'Max distance is: ', max_distance)
        inference_times = []
        bbox_gts = []
        candidate_targets = []
        temp = 0  # 用来跳过标签存在这一帧，但不存在图像数据的帧
        scale_w, scale_h = 0, 0
        for i, (frame_num, file_name) in enumerate(zip(frame_numbers, file_names)):
            # print(f"当前处理第{i}帧，帧号为{frame_num}")
            if frame_num < break_frame_num:
                continue

            distance = df_flight['range_distance_m'].iloc[i]
            if min_distance < distance <= max_distance or np.isnan(distance):

                if not flag[i]:
                    if i < frame_step * (input_frames - 1):
                        continue

                    if i < temp:
                        continue

                    cur_file = os.path.join(img_path, file_name)
                    cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)

                    # 若当前图片不存在，则使用temp使得cur_img读取 i + frame_step * input_frames 帧
                    if cur_img is None:
                        print('miss img: ', cur_file)
                        temp = i + frame_step * input_frames
                        continue

                    # 图像存在才进行后续操作
                    orig_h, orig_w = cur_img.shape[:2]
                    scale_w = img_size_w / orig_w
                    scale_h = img_size_h / orig_h
                    cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

                    # 读取prev帧图像数据
                    prev_images = get_prev_step_frame(img_path, file_names, i, frame_step,
                                                      input_frames, img_size_w, img_size_h)

                    # 进行检测，返回推理耗时
                    detected_objects, inference_time = detect(model, cfg, cur_img, prev_images, device=device)
                    inference_times.append(inference_time)

                    # 取出列表中标题的index行的数据
                    gt_left = df_flight['gt_left'].iloc[i]
                    gt_right = df_flight['gt_right'].iloc[i]
                    gt_top = df_flight['gt_top'].iloc[i]
                    gt_bottom = df_flight['gt_bottom'].iloc[i]
                    distance = df_flight['range_distance_m'].iloc[i]

                    bbox_gt = (gt_left, gt_top, gt_right, gt_bottom)

                    bbox_gts.append(bbox_gt)
                    candidate_targets.append(detected_objects)

                    # 检测结果可视化
                    if cfg['visualize_flag']:
                        # 可视化结果保存
                        visualize_and_save(cur_file, detected_objects, os.path.join((cfg['visualize_save_dir']), part),
                                           scale_w, scale_h, bbox_gt, distance)

                    flag[i] = 1
            else:
                break_frame_num = frame_num
                break

        if cfg['calculate_metrics_flag']:
            save_file_dir = cfg['metrics_cache_save_file']
            os.makedirs(save_file_dir, exist_ok=True)
            cache_file = f'{save_file_dir}/cache_{flight_id}.pkl'
            with open(cache_file, 'wb') as file:
                pickle.dump({'bbox_gts': bbox_gts, 'candidate_targets': candidate_targets}, file)

            calculate_metrics_without_distance(bbox_gts, candidate_targets, scale_w, scale_h,
                                               threshold=cfg['threshold'],
                                               target_min_size=cfg['target_min_size'], iou=cfg['iou_flag'])

        inference_times.sort()
        inference_times = inference_times[2:-2]
        print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms\n')


if __name__ == "__main__":
    """有标签预测"""
    path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/123/AirTrack_results/model_saved/HRNet48/8km/normal_gaussian/6_6'
    dataset_yaml = path + '/dataset.yaml'
    train_yaml = path + '/hrnet_train.yaml'
    pred_yaml = path + '/predict.yaml'
    # 合并若干个yaml的配置文件内容
    yaml_list = [dataset_yaml, train_yaml, pred_yaml]
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 固定随机数种子
    reprod_init(cfg_data['seed'])

    range_distance = [
        [25000, 35000],
        [20000, 25000],
        [15000, 20000],
        [10000, 15000],
        [5000, 10000],
        [0, 5000]
    ]

    for part in cfg_data['part_test']:
        print('当前测试数据集 part：', part)
        for flight_id in os.listdir(os.path.join(cfg_data['data_dir'], part, cfg_data['img_folder'])):
            print('当前测试数据集 flight_id：', flight_id)
            predict_with_gt(cfg_data, cfg_data['data_dir'], part, flight_id, range_distance=range_distance)
