import math
import os
import pickle
import cv2
import torch
import numpy as np

from air_track.detector.data.base_dataset.gaussian_render import gaussian2D
from air_track.detector.model import hrnet
from air_track.detector.utils.detect_utils import argmax2d, calc_iou, combine_images
from air_track.detector.visualization.visualize_and_save import draw_feature_img_orig_data
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


def process_output(pred, cur_file, target_min_size=2, save_path='output', ):
    # pred = [size, offset, distance, tracking, mask]
    # pred = {
    #     'cls': pred[0],
    #     'size': pred[1],
    #     'offset': pred[2],
    #     'distance': pred[3],
    #     'tracking': pred[4],
    #     'mask': pred[5],  # 按照当前模型设置，始终存在mask的输出
    # }
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

    base_name = os.path.basename(cur_file).split('.')[0]

    max_iloc_y, max_iloc_x = argmax2d(pred['mask'])
    for k in list(pred.keys()):
        pkl_path = os.path.join(save_path, f'model_pred_data_{target_min_size}_{target_min_size}/model_pred_{k}_data')
        os.makedirs(pkl_path, exist_ok=True)
        with open(pkl_path + f'/{base_name}.pkl', 'wb') as file:
            pickle.dump(pred[k], file)
        draw_feature_img_orig_data(pred[k],
                                   save_path + f'/model_pred_{target_min_size}_{target_min_size}_with_star/model_pred_{k}_with_star/{base_name}.png',
                                   is_plt_star=True, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)
        draw_feature_img_orig_data(pred[k],
                                   save_path + f'/model_pred_{target_min_size}_{target_min_size}_without_star/model_pred_{k}_without_star/{base_name}.png',
                                   is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)


def detect(model, cfg, cur_file, cur_image, prev_images, device='cpu', save_path='output'):
    """模型检测"""
    model.eval()

    # False禁用梯度跟踪
    torch.set_grad_enabled(False)

    # 输入数据前处理
    input_data = process_input(cur_image, prev_images, cfg['max_pixel'], device)

    # 模型预测
    if cfg['half']:
        model = model.half()
        input_data = input_data.half()
        pred = model(input_data)
    else:
        # 自动混合精度计算
        # with torch.cuda.amp.autocast():
        pred = model(input_data)

    # 输出数据后处理
    process_output(pred, cur_file=cur_file, target_min_size=cfg['target_min_size'], save_path=save_path)


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


def predict_without_gt(cfg, img_path, save_path):
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

    # 定义及load模型
    model = build_model(cfg)
    model = load_model(model, model_path, device)

    file_names = os.listdir(img_path)
    file_names.sort()

    for i, file_name in enumerate(file_names):
        print(f"当前处理第{i}帧，名称为{file_name}")

        if i < frame_step * (input_frames - 1):
            continue
        cur_file = os.path.join(img_path, file_name)

        cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

        # 读取prev帧图像数据
        prev_images = get_prev_step_frame(img_path, file_names, i, frame_step, input_frames, img_size_w, img_size_h)

        # 进行检测，返回推理耗时
        detect(model, cfg, cur_file, cur_img, prev_images, device=device, save_path=save_path)


if __name__ == "__main__":
    """有标签预测"""
    path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/model_saved/HRNet48/8km_without_over_20/8km_trained_under20km_val_test_all_distances_删除纯数字数据集_删除20公里外的小目标图片_验证集包含大于20km数据_测试集为全集/center_all_1/6_6'
    dataset_yaml = path + '/dataset.yaml'
    train_yaml = path + '/hrnet_train.yaml'
    pred_yaml = path + '/predict.yaml'

    # 合并若干个yaml的配置文件内容
    yaml_list = [dataset_yaml, train_yaml, pred_yaml]
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 固定随机数种子
    reprod_init(cfg_data['seed'])

    # parts = cfg_data['part_train'] + cfg_data['part_val'] + cfg_data['part_test']
    parts = cfg_data['part_test']
    for part in parts:
        print('当前测试数据集 part：', part)
        for flight_id in os.listdir(os.path.join(cfg_data['data_dir'], part, cfg_data['img_folder'])):
            print('当前测试数据集 flight_id：', flight_id)

            files_path = os.path.join(cfg_data['data_dir'], part, cfg_data['img_folder'], flight_id)
            save_path = f'/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/特征图可视化/8km_without_over_20/8km_trained_under20km_val_test_all_distances_删除纯数字数据集_删除20公里外的小目标图片_验证集包含大于20km数据_测试集为全集/center_all_1/6_6/visual_feature_{part}'

            predict_without_gt(cfg_data, files_path, save_path)
