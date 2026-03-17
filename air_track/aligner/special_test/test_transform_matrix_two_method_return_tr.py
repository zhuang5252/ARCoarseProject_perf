import os
import cv2
import numpy as np
from air_track.aligner.engine.predictor_return_tr import Predictor
from air_track.aligner.utils.transform_utils import build_geom_transform_predict
from air_track.utils import combine_load_cfg_yaml, check_and_change_img_size, reprod_init


def predict(predictor, data_dir):
    sigma_scale = 0.0025  # 图像缩放参数
    sigma_angle = 0.2  # 图像旋转参数
    sigma_offset = 12.0  # 图像位移参数
    # sigma_scale = 0.025  # 图像缩放参数
    # sigma_angle = 0.5  # 图像旋转参数
    # sigma_offset = 18.0  # 图像位移参数
    cfg = predictor.cfg
    img_read_method = cfg['dataset_params']['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']
    save_path = cfg['visualize_save_dir']

    img_paths = [f for f in os.listdir(data_dir) if f.endswith(cfg['img_format'])]
    img_paths.sort()

    frame_num = len(img_paths)
    inference_times = []
    loss_1_list, loss_2_list = [], []
    loss_1_img_means, loss_2_img_means = [], []
    loss_1_img_maxes, loss_2_img_maxes = [], []
    loss_1_img_mins, loss_2_img_mins = [], []
    for idx, file_name in enumerate(img_paths):
        print(f'当前处理第{idx + 1}/{frame_num}帧')

        # 生成随机变化参数
        scale = np.exp(np.random.normal(0, sigma_scale * 2))
        angle = np.random.normal(0, sigma_angle * 2)
        dx = np.random.normal(0, sigma_offset * 2)
        dy = np.random.normal(0, sigma_offset * 2)
        # scale = np.exp(sigma_scale * 2)
        # angle = sigma_angle * 2
        # dx = sigma_offset * 2
        # dy = sigma_offset * 2
        tr = build_geom_transform_predict(translation_x=dx, translation_y=dy,
                                          scale_x=scale, scale_y=scale,
                                          angle=angle,
                                          return_params=True)

        prev_img_file = os.path.join(data_dir, file_name)

        if img_read_method == 'gray':
            prev_img = cv2.imread(prev_img_file, cv2.IMREAD_GRAYSCALE)
            predictor.max_pixel = 255
        elif img_read_method == 'unchanged':
            prev_img = cv2.imread(prev_img_file, cv2.IMREAD_UNCHANGED)
        else:
            prev_img = cv2.imread(prev_img_file)

        orig_h, orig_w = prev_img.shape[:2]
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        # if (orig_h, orig_w) != (512, 640):
        #     prev_img = transform_img_size(prev_img, w=img_size_w, h=img_size_h)

        img_h, img_w = prev_img.shape[:2]
        cur_img = cv2.warpAffine(prev_img,
                                 tr[:2, :],
                                 dsize=(img_w, img_h),
                                 flags=cv2.INTER_LINEAR)

        transform, inference_time = predictor.predict(prev_img, cur_img)
        # transform, inference_time = predictor.predict(cur_img, prev_img)
        inference_times.append(inference_time)

        cur_img_1 = cv2.warpAffine(prev_img,
                                   transform[:2, :],
                                   dsize=(img_w, img_h),
                                   flags=cv2.INTER_LINEAR)
        cur_img_2 = cv2.warpAffine(prev_img,
                                   transform[:2, :],
                                   dsize=(img_w, img_h),
                                   flags=cv2.INTER_LINEAR)

        if cfg['visualize_flag']:
            save_path_1 = os.path.join(save_path, 'cur_img_1')
            save_path_2 = os.path.join(save_path, 'cur_img_2')
            os.makedirs(save_path_1, exist_ok=True)
            os.makedirs(save_path_2, exist_ok=True)
            merge_result = np.stack([cur_img, cur_img_1, cur_img], axis=2)
            cv2.imwrite(os.path.join(save_path_1, f'{idx}_merge_result.png'), merge_result)
            merge_result = np.stack([cur_img, cur_img_2, cur_img], axis=2)
            cv2.imwrite(os.path.join(save_path_2, f'{idx}_merge_result.png'), merge_result)

        loss_1 = np.abs(tr[:2] - transform[:2])
        loss_2 = np.abs(tr[:2] - transform[:2])
        loss_1_list.append(loss_1)
        loss_2_list.append(loss_2)

        cur_img_1 = cur_img_1.astype(np.int16)
        cur_img_2 = cur_img_2.astype(np.int16)
        prev_img = prev_img.astype(np.int16)

        temp = 50
        loss_1_img = np.abs(
            cur_img_1[temp:img_h - temp, temp:img_w - temp] - prev_img[temp:img_h - temp, temp:img_w - temp])
        loss_2_img = np.abs(
            cur_img_2[temp:img_h - temp, temp:img_w - temp] - prev_img[temp:img_h - temp, temp:img_w - temp])
        loss_1_img_mean = np.mean(loss_1_img)
        loss_2_img_mean = np.mean(loss_2_img)
        loss_1_img_max = np.max(loss_1_img)
        loss_2_img_max = np.max(loss_2_img)
        loss_1_img_min = np.min(loss_1_img)
        loss_2_img_min = np.min(loss_2_img)
        loss_1_img_means.append(loss_1_img_mean)
        loss_2_img_means.append(loss_2_img_mean)
        loss_1_img_maxes.append(loss_1_img_max)
        loss_2_img_maxes.append(loss_2_img_max)
        loss_1_img_mins.append(loss_1_img_min)
        loss_2_img_mins.append(loss_2_img_min)

    loss_1_mean = np.mean(np.array(loss_1_list))
    loss_1_max = np.max(np.array(loss_1_list))
    loss_1_min = np.min(np.array(loss_1_list))
    loss_2_mean = np.mean(np.array(loss_2_list))
    loss_2_max = np.max(np.array(loss_2_list))
    loss_2_min = np.min(np.array(loss_2_list))
    print(f'loss_1: loss_1_mean: {loss_1_mean}, loss_1_max: {loss_1_max}, loss_1_min: {loss_1_min}')
    print(f'loss_2: loss_2_mean: {loss_2_mean}, loss_2_max: {loss_2_max}, loss_2_min: {loss_2_min}')

    loss_1_img_mean = np.mean(np.array(loss_1_img_means))
    loss_1_img_max = np.max(np.array(loss_1_img_maxes))
    loss_1_img_min = np.min(np.array(loss_1_img_mins))
    loss_2_img_mean = np.mean(np.array(loss_2_img_means))
    loss_2_img_max = np.max(np.array(loss_2_img_maxes))
    loss_2_img_min = np.min(np.array(loss_2_img_mins))
    print(f'loss_1_img_mean: {loss_1_img_mean}, loss_1_img_max: {loss_1_img_max}, loss_1_img_min: {loss_1_img_min}')
    print(f'loss_2_img_mean: {loss_2_img_mean}, loss_2_img_max: {loss_2_img_max}, loss_2_img_min: {loss_2_img_min}')


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.model_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack_all/AirTrack_multi_frame_train/model_saved/TrEstimatorTsm/aligner_return_tr/minimum_loss.pt'
    predictor.set_model()

    predict(predictor, cfg_predict['test_data_dir'])
