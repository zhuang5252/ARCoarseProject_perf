import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from air_track.aligner.engine.predictor_return_dx_dy_angle import Predictor
from air_track.utils import combine_load_cfg_yaml, reprod_init, check_and_change_img_size, natural_sort_key


def predict_aot_folder_offsets(predictor, part):
    """预测数据集的偏移量"""
    cfg = predictor.cfg

    data_dir = cfg['data_dir']
    align_save_dir = cfg['align_save_dir']
    img_read_method = cfg['dataset_params']['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']

    torch.set_grad_enabled(False)

    df = pd.read_csv(f'{data_dir}/{part}/ImageSets/groundtruth.csv')
    flight_ids = list(sorted(df['flight_id'].unique()))

    inference_times = []
    for flight_id in tqdm(flight_ids):
        df_flight = df[df.flight_id == flight_id][['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
        img_files = [f'{data_dir}/{part}/Images/{flight_id}/{fn}' for fn in df_flight.img_name]

        load_failed = []

        frame_transforms_dx_dy_angle = [[0, 0, 0]]
        frame_transforms_tr = [np.zeros((2, 3))]

        prev_img = None
        for i, img_file in tqdm(enumerate(img_files), total=len(img_files)):
            cur_img_file = img_file
            if img_read_method == 'gray':
                cur_img = cv2.imread(cur_img_file, cv2.IMREAD_GRAYSCALE)
                predictor.max_pixel = 255
            elif img_read_method == 'unchanged':
                cur_img = cv2.imread(cur_img_file, cv2.IMREAD_UNCHANGED)
            else:
                cur_img = cv2.imread(cur_img_file)

            load_failed.append(cur_img is None)
            cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

            # 第一帧无需帧对齐
            if i == 0:
                prev_img = cur_img
                continue
            # 两帧图像有一帧不存在，dx、dy、angle均为0
            if prev_img is None or cur_img is None:
                frame_transforms_dx_dy_angle.append([0, 0, 0])
                frame_transforms_tr.append(np.zeros((2, 3)))
                inference_times.append(0)
                prev_img = cur_img
                continue

            item_dx_dy_angle, transform, inference_time = predictor.predict(prev_img, cur_img)
            frame_transforms_dx_dy_angle.append(item_dx_dy_angle)
            frame_transforms_tr.append(transform)
            inference_times.append(inference_time)
            prev_img = cur_img

        dst_dir = f'{align_save_dir}/frame_transforms/{part}'
        os.makedirs(dst_dir, exist_ok=True)

        frame_transforms_dx_dy_angle = np.array(frame_transforms_dx_dy_angle)
        frame_transforms_tr = np.array(frame_transforms_tr)
        df_flight['load_failed'] = load_failed  # 图像为空
        df_flight['dx'] = frame_transforms_dx_dy_angle[:, 0]
        df_flight['dy'] = frame_transforms_dx_dy_angle[:, 1]
        df_flight['angle'] = frame_transforms_dx_dy_angle[:, 2]
        # df_flight['tr'] = frame_transforms_tr

        df_flight.to_pickle(f'{dst_dir}/{flight_id}.pkl')
        df_flight.to_csv(f'{dst_dir}/{flight_id}.csv', index=False)


def predict_folder_offsets(predictor, part):
    """预测数据集的偏移量"""
    cfg = predictor.cfg

    data_dir = cfg['data_dir']
    img_folder = cfg['img_folder']
    img_format = cfg['img_format']
    align_save_dir = cfg['align_save_dir']
    img_read_method = cfg['dataset_params']['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']

    torch.set_grad_enabled(False)

    data_path = os.path.join(data_dir, part, img_folder)
    # 获取图片列表并排序
    img_names = [img for img in os.listdir(data_path) if img.endswith(img_format)]
    # 按自然顺序排序
    img_names.sort(key=natural_sort_key)

    inference_times = []
    frame_nums = []
    load_failed = []
    frame_transforms_dx_dy_angle = [[0, 0, 0]]
    frame_transforms_tr = [np.zeros((2, 3))]

    # 创建一个空的 DataFrame
    df_flight = pd.DataFrame()

    prev_img = None
    for i, img_name in tqdm(enumerate(img_names), total=len(img_names)):
        img_file = os.path.join(data_path, img_name)
        cur_img_file = img_file
        if img_read_method == 'gray':
            cur_img = cv2.imread(cur_img_file, cv2.IMREAD_GRAYSCALE)
            predictor.max_pixel = 255
        elif img_read_method == 'unchanged':
            cur_img = cv2.imread(cur_img_file, cv2.IMREAD_UNCHANGED)
        else:
            cur_img = cv2.imread(cur_img_file)

        frame_nums.append(i)
        load_failed.append(cur_img is None)
        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

        # 第一帧无需帧对齐
        if i == 0:
            prev_img = cur_img
            continue
        # 两帧图像有一帧不存在，dx、dy、angle均为0
        if prev_img is None or cur_img is None:
            frame_transforms_dx_dy_angle.append([0, 0, 0])
            frame_transforms_tr.append(np.zeros((2, 3)))
            inference_times.append(0)
            prev_img = cur_img
            continue

        item_dx_dy_angle, transform, inference_time = predictor.predict(prev_img, cur_img)
        frame_transforms_dx_dy_angle.append(item_dx_dy_angle)
        frame_transforms_tr.append(transform)
        inference_times.append(inference_time)
        prev_img = cur_img

    dst_dir = f'{align_save_dir}/frame_transforms/{part}'
    os.makedirs(dst_dir, exist_ok=True)

    frame_transforms_dx_dy_angle = np.array(frame_transforms_dx_dy_angle)
    frame_transforms_tr = np.array(frame_transforms_tr)
    df_flight['img_name'] = img_names
    df_flight['frame'] = frame_nums
    df_flight['load_failed'] = load_failed  # 图像为空
    df_flight['dx'] = frame_transforms_dx_dy_angle[:, 0]
    df_flight['dy'] = frame_transforms_dx_dy_angle[:, 1]
    df_flight['angle'] = frame_transforms_dx_dy_angle[:, 2]
    # df_flight['tr'] = frame_transforms_tr

    if '/' in part:
        flight = part.split('/')[-1]
    else:
        flight = part

    df_flight.to_pickle(f'{dst_dir}/{flight}.pkl')
    df_flight.to_csv(f'{dst_dir}/{flight}.csv', index=False)


if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, '../config/predict_multi_offset_dabaige.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model(model_path=None)

    data_dir = cfg_predict['data_dir']
    parts = cfg_predict['part_train'] + cfg_predict['part_val'] + cfg_predict['part_test']
    for part in parts:
        test_data_dir = os.path.join(data_dir, part)
        if os.path.isdir(test_data_dir):
            print('当前处理part: ', part)
            # predict_aot_folder_offsets(predictor, part)
            predict_folder_offsets(predictor, part)
