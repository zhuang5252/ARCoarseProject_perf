import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from air_track.aligner.engine.predictor_return_dx_dy_angle import Predictor
from air_track.utils import combine_load_cfg_yaml, reprod_init, check_and_change_img_size


def process_single_flight(predictor, part, data_dir, align_save_dir, img_read_method, img_size_w, img_size_h,
                          flight_id):
    """处理单个 flight_id 的优化函数"""
    # 读取数据
    df = pd.read_csv(f'{data_dir}/{part}/ImageSets/groundtruth.csv')
    df_flight = df[df.flight_id == flight_id][['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
    img_files = [f'{data_dir}/{part}/Images/{flight_id}/{fn}' for fn in df_flight.img_name]

    # 预加载所有图像到内存
    images = []
    load_failed = []
    for img_file in img_files:
        if img_read_method == 'gray':
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(img_file)
        if img is None:
            load_failed.append(True)
            images.append(None)
        else:
            load_failed.append(False)
            img = check_and_change_img_size(img, img_size_w, img_size_h)
            images.append(img)

    # 初始化变量
    frame_transforms_dx_dy_angle = [[0, 0, 0]]
    frame_transforms_tr = [np.zeros((3, 3))]
    inference_times = []
    prev_img = images if images else None

    # 批量处理帧对
    batch_size = 1  # 根据 GPU 内存调整
    for i in range(1, len(images)):
        cur_img = images[i]
        if prev_img is None or cur_img is None:
            frame_transforms_dx_dy_angle.append([0, 0, 0])
            frame_transforms_tr.append(np.zeros((3, 3)))
            inference_times.append(0)
            prev_img = cur_img
            continue

        # 批量推理（假设 predictor 支持批量）
        if i % batch_size == 0:
            batch_prev = images[i - batch_size:i]
            batch_cur = images[i - batch_size + 1:i + 1]
            batch_results = predictor.predict(batch_prev, batch_cur)  # 需实现 batch_predict
            for res in batch_results:
                frame_transforms_dx_dy_angle.append(res)
                frame_transforms_tr.append(res)
                inference_times.append(res)
        else:
            item_dx_dy_angle, transform, inference_time = predictor.predict(prev_img, cur_img)
            frame_transforms_dx_dy_angle.append(item_dx_dy_angle)
            frame_transforms_tr.append(transform)
            inference_times.append(inference_time)

        prev_img = cur_img

    # 保存结果
    dst_dir = f'{align_save_dir}/frame_transforms/{part}'
    os.makedirs(dst_dir, exist_ok=True)
    df_flight['load_failed'] = load_failed
    df_flight['dx'] = np.array(frame_transforms_dx_dy_angle)[:, 0]
    df_flight['dy'] = np.array(frame_transforms_dx_dy_angle)[:, 1]
    df_flight['angle'] = np.array(frame_transforms_dx_dy_angle)[:, 2]
    df_flight.to_pickle(f'{dst_dir}/{flight_id}.pkl')
    df_flight.to_csv(f'{dst_dir}/{flight_id}.csv', index=False)
    return inference_times


def predict_folder_offsets(predictor, part, num_threads=1):
    """主函数：多线程并行处理所有 flight_id"""
    cfg = predictor.cfg
    data_dir = cfg['data_dir']
    align_save_dir = cfg['align_save_dir']
    img_read_method = cfg['dataset_params']['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']
    torch.set_grad_enabled(False)

    # 读取所有 flight_id
    df = pd.read_csv(f'{data_dir}/{part}/ImageSets/groundtruth.csv')
    flight_ids = sorted(df['flight_id'].unique())

    for flight_id in flight_ids:
        process_single_flight(predictor, part, data_dir, align_save_dir, img_read_method, img_size_w, img_size_h,
                              flight_id)

    # 使用 ThreadPool 替代 ProcessPool
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     # 提交任务
    #     futures = [
    #         executor.submit(
    #             process_single_flight,
    #             predictor, part, data_dir, align_save_dir,
    #             img_read_method, img_size_w, img_size_h, flight_id
    #         ) for flight_id in flight_ids
    #     ]
    #
    #     # 使用 tqdm 显示进度
    #     all_times = []
    #     for future in tqdm(futures, total=len(flight_ids)):
    #         all_times.extend(future.result())

    # 合并时间统计
    print(f"平均推理时间: {np.mean(all_times):.2f}s")


if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, '../config/predict_multi_offset.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    data_dir = cfg_predict['data_dir']
    for part in os.listdir(data_dir):
        test_data_dir = os.path.join(data_dir, part)
        if os.path.isdir(test_data_dir):
            print('当前处理part: ', part)
            predict_folder_offsets(predictor, part)
