import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from air_track.aligner.engine.predictor_return_dx_dy_angle import Predictor
from air_track.utils import combine_load_cfg_yaml, reprod_init, check_and_change_img_size


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, part, flight_id, df):
        self.cfg = cfg
        self.data_dir = cfg['data_dir']
        self.align_save_dir = cfg['align_save_dir']
        self.img_read_method = cfg['dataset_params']['img_read_method'].lower()
        self.img_size_w, self.img_size_h = cfg['dataset_params']['img_size']
        self.part = part
        self.flight_id = flight_id
        self.df = df

        df_flight = df[df.flight_id == self.flight_id][['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
        self.img_files = [f'{self.data_dir}/{self.part}/Images/{self.flight_id}/{fn}' for fn in self.df_flight.img_name]

    def __len__(self):
        return len(self.img_files)


def predict_folder_offsets(predictor, part):
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
        frame_transforms_tr = [np.zeros((3, 3))]

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
                frame_transforms_tr.append(np.zeros((3, 3)))
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
        # frame_transforms_tr = np.array(frame_transforms_tr)
        df_flight['load_failed'] = load_failed     # 图像为空
        df_flight['dx'] = frame_transforms_dx_dy_angle[:, 0]
        df_flight['dy'] = frame_transforms_dx_dy_angle[:, 1]
        df_flight['angle'] = frame_transforms_dx_dy_angle[:, 2]
        # df_flight['tr'] = frame_transforms_tr

        df_flight.to_pickle(f'{dst_dir}/{flight_id}.pkl')
        df_flight.to_csv(f'{dst_dir}/{flight_id}.csv', index=False)

    inference_times.sort()
    inference_times = inference_times[1:-1]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


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
