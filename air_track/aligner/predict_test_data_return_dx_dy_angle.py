import os
import cv2
import numpy as np
from air_track.aligner.engine.predictor_return_dx_dy_angle import Predictor
from air_track.utils import reprod_init, combine_load_cfg_yaml, check_and_change_img_size


def predict(predictor, data_dir):
    cfg = predictor.cfg
    img_read_method = cfg['dataset_params']['img_read_method'].lower()
    img_size_w, img_size_h = predictor.dataset_params['img_size']

    img_paths = [f for f in os.listdir(data_dir) if f.endswith(cfg['img_format'])]
    img_paths.sort()

    frame_num = len(img_paths)
    inference_times = []
    for idx, file_name in enumerate(img_paths):
        print(f'当前处理第{idx + 1}/{frame_num}帧')

        if 'prev' in file_name:
            prev_img_file = os.path.join(data_dir, file_name)
            temp = file_name.split('_')[0]
            cur_img_file = os.path.join(data_dir, temp + cfg['img_format'])
        else:
            cur_img_file = os.path.join(data_dir, file_name)
            temp = file_name.split('.')[0]
            prev_img_file = os.path.join(data_dir, temp + '_prev' + cfg['img_format'])

        if not os.path.exists(cur_img_file) or not os.path.exists(prev_img_file):
            continue

        if img_read_method == 'gray':
            cur_img = cv2.imread(cur_img_file, cv2.IMREAD_GRAYSCALE)
            prev_img = cv2.imread(prev_img_file, cv2.IMREAD_GRAYSCALE)
            predictor.max_pixel = 255
        elif img_read_method == 'unchanged':
            cur_img = cv2.imread(cur_img_file, cv2.IMREAD_UNCHANGED)
            prev_img = cv2.imread(prev_img_file, cv2.IMREAD_UNCHANGED)
        else:
            cur_img = cv2.imread(cur_img_file)
            prev_img = cv2.imread(prev_img_file)

        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        img_h, img_w = cur_img.shape[:2]

        dx_dy_angle, transform, inference_time = predictor.predict(prev_img, cur_img)
        inference_times.append(inference_time)

        prev_img_aligned = cv2.warpAffine(prev_img,
                                          transform[:2, :],
                                          dsize=(img_w, img_h),
                                          flags=cv2.INTER_LINEAR)

        if cfg['visualize_flag'] and 'prev' not in file_name:
            if len(cur_img.shape) == 3 and len(prev_img.shape) == 3:
                cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                prev_img_aligned = cv2.cvtColor(prev_img_aligned, cv2.COLOR_BGR2GRAY)
            save_path = cfg['visualize_save_dir']
            os.makedirs(save_path, exist_ok=True)
            file_name = file_name.split('.')[0]
            # cv2.imwrite(os.path.join(save_path, f'{file_name}_cur_img.png'), cur_img)
            # cv2.imwrite(os.path.join(save_path, f'{file_name}_prev_img.png'), prev_img)
            # cv2.imwrite(os.path.join(save_path, f'{file_name}_prev_img_aligned.png'), prev_img_aligned)
            merge_result = np.stack([cur_img, prev_img_aligned, cur_img], axis=2)
            cv2.imwrite(os.path.join(save_path, f'{file_name}_merge_result.png'), merge_result)


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    predict(predictor, cfg_predict['test_data_dir'])
