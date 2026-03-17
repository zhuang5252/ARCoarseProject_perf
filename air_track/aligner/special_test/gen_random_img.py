import os
import cv2
import numpy as np
from air_track.aligner.utils.transform_utils import gen_transform, img_apply_transform
from air_track.utils import combine_load_cfg_yaml, reprod_init

"""
使用随机生成的transform矩阵
对当前帧进行变换得到前一帧图像
"""

if __name__ == "__main__":
    path = f'/home/csz_changsha/data/test_data/orig_img'
    save_path = '/home/csz_changsha/data/test_data/single_channel_20250409'

    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    dataset_params = cfg_predict['dataset_params']
    img_size = tuple(dataset_params['img_size'])
    downscale = dataset_params['downscale']
    sigma_scale = dataset_params['sigma_scale']
    sigma_angle = dataset_params['sigma_angle']
    sigma_offset = dataset_params['sigma_offset']

    os.makedirs(save_path, exist_ok=True)
    img_paths = os.listdir(path)

    for item in img_paths:
        idx = item.split('.')[0]
        orig_img_path = os.path.join(path, item)
        cur_img_path = os.path.join(save_path, f'{idx}.jpg')
        prev_img_path = os.path.join(save_path, f'{idx}_prev.jpg')

        prev_img = cv2.imread(orig_img_path, cv2.IMREAD_UNCHANGED)
        # 按照灰度图来读取
        if len(prev_img.shape) > 2:
            prev_img = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)

        prev_img_crop = cv2.resize(prev_img, img_size)
        # prev_img_crop = prev_img

        # 生成随机变化参数
        prev_tr_params = dict(
            scale=np.exp(sigma_scale * 2),
            angle=sigma_angle * 2,
            dx=sigma_offset * 2,
            dy=sigma_offset * 2
        )

        shape = (prev_img_crop.shape[1], prev_img_crop.shape[0])
        tr_transform = gen_transform(shape, prev_tr_params['dx'],
                                     prev_tr_params['dy'], prev_tr_params['angle'])
        cur_img = img_apply_transform(prev_img_crop, tr_transform)

        cv2.imwrite(cur_img_path, cur_img)
        cv2.imwrite(prev_img_path, prev_img_crop)
