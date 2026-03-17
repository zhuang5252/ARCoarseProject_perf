import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from air_track.aligner.data.data_load import load_datasets
from air_track.aligner.utils.transform_utils import transform_img_size, gen_transform, img_apply_transform
from air_track.utils import common_utils


class AlignDataset(torch.utils.data.Dataset):
    def __init__(self,
                 stage,
                 cfg_data
                 ):
        self.stage = stage
        self.cfg = cfg_data
        self.dataset_params = cfg_data['dataset_params']

        self.points_shape = eval(cfg_data['points_shape'])

        # 图像变换矩阵参数
        self.sigma_scale = self.dataset_params['sigma_scale']
        self.sigma_angle = self.dataset_params['sigma_angle']
        self.sigma_offset = self.dataset_params['sigma_offset']

        self.img_read_method = self.dataset_params['img_read_method'].lower()
        self.downscale = self.dataset_params['downscale']
        self.return_torch_tensors = self.dataset_params['return_torch_tensors']
        self.img_w, self.img_h = self.dataset_params['img_size']

        # 通过stage来选择对应的parts
        if self.stage == cfg_data['stage_train']:
            parts = cfg_data['part_train']
            sample_nums = int(self.dataset_params['train_sample_nums'])
        else:
            parts = cfg_data['part_val']
            sample_nums = int(self.dataset_params['val_sample_nums'])

        # 加载数据
        self.parts = parts
        self.frames = load_datasets(self.cfg, self.parts, self.stage)

        if len(self.frames) > sample_nums:
            # 打乱 DataFrame 的顺序
            self.frames = self.frames.sample(frac=1)
            # 取前3000个数据
            self.frames = self.frames.head(sample_nums)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):

        frame = self.frames.iloc[index]  # 当前帧

        if self.img_read_method == 'gray':
            prev_img = cv2.imread(frame.img_path, cv2.IMREAD_GRAYSCALE)
        elif self.img_read_method == 'unchanged':
            prev_img = cv2.imread(frame.img_path, cv2.IMREAD_UNCHANGED)
        else:
            prev_img = cv2.imread(frame.img_path)

        # 当前帧的图像不存在
        if prev_img is None:
            return self.__getitem__(index + 1)

        # 随机生成变换矩阵
        prev_tr_params = dict(
            scale=np.exp(np.random.normal(0, self.sigma_scale * 2)),  # 接近于 1
            angle=np.random.normal(0, self.sigma_angle * 2),
            dx=np.random.normal(0, self.sigma_offset * 2),
            dy=np.random.normal(0, self.sigma_offset * 2)
        )

        # 大图变换矩阵
        shape_orig = (prev_img.shape[1], prev_img.shape[0])
        tr_transform_orig = gen_transform(shape_orig, prev_tr_params['dx'],
                                          prev_tr_params['dy'], prev_tr_params['angle'])
        cur_img_orig = img_apply_transform(prev_img, tr_transform_orig)

        # 从大图中心扣出小图（若原图尺寸小于小图尺寸，则直接resize）
        prev_img_crop = transform_img_size(prev_img, self.img_w, self.img_h)
        cur_img_crop = transform_img_size(cur_img_orig, self.img_w, self.img_h)

        # 中心扣出的小图的变换矩阵
        shape_new = (self.img_w, self.img_h)
        tr_transform_new = gen_transform(shape_new, prev_tr_params['dx'],
                                         prev_tr_params['dy'], prev_tr_params['angle'])

        res = {'part': frame.part, 'flight_id': frame.flight_id, 'image_name': frame.img_name,
               'cur_img': cur_img_crop, 'prev_img': prev_img_crop,
               'dx': prev_tr_params['dx'], 'dy': prev_tr_params['dy'], 'angle': prev_tr_params['angle'],
               'tr_transform': tr_transform_new}

        if self.return_torch_tensors:
            if len(res['cur_img'].shape) == 2:
                res['cur_img'] = torch.from_numpy(res['cur_img'] / self.cfg['max_pixel']).unsqueeze(0).float()
                res['prev_img'] = torch.from_numpy(res['prev_img'] / self.cfg['max_pixel']).unsqueeze(0).float()
            else:
                res['cur_img'] = torch.from_numpy(res['cur_img'] / self.cfg['max_pixel']).float().permute(2, 0, 1)
                res['prev_img'] = torch.from_numpy(res['prev_img'] / self.cfg['max_pixel']).float().permute(2, 0, 1)

            res['dx'] = torch.from_numpy(np.array(res['dx'])).float()
            res['dy'] = torch.from_numpy(np.array(res['dy'])).float()
            res['angle'] = torch.from_numpy(np.array(res['angle'])).float()
            res['tr_transform'] = torch.from_numpy(res['tr_transform'][:2, :]).float()

        return res


if __name__ == '__main__':
    """用于测试阶段使用，现作为展示代码，若需单独使用，将下段代码拿出去到单独脚本中使用，莫要修改此文件"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_yaml = os.path.join(script_dir, 'config/dataset.yaml')
    train_yaml = os.path.join(script_dir, 'config/align_train.yaml')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]

    # 合并读取若干个yaml的配置文件内容
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    common_utils.reprod_init(seed=cfg_data['seed'])

    dataset_train = AlignDataset(
        stage=cfg_data['stage_train'],
        cfg_data=cfg_data
    )

    train_dl = DataLoader(
        dataset_train,
        num_workers=0,
        shuffle=True,
        batch_size=1,
    )

    for i, item in enumerate(train_dl):
        print(i, item.keys())
