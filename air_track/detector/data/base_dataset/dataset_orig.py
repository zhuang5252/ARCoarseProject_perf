import os
import cv2
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from air_track.detector.data.base_dataset.gaussian_render import render_y
from air_track.utils import common_utils, check_and_change_img_size


class DetectDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            stage,
            cfg_data,
    ):
        self.stage = stage
        self.cfg_data = cfg_data
        self.dataset_params = self.cfg_data['dataset_params']
        self.classes = self.cfg_data['classes']
        self.img_read_method = self.dataset_params['img_read_method'].lower()

        if stage == self.cfg_data['stage_train']:
            self.is_training = True
            self.parts = self.cfg_data['part_train']
        elif stage == self.cfg_data['stage_valid']:
            self.is_training = False
            self.parts = self.cfg_data['part_val']
        else:
            self.is_training = False
            self.parts = self.cfg_data['part_test']

        self.img_size_w, self.img_size_h = self.dataset_params['img_size']  # 指定图像尺寸
        self.down_scale = self.dataset_params['down_scale']  # 1 模型及数据下采样的比例
        self.target_min_size = self.dataset_params['target_min_size']  # 5 目标最小尺寸

        self.return_torch_tensors = self.dataset_params.get('return_torch_tensors', True)
        self.gamma_flag = self.dataset_params.get('gamma_flag', False)

        self.img_paths = []
        for part in self.parts:
            data_dir = os.path.join(self.cfg_data['data_dir'], part, self.cfg_data['img_folder'])
            img_paths = self.load_dataset(data_dir)
            self.img_paths.extend(img_paths)

        print(self.stage, len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def load_dataset(self, data_dir):
        """读取对应 self.stage 的数据"""
        # 读取数据
        img_paths = []
        for img_name in os.listdir(data_dir):
            if img_name.endswith(self.cfg_data['img_format']):
                img_path = os.path.join(data_dir, img_name)
                img_paths.append(img_path)

        img_paths.sort()

        return img_paths

    def process_label(self, img_path, labels):
        error = False
        res_labels = []

        for i, label in enumerate(labels):
            res_label = {}
            cls_idx = int(label[0])
            try:
                res_label['cls_name'] = self.classes[cls_idx]
            except:
                print(img_path, labels)
                error = True
            res_label['target_id'] = i + 1
            res_label['distance'] = np.nan
            res_label['is_above_horizon'] = -1
            res_label['cx'] = float(label[1])
            res_label['cy'] = float(label[2])
            res_label['w'] = float(label[3])
            res_label['h'] = float(label[4])

            res_labels.append(res_label)

        return res_labels, error

    def gamma_and_normalize(self, res, cfg, gamma_flag=False):
        """gamma变换和归一化操作"""
        gamma_aug = 2 ** np.random.normal(cfg['gamma_loc'], cfg['gamma_scale'])

        for k in list(res.keys()):
            if 'image' in k:
                if len(res[k].shape) == 2:
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / self.cfg_data['max_pixel']).float().unsqueeze(
                        -1)
                else:
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / self.cfg_data['max_pixel']).float()

                if self.is_training and gamma_flag:
                    res[k] = torch.pow(res[k], gamma_aug)
                res[k] = res[k].permute(2, 0, 1)

            elif isinstance(res[k], np.ndarray):
                res[k] = torch.from_numpy(res[k].astype(np.float32)).float()

        return res

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(self.cfg_data['img_format'], self.cfg_data['gt_format'])
        label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), self.cfg_data['label_folder'], label_name)

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            return self.__getitem__(idx + 1)

        # 读取图片
        if self.img_read_method == 'gray':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif self.img_read_method == 'unchanged':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(img_path)

        input_img = check_and_change_img_size(img, self.img_size_w, self.img_size_h)

        # 读取标签
        with open(label_path, 'r') as file:
            labels = file.readlines()
        labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的

        labels, error = self.process_label(img_path, labels)
        if error:
            return self.__getitem__(idx + 1)

        '''高斯、下采样，制作标签'''
        res = render_y(self.cfg_data, labels, img_w=self.img_size_w, img_h=self.img_size_h,
                       down_scale=self.down_scale, target_min_size=self.target_min_size)

        res['image'] = input_img
        # res['labels'] = [labels]
        res['img_path'] = img_path

        if self.img_read_method == 'admix':
            input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_gray = np.stack([input_img_gray] * 3, axis=-1)
            res['image_rgb'] = input_img
            res['image_gray'] = input_img_gray

            img_path_filter = img_path.replace('images', 'images_filtered')
            if os.path.exists(img_path_filter):
                input_img_filter = cv2.imread(img_path_filter)
                input_img_filter = check_and_change_img_size(input_img_filter, self.img_size_w, self.img_size_h)
                res['image_filter'] = input_img_filter

        # 是否返回torch_tensors
        if self.return_torch_tensors:
            # gamma转换和归一化
            res = self.gamma_and_normalize(res, cfg=self.dataset_params, gamma_flag=self.gamma_flag)

        return res


if __name__ == '__main__':
    dataset_yaml = '../config/dataset_dabaige.yaml'
    train_yaml = '../config/hrnet_train_dabaige.yaml'

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]

    # 合并读取若干个yaml的配置文件内容
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    common_utils.reprod_init(seed=cfg_data['seed'])

    # base_dataset = BaseDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)

    dataset = DetectDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)

    dataloader = DataLoader(
        dataset,
        num_workers=0,
        shuffle=False,
        batch_size=1,
    )

    for i, batch in enumerate(dataloader):
        print(f"Processing {i + 1}/{len(dataset)}")
        # if i < 3:  # 查看前3个样本
        # 获取数据和元数据
        img_tensor = batch['image'][0]
        labels = batch['labels'][0]

        # 转换为numpy并反归一化
        max_pixel = cfg_data['max_pixel']  # 从配置读取最大值（通常为255）
        img_np = img_tensor.numpy().transpose(1, 2, 0) * max_pixel
        img_np = img_np.astype(np.uint8)

        # 转换颜色空间（根据实际存储格式调整）
        if dataset.img_read_method == 'gray':
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_np = cv2.resize(img_np, (1920, 1080))

        # 绘制边界框和标签
        h, w = img_np.shape[:2]
        for label in labels:
            # 解析归一化坐标
            cx = label['cx'] * w
            cy = label['cy'] * h
            box_w = label['w'] * w
            box_h = label['h'] * h

            # 计算边界坐标
            x1 = int(cx - box_w / 2)
            y1 = int(cy - box_h / 2)
            x2 = int(cx + box_w / 2)
            y2 = int(cy + box_h / 2)

            # 绘制矩形
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加类别标签
            text = f"{label['cls_name']}"
            # cv2.putText(img_np, text, (x1 + 2, y1 + 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Augmentation Demo', img_np)
        # cv2.imshow(f'{batch["img_path"]}', img_np)
        # cv2.waitKey()
        if cv2.waitKey(2000) == 27:
            break
