# -*- coding: utf-8 -*-
# @Author    : 
# @File      : base_trainer.py
# @Created   : 2025/7/1 下午2:55
# @Desc      : 一级检测器trainer基类
import os
import time
import copy
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from air_track.detector.model.model import Model
from air_track.utils import loss, plt_feature_map
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from air_track.detector.data.dataset import DetectDataset
from air_track.schedulers import optimize_lr, CosineAnnealingWarmRestarts
from air_track.utils.train_log import log_configurator, TensorboardLogger
from air_track.detector.utils.detect_utils import combine_images


class BaseTrainer:
    def __init__(self, cfg, yaml_list):
        self.cfg = cfg
        self.yaml_list = yaml_list
        self.dataset_params = cfg['dataset_params']
        self.model_params = cfg['model_params']
        self.train_params = cfg['train_params']

        # 数据集参数
        self.classes = cfg['classes']
        self.input_frames = self.dataset_params['input_frames']
        self.img_read_method = self.dataset_params['img_read_method'].lower()
        self.down_scale = self.dataset_params['down_scale']
        self.target_min_size = self.dataset_params['target_min_size']

        # 设置gpu设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg['cuda_device'])
        print('cuda_device: ', os.environ["CUDA_VISIBLE_DEVICES"])

        # DDP 训练标记
        if torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ:
            self.distributed = True
        else:
            self.distributed = False

        if self.distributed:
            # 分布式初始化
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_rank0 = (self.rank == 0)
        else:
            # 单卡模式下，把 rank/world_size/local_rank 都设为默认值
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.is_rank0 = True  # 单卡默认就是 rank0

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        self.dataset_train = None
        self.dataset_valid = None
        self.train_dl = None
        self.valid_dl = None
        self.train_sampler = None
        self.valid_sampler = None

        self.model = None
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        # 训练参数
        self.grad_clip_value = self.train_params.get('grad_clip', 8.0)  # 梯度裁减
        self.epoch_minimum_loss = 1000000
        self.patience = self.train_params.get('patience', self.train_params['scheduler_period'])
        self.min_delta = self.train_params.get('min_delta', 0)
        self.return_feature_map = self.model_params.get('return_feature_map', False)
        self.similarity_learning = self.train_params.get('similarity_learning', False)
        self.plot_tensorboard_feature = self.train_params.get('plot_tensorboard_feature', False)
        self.plot_tensorboard_feature_period = int(self.train_params.get('plot_tensorboard_feature_period', 10000))
        self.no_improvement_count = 0
        self.loss_scale = None
        self.early_stop = False

        # 创建模型保存路径
        time_str_ymd = time.strftime('%Y_%m_%d_%H')
        model_save_root = cfg['model_save_root']
        self.model_save_path = os.path.join(
            model_save_root, self.model_params['model_type'],
            self.model_params['backbone_type'], self.model_params['backbone_name'],
            self.model_params['base_model_name'], time_str_ymd
        )
        if self.is_rank0:  # 仅在rank=0上创建文件夹
            os.makedirs(self.model_save_path, exist_ok=True)

        if cfg['detect_log'] != str(None) and cfg['detect_log'] != model_save_root:
            log_dir = os.path.join(cfg['detect_log'], time_str_ymd)
        else:
            log_dir = self.model_save_path

        # 仅rank=0的进程写日志
        if self.is_rank0:
            self.logger, self.log_file = log_configurator(log_dir)
            self.tensorboard_logger = TensorboardLogger(log_dir=log_dir, log_hist=False)
            print('Model Save Path: ', self.model_save_path)
        else:
            self.logger, self.log_file = None, None
            self.tensorboard_logger = None

    def set_dataset(self):
        """构建数据"""
        self.dataset_train = DetectDataset(stage=self.cfg['stage_train'], cfg_data=self.cfg)

        self.dataset_valid = DetectDataset(stage=self.cfg['stage_valid'], cfg_data=self.cfg)

        # 多卡情况下使用DistributedSampler
        if self.distributed:
            self.train_sampler = DistributedSampler(
                self.dataset_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.valid_sampler = DistributedSampler(
                self.dataset_valid,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )

            self.train_dl = DataLoader(
                self.dataset_train,
                num_workers=self.cfg['train_data_loader']['num_workers'],
                batch_size=self.cfg['train_data_loader']['batch_size'],
                pin_memory=True,
                sampler=self.train_sampler,
                collate_fn=lambda x: x
            )
            self.valid_dl = DataLoader(
                self.dataset_valid,
                num_workers=self.cfg['val_data_loader']['num_workers'],
                batch_size=self.cfg['val_data_loader']['batch_size'],
                pin_memory=True,
                sampler=self.valid_sampler,
                collate_fn=lambda x: x
            )
        else:
            # 单卡直接用Dataloader
            self.train_dl = DataLoader(
                self.dataset_train,
                num_workers=self.cfg['train_data_loader']['num_workers'],
                batch_size=self.cfg['train_data_loader']['batch_size'],
                pin_memory=True,
                shuffle=True,
                collate_fn=lambda x: x
            )
            self.valid_dl = DataLoader(
                self.dataset_valid,
                num_workers=self.cfg['val_data_loader']['num_workers'],
                batch_size=self.cfg['val_data_loader']['batch_size'],
                pin_memory=True,
                shuffle=False,
                collate_fn=lambda x: x
            )

        return self.train_dl, self.valid_dl

    def set_model(self):
        """定义模型"""
        model_params = self.cfg['model_params']
        base_model = Model(
            cfg=model_params,
            pretrained=model_params['pretrained']
        ).to(self.device)

        # 多卡：用DDP包裹
        if self.distributed:
            self.model = DDP(
                base_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        else:
            # 单卡：直接使用普通模型
            self.model = base_model
        return self.model

    def set_optimizer_and_scheduler(self):
        """定义优化器和学习率调度器"""
        self.scaler = torch.cuda.amp.GradScaler()
        initial_lr = float(self.train_params['initial_lr'])

        # DDP 情况下，只需要拿 self.model.parameters() 即可
        self.optimizer = optimize_lr.optimize_madgrad(
            self.model.parameters(),
            lr=initial_lr
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.train_params['scheduler_period'],
            T_mult=self.train_params.get('scheduler_t_mult', 1),
            eta_min=initial_lr / 1000.0,
            last_epoch=-1,
            first_epoch_lr_scale=0.01
        )

        return self.optimizer, self.scheduler

    def set_loss_fn(self):
        """定义损失函数"""
        mask_loss_name = self.train_params.get('mask_loss', 'bce')  # fl
        cr_mse = torch.nn.MSELoss(reduction='none')  # 均方误差
        cr_mae = torch.nn.L1Loss(reduction='none')  # 绝对值损失
        cr_bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # 结合 Sigmoid 函数和二进制交叉熵损失函数
        cr_fl = loss.FocalLossV2(reduction='none')  # FocalLoss：平衡正负样本和难易样本
        cr_kl = torch.nn.KLDivLoss(reduction='batchmean')
        cr_sim = loss.SimilarityLoss(alpha=1.0, delta=1.0, normalize=True)

        if mask_loss_name == 'bce':
            mask_loss = cr_bce
        elif mask_loss_name == 'fl':
            mask_loss = cr_fl
        else:
            raise RuntimeError('Invalid mask loss name ' + mask_loss_name)
        self.loss_fn = {  # 将整个loss存入字典
            'cr_mse': cr_mse,
            'cr_mae': cr_mae,
            'cr_bce': cr_bce,
            'cr_fl': cr_fl,
            'cr_kl': cr_kl,
            'cr_sim': cr_sim,
            'mask_loss': mask_loss
        }

        return self.loss_fn

    def set_loss_scale(self):
        if 'loss_scale' in self.train_params:
            self.loss_scale = self.train_params['loss_scale']
        else:
            self.loss_scale = {
                'mask': 2000,
                'cls': 1,
                'size': 1,
                'offset': 0.2,
                'distance': 1,
                'tracking': 1,
                'sim': 1
            }

        return self.loss_scale

    def read_prev_data(self, data, prev_key):
        """读取前序帧数据"""
        prev_images = []
        for i in range(self.model_params['input_frames'] - 1):
            prev_images.append(data[f'{prev_key}{i}'])
        return prev_images

    def check_unfreeze_stage(self, epoch):
        """检查并执行阶段解冻"""
        # 获取实际模型对象（处理DDP包装情况）
        model = self.model.module if self.distributed else self.model

        # 检查是否配置了渐进式解冻
        if not hasattr(model, 'freeze_config'):
            return

        if model.freeze_config.get('freeze_strategy') != 'auto':
            return

        # 检查解冻时机
        unfreeze_epochs = model.freeze_config.get('unfreeze_epochs', [])
        if epoch in unfreeze_epochs:
            if self.is_rank0 and self.logger:
                self.logger.info(f'Epoch {epoch}: Attempting to unfreeze next stage')

            # 执行解冻
            if self.distributed:
                state = self.model.module.unfreeze_next_stage(self.logger)
            else:
                state = self.model.unfreeze_next_stage(self.logger)

            if self.is_rank0 and self.logger:
                if state:
                    self.logger.info(f'Unfreeze successful! epoch: {epoch}')
                else:
                    self.logger.info('Unfreeze failed! epoch: {epoch}')

            # 解冻后重置学习率
            self.scheduler.step()

    def process_input(self, data):
        """模型输入数据前处理"""
        # 对比学习
        if self.similarity_learning and self.return_feature_map and self.img_read_method == 'admix':
            if self.input_frames == 1:
                images_rgb = data['cur_image_rgb'].to(self.device)
                images_gray = data['cur_image_gray'].to(self.device)

                # 判断是否存在image_filter的数据
                matching_keys = [key for key in data.keys() if 'image_filter' in key]
                # 拼接两个分支（在 batch 维度上拼接）
                if matching_keys:
                    images_filter = data['cur_image_filter'].to(self.device)
                    input_images = torch.cat([images_rgb, images_gray, images_filter], dim=0)  # shape: [3B,C,H,W]
                    # input_images = torch.cat([images_rgb, images_filter], dim=0)  # shape: [2B,C,H,W]
                else:
                    input_images = torch.cat([images_rgb, images_gray], dim=0)  # shape: [2B, C, H, W]
            elif self.input_frames > 1:
                cur_image_rgb = data['cur_image_rgb']
                prev_images_rgb = self.read_prev_data(data, prev_key='prev_image_rgb')
                input_images_rgb = combine_images(prev_images_rgb, cur_image_rgb)
                cur_image_gray = data['cur_image_gray']
                prev_images_gray = self.read_prev_data(data, prev_key='prev_image_gray')
                input_images_gray = combine_images(prev_images_gray, cur_image_gray)

                # 判断是否存在image_filter的数据
                matching_keys = [key for key in data.keys() if 'image_filter' in key]
                # 拼接两个分支（在 batch 维度上拼接）
                if matching_keys:
                    cur_image_filter = data['cur_image_filter']
                    prev_images_filter = self.read_prev_data(data, prev_key='prev_image_filter')
                    input_images_filter = combine_images(prev_images_filter, cur_image_filter)
                    input_images = torch.cat([input_images_rgb, input_images_gray, input_images_filter],
                                             dim=0)  # shape: [3B,C,H,W]
                else:
                    input_images = torch.cat([input_images_rgb, input_images_gray], dim=0)  # shape: [2B, C, H, W]
            else:
                raise ValueError('Invalid input_frames value')
        # 非对比学习
        else:
            if self.input_frames == 1:
                input_images = data['cur_image']
            elif self.input_frames > 1:
                cur_image = data['cur_image']
                prev_images = self.read_prev_data(data, prev_key='prev_image_aligned')
                input_images = combine_images(prev_images, cur_image)
            else:
                raise ValueError('Invalid input_frames value')

        input_images = input_images.float().to(self.device)

        return input_images

    def calc_sim_loss(self, batch_size, features=None):
        """对比学习时，计算模型输出特征的sim loss"""
        # 主任务前向：使用 RGB 图像进行预测
        if self.similarity_learning and self.img_read_method == 'admix':
            # 对中间特征也进行分割
            if features.shape[0] == 2 * batch_size:
                features_rgb = features[:batch_size]
                features_2 = features[batch_size:]
                # 相似性损失：比较 RGB 与灰度分支的中间特征
                loss_sim = self.loss_fn['cr_sim']((features_rgb, features_2)) * self.loss_scale['sim']
            elif features.shape[0] == 3 * batch_size:
                features_rgb = features[:batch_size]
                features_gray = features[batch_size: 2 * batch_size]
                features_filter = features[2 * batch_size:]
                # 相似性损失：比较 RGB 与灰度分支的中间特征
                loss_sim_1 = self.loss_fn['cr_sim']((features_rgb, features_gray))
                loss_sim_2 = self.loss_fn['cr_sim']((features_rgb, features_filter))
                loss_sim_3 = self.loss_fn['cr_sim']((features_gray, features_filter))
                loss_sim = (loss_sim_1 + loss_sim_2 + loss_sim_3) * self.loss_scale['sim']
            else:
                raise ValueError('Invalid features shape')
        else:
            loss_sim = None

        return loss_sim

    def load_checkpoint(self, checkpoint_path):
        """用于继续训练"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint['epoch'] > 0:
            print('Continue model: ', checkpoint_path)
            if self.distributed:
                self.model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        else:
            self.set_model()
        return checkpoint['epoch']

    def save_model(self, model_name, epoch):
        """
        仅在 rank=0 的进程上调用。
        """
        if not self.is_rank0:
            return
        if self.distributed:
            base_model = self.model.module
        else:
            base_model = self.model
        epoch_model = copy.deepcopy(base_model)

        save_dict = {
            "epoch": epoch,
            "model_state_dict": epoch_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        torch.save(save_dict, os.path.join(self.model_save_path, model_name))

    def plot_tensorboard_feature_map(self, tag, features, epoch):
        """在tensorboard中记录特征图"""
        for key in features:
            for i in range(features[key].shape[1]):
                feature_map = features[key][-1:, i:i + 1, :, :]
                feature = plt_feature_map(tag, feature_map)
                self.tensorboard_logger.add_figure(f'{tag}/{key}_channel_{i}', feature, epoch)

    def __del__(self):
        if self.is_rank0 and self.tensorboard_logger:
            self.tensorboard_logger.writer.close()


if __name__ == '__main__':
    pass
