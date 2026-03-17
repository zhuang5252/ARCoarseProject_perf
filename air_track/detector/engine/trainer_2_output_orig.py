import copy
import os
import shutil
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from air_track.detector.data.dataset import DetectDataset
from air_track.detector.model.model import Model
from air_track.utils import loss, plt_feature_map
from air_track.schedulers import optimize_lr, CosineAnnealingWarmRestarts
from air_track.utils.train_log import TensorboardLogger, log_configurator


class Trainer:
    def __init__(self, cfg, yaml_list):
        self.cfg = cfg
        self.yaml_list = yaml_list
        self.model_params = cfg['model_params']
        self.train_params = cfg['train_params']
        self.classes = cfg['classes']
        self.img_read_method = cfg['dataset_params']['img_read_method'].lower()
        self.down_scale = cfg['dataset_params']['down_scale']
        self.target_min_size = cfg['dataset_params']['target_min_size']
        # 设置gpu设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg['cuda_device'])
        print('cuda_device: ', os.environ["CUDA_VISIBLE_DEVICES"])

        # DDP 训练标记
        if torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ:
            self.distributed = True
        else:
            self.distributed = False

        if self.distributed:
            # 如果是分布式，则初始化进程组
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

        # 设备
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        self.train_dl = None
        self.valid_dl = None
        self.train_sampler = None
        self.valid_sampler = None

        self.model = None
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        self.grad_clip_value = self.train_params.get('grad_clip', 8.0)  # 梯度裁减
        self.epoch_minimum_loss = 1000000
        self.patience = self.train_params.get('patience', self.train_params['scheduler_period'])
        self.min_delta = self.train_params.get('min_delta', 0)
        self.no_improvement_count = 0
        self.early_stop = False

        # 创建模型保存路径
        time_str_ymd = time.strftime('%Y_%m_%d_%H')
        model_save_root = cfg['model_save_root']
        self.model_save_path = os.path.join(
            model_save_root, self.model_params['backbone_type'], self.model_params['backbone_name'],
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
        dataset_train = DetectDataset(stage=self.cfg['stage_train'], cfg_data=self.cfg)

        dataset_valid = DetectDataset(stage=self.cfg['stage_valid'], cfg_data=self.cfg)

        # 多卡情况下使用DistributedSampler
        if self.distributed:
            self.train_sampler = DistributedSampler(
                dataset_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.valid_sampler = DistributedSampler(
                dataset_valid,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )

            self.train_dl = DataLoader(
                dataset_train,
                num_workers=self.cfg['train_data_loader']['num_workers'],
                batch_size=self.cfg['train_data_loader']['batch_size'],
                pin_memory=True,
                sampler=self.train_sampler,
            )
            self.valid_dl = DataLoader(
                dataset_valid,
                num_workers=self.cfg['val_data_loader']['num_workers'],
                batch_size=self.cfg['val_data_loader']['batch_size'],
                pin_memory=True,
                sampler=self.valid_sampler,
            )
        else:
            # 单卡直接用Dataloader
            self.train_dl = DataLoader(
                dataset_train,
                num_workers=self.cfg['train_data_loader']['num_workers'],
                batch_size=self.cfg['train_data_loader']['batch_size'],
                pin_memory=True,
                shuffle=True
            )
            self.valid_dl = DataLoader(
                dataset_valid,
                num_workers=self.cfg['val_data_loader']['num_workers'],
                batch_size=self.cfg['val_data_loader']['batch_size'],
                pin_memory=True,
                shuffle=False
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
            'mask_loss': mask_loss
        }

        return self.loss_fn

    def calc_loss(self, label, pred, epoch):
        """使用标签和模型输出计算loss"""
        mask = label['mask'].float().to(self.device)
        mask_planned = label['mask_planned'].float().to(self.device)
        reg_size = label['reg_size'].float().to(self.device)
        reg_size_mask = label['reg_size_mask'].float().to(self.device)

        if 'loss_scale' in self.train_params:
            loss_items_scale = self.train_params['loss_scale']
        else:
            loss_items_scale = {
                'mask': 2000,
                'cls': 1,
                'size': 1,
                'offset': 0.2,
                'distance': 1,
                'tracking': 1,
            }

        # mask损失
        loss_mask_all = self.loss_fn['mask_loss'](pred['mask'], 1.0 - mask)
        loss_mask_planned = self.loss_fn['mask_loss'](pred['mask'], 1.0 - mask_planned)
        if epoch > 75:
            loss_mask = torch.minimum(loss_mask_all, loss_mask_planned).mean()
        else:
            loss_mask = loss_mask_all.mean()

        # 尺寸预测
        reg_size_mask = reg_size_mask[:, None, :, :]
        mask_size = reg_size_mask.sum() + 0.1
        loss_size = self.loss_fn['cr_mse'](
            pred['size'] * reg_size_mask,
            reg_size * reg_size_mask
        ).sum() / mask_size

        losses = [
            ('mask', loss_mask),
            ('size', loss_size)
        ]
        loss = sum([loss_value * loss_items_scale[loss_name] for loss_name, loss_value in losses])

        mask_loss_item = loss_mask.detach().item()
        size_loss_item = loss_size.detach().item()
        total_loss_item = loss.detach().item()

        res_loss_item = {
            'mask': mask_loss_item,
            'size': size_loss_item,
            'total_loss': total_loss_item
        }
        return loss, res_loss_item

    def train(self, epoch):
        if self.distributed:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        index = 0
        num_batches = len(self.train_dl)
        train_mask_loss = 0
        train_size_loss = 0
        train_total_loss = 0

        # 仅 rank=0 打印
        if self.is_rank0 and self.logger:
            size = len(self.train_dl.dataset)
            self.logger.info(f'训练集大小:{size}\t 训练batch数量:{num_batches}')

        for data in self.train_dl:
            with torch.set_grad_enabled(True):
                input_images = data['image'].to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(False):
                    pred = self.model(input_images)
                    pred = {
                        'mask': pred[0],
                        'size': pred[1],
                    }
                    loss, res_loss_item = self.calc_loss(data, pred, epoch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_mask_loss += res_loss_item['mask']
                train_size_loss += res_loss_item['size']
                train_total_loss += res_loss_item['total_loss']

                if self.is_rank0 and self.logger and (index % 10 == 0):
                    self.logger.info(
                        f'Epoch:{epoch} 第[{index}/{num_batches}]次迭代loss:'
                        f'mask_loss={res_loss_item["mask"]}, '
                        f'size_loss={res_loss_item["size"]}, total_loss={res_loss_item["total_loss"]}'
                    )
                    # 记录特征图
                    self.plot_tensorboard_feature_map('train', pred, epoch)

                index += 1

        # 计算 epoch 均值
        train_mask_loss /= num_batches
        train_size_loss /= num_batches
        train_total_loss /= num_batches

        train_loss_mean = [
            ('train_mask_loss', train_mask_loss),
            ('train_size_loss', train_size_loss),
            ('train_total_loss', train_total_loss)
        ]

        # 仅 rank=0 写 tensorboard
        if self.is_rank0 and self.tensorboard_logger:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            tensorboard_log = [
                ("train/train_mask_loss", train_mask_loss),
                ("train/train_size_loss", train_size_loss),
                ("train/train_total_loss", train_total_loss),
                ("train/lr", lr)
            ]
            self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        return train_loss_mean

    def valid(self, epoch):
        if self.distributed:
            self.valid_sampler.set_epoch(epoch)

        self.model.eval()

        index = 0
        num_batches = len(self.valid_dl)
        test_mask_loss = 0
        test_size_loss = 0
        test_total_loss = 0

        if self.is_rank0 and self.logger:
            size = len(self.valid_dl.dataset)
            self.logger.info(f'验证集大小:{size}\t 验证batch数量:{num_batches}')

        for data in self.valid_dl:
            with torch.set_grad_enabled(False):
                input_images = data['image'].to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(False):
                    pred = self.model(input_images)
                    pred = {
                        'mask': pred[0],
                        'size': pred[1],
                    }
                    loss, res_loss_item = self.calc_loss(data, pred, epoch)

                test_mask_loss += res_loss_item['mask']
                test_size_loss += res_loss_item['size']
                test_total_loss += res_loss_item['total_loss']

                if self.is_rank0 and self.logger and (index % 10 == 0):
                    self.logger.info(
                        f'Epoch:{epoch} 第[{index}/{num_batches}]次验证loss:'
                        f'mask_loss={res_loss_item["mask"]}, '
                        f'size_loss={res_loss_item["size"]}, total_loss={res_loss_item["total_loss"]}'
                    )
                    # 记录特征图
                    self.plot_tensorboard_feature_map('test', pred, epoch)

                index += 1

        test_mask_loss /= num_batches
        test_size_loss /= num_batches
        test_total_loss /= num_batches

        test_loss_mean = [
            ('test_mask_loss', test_mask_loss),
            ('test_size_loss', test_size_loss),
            ('test_total_loss', test_total_loss)
        ]

        if self.is_rank0 and self.tensorboard_logger:
            tensorboard_log = [
                ("test/test_mask_loss", test_mask_loss),
                ("test/test_size_loss", test_size_loss),
                ("test/test_total_loss", test_total_loss)
            ]
            self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        # 仅 rank=0 进程保存模型
        if self.is_rank0 and self.logger:
            # 保存第一个epoch模型
            if epoch == 1:
                for yaml_file in self.yaml_list:
                    shutil.copy(
                        yaml_file,
                        os.path.join(self.model_save_path, os.path.basename(yaml_file))
                    )

            # 简单 early stopping，保存loss最小的模型
            valid_loss_min = test_loss_mean[-1][1]
            if valid_loss_min < self.epoch_minimum_loss - self.min_delta:
                self.epoch_minimum_loss = valid_loss_min
                self.no_improvement_count = 0
                # 也可以在这里保存模型
                self.logger.info(
                    f'minimum_validate_loss = {self.epoch_minimum_loss:.5f}, '
                    f'saving model to minimum_loss.pt'
                )
                self.save_model('minimum_loss.pt', epoch)
            else:
                self.no_improvement_count += 1
                self.logger.info(f"No improvement count: {self.no_improvement_count}")

            if self.no_improvement_count >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                self.early_stop = True

        return test_loss_mean

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
