import copy
import os
import shutil
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from air_track.aligner.data.dataset import AlignDataset
from air_track.aligner.model import resnet34_return_dx_dy_angle as resnet34
from air_track.schedulers import optimize_lr, CosineAnnealingWarmRestarts
from air_track.utils.train_log import log_configurator, TensorboardLogger


class Trainer:
    def __init__(self, cfg, yaml_list):
        self.cfg = cfg
        self.yaml_list = yaml_list
        self.model_params = cfg['model_params']
        self.train_params = cfg['train_params']
        self.device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.train_dl = None
        self.valid_dl = None
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

        self.distributed = False

        # 创建模型保存路径
        time_str_ymd = time.strftime('%Y_%m_%d_%H')
        model_save_root = cfg['model_save_root']
        self.model_save_path = os.path.join(
            model_save_root, self.model_params['model_type'],
            self.model_params['model_cls'], self.model_params['base_model_name'], time_str_ymd
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        if cfg['align_log'] != str(None) and cfg['align_log'] != model_save_root:
            log_dir = os.path.join(cfg['align_log'], time_str_ymd)
        else:
            log_dir = self.model_save_path
        self.logger, self.log_file = log_configurator(log_dir)
        self.tensorboard_logger = TensorboardLogger(log_dir=log_dir, log_hist=False)
        print('Model Save Path: ', self.model_save_path)

    def set_dataset(self):
        """构建数据"""
        dataset_train = AlignDataset(stage=self.cfg['stage_train'], cfg_data=self.cfg)

        dataset_valid = AlignDataset(stage=self.cfg['stage_valid'], cfg_data=self.cfg)

        self.train_dl = DataLoader(dataset_train,
                                   num_workers=self.cfg['train_data_loader']['num_workers'],
                                   shuffle=True,
                                   batch_size=self.cfg['train_data_loader']['batch_size'],
                                   )
        self.valid_dl = DataLoader(dataset_valid,
                                   num_workers=self.cfg['val_data_loader']['num_workers'],
                                   shuffle=False,
                                   batch_size=self.cfg['val_data_loader']['batch_size'],
                                   )

        return self.train_dl, self.valid_dl

    def set_model(self):
        """定义模型"""
        model_params = self.cfg['model_params']
        self.model: nn.Module = resnet34.__dict__[model_params['model_cls']](cfg=model_params,
                                                                             pretrained=model_params['pretrained'])

        return self.model.to(self.device)

    def set_optimizer_and_scheduler(self):
        """定义优化器和学习率调度器"""
        # 自动混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()

        # 优化器
        initial_lr = float(self.train_params['initial_lr'])  # 1e-3
        self.optimizer = optimize_lr.optimize_madgrad(self.model.parameters(), lr=initial_lr)

        # 学习率调度器
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
        cr_mse = torch.nn.MSELoss(reduction='mean')  # 均方误差
        cr_mae = torch.nn.L1Loss(reduction='mean')  # 平均绝对误差
        # Smooth L1 Loss (Huber Loss)
        cr_sl1 = nn.SmoothL1Loss(reduction='mean')  # 对异常值更鲁棒‌
        self.loss_fn = {
            'cr_mse': cr_mse,
            'cr_mae': cr_mae,
            'cr_sl1': cr_sl1,
        }
        return self.loss_fn

    def calc_loss(self, pred, label):
        """使用标签和模型输出计算loss"""
        loss = self.loss_fn['cr_sl1'](pred, label)
        loss_item = loss.detach().item()

        return loss  # , loss_item

    def train(self, epoch):
        self.model.train()

        index = 0
        train_loss = 0
        dx_loss = 0
        dy_loss = 0
        angle_loss = 0

        size = len(self.train_dl.dataset)  # 训练集的大小
        num_batches = len(self.train_dl)  # 批次
        self.logger.info('训练集大小: {}\t 训练batch数量: {}'.format(size, num_batches))

        for data in self.train_dl:
            with torch.set_grad_enabled(True):
                cur_frame = data['cur_img'].to(self.device)
                prev_frame = data['prev_img'].to(self.device)

                dx = data['dx'].to(self.device)
                dy = data['dy'].to(self.device)
                angle = data['angle'].to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(False):  # 混合精度
                    dx_pred, dy_pred, angle_pred = self.model(prev_frame, cur_frame)
                    # 计算损失
                    loss_dx = self.calc_loss(dx_pred, dx)
                    loss_dy = self.calc_loss(dy_pred, dy)
                    loss_angle = self.calc_loss(angle_pred, angle)

                    loss_scale = self.train_params['loss_scale']
                    loss_dx = loss_scale['dx'] * loss_dx
                    loss_dy = loss_scale['dy'] * loss_dy
                    loss_angle = loss_scale['angle'] * loss_angle

                    loss = loss_dx + loss_dy + loss_angle

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # 梯度裁剪，以防止梯度爆炸
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_item = loss.detach().item()
                train_loss += loss_item
                dx_loss += loss_dx.detach().item()
                dy_loss += loss_dy.detach().item()
                angle_loss += loss_angle.detach().item()

                if index % 10 == 0:
                    self.logger.info(
                        f'Epoch:{epoch} 第[{index}/{num_batches}]次迭代, '
                        f'loss_dx={loss_dx.detach().item()}, '
                        f'loss_dy={loss_dy.detach().item()}, '
                        f'loss_angle={loss_angle.detach().item()}, '
                        f'total_loss={loss_item}\t ')
                index += 1

        train_loss /= num_batches
        dx_loss /= num_batches
        dy_loss /= num_batches
        angle_loss /= num_batches

        # 获取当前的学习率
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # tensorboard可视化
        tensorboard_log = [
            ("train/train_loss", train_loss),
            ("train/dx_loss", dx_loss),
            ("train/dy_loss", dy_loss),
            ("train/angle_loss", angle_loss),
            ("train/lr", lr)
        ]
        self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        return train_loss

    def valid(self, epoch):
        self.model.eval()

        index = 0
        test_loss = 0
        dx_loss = 0
        dy_loss = 0
        angle_loss = 0

        size = len(self.valid_dl.dataset)  # 验证集的大小
        num_batches = len(self.valid_dl)  # 批次
        self.logger.info('验证集大小: {}\t 验证batch数量: {}'.format(size, num_batches))

        for data in self.valid_dl:
            with torch.set_grad_enabled(False):
                cur_frame = data['cur_img'].to(self.device)
                prev_frame = data['prev_img'].to(self.device)

                dx = data['dx'].to(self.device)
                dy = data['dy'].to(self.device)
                angle = data['angle'].to(self.device)

                with torch.cuda.amp.autocast(False):  # 混合精度
                    dx_pred, dy_pred, angle_pred = self.model(prev_frame, cur_frame)
                    # 计算损失
                    loss_dx = self.calc_loss(dx_pred, dx)
                    loss_dy = self.calc_loss(dy_pred, dy)
                    loss_angle = self.calc_loss(angle_pred, angle)

                    loss_scale = self.train_params['loss_scale']
                    loss_dx = loss_scale['dx'] * loss_dx
                    loss_dy = loss_scale['dy'] * loss_dy
                    loss_angle = loss_scale['angle'] * loss_angle

                    loss = loss_dx + loss_dy + loss_angle

                loss_item = loss.detach().item()
                test_loss += loss_item
                dx_loss += loss_dx.detach().item()
                dy_loss += loss_dy.detach().item()
                angle_loss += loss_angle.detach().item()

                if index % 10 == 0:
                    self.logger.info(
                        f'Epoch:{epoch} 第[{index}/{num_batches}]次迭代, '
                        f'loss_dx={loss_dx.detach().item()}, '
                        f'loss_dy={loss_dy.detach().item()}, '
                        f'loss_angle={loss_angle.detach().item()}, '
                        f'total_loss={loss_item}\t ')
                index += 1

        test_loss /= num_batches
        dx_loss /= num_batches
        dy_loss /= num_batches
        angle_loss /= num_batches

        # tensorboard可视化
        tensorboard_log = [
            ("test/test_loss", test_loss),
            ("test/dx_loss", dx_loss),
            ("test/dy_loss", dy_loss),
            ("test/angle_loss", angle_loss)
        ]
        self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        # 保存第一个epoch模型
        if epoch == 1:
            for yaml_file in self.yaml_list:
                shutil.copy(yaml_file, os.path.join(self.model_save_path, os.path.basename(yaml_file)))

        # 简单 early stopping，保存loss最小的模型
        valid_loss_min = test_loss
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

        return test_loss

    def load_checkpoint(self, model_path, continue_epoch: int = -1):
        """用于继续训练"""
        checkpoint = torch.load(model_path, map_location='cpu')
        if continue_epoch > 0:
            print('Continue model: ', model_path)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            return self.set_model()

    def save_model(self, model_name, epoch):
        epoch_model = copy.deepcopy(self.model)

        torch.save({
            "epoch": epoch,
            "model_state_dict": epoch_model.module.state_dict() if self.distributed else epoch_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(), },
            f"{self.model_save_path}/{model_name}", )
