import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from air_track.detector.model.model import Model
from air_track.schedulers import optimize_lr, CosineAnnealingWarmRestarts
from air_track.detector.utils.detect_utils import combine_images
from air_track.utils import combine_load_cfg_yaml, loss


class PlModel(pl.LightningModule):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model_params = cfg['model_params']
        self.dataset_params = cfg['dataset_params']
        self.train_params = cfg['train_params']

        self.cls_num = self.model_params['nb_classes']
        self.img_read_method = self.dataset_params['img_read_method'].lower()
        self.similarity_learning = self.model_params['similarity_learning']

        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        self.model = Model(self.model_params, pretrained=pretrained)

        # 创建模型保存路径
        time_str_ymd = time.strftime('%Y_%m_%d_%H')
        model_save_root = cfg['model_save_root']
        self.model_save_path = os.path.join(
            model_save_root, self.model_params['backbone_type'], self.model_params['backbone_name'],
            self.model_params['base_model_name'], time_str_ymd
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        if cfg['detect_log'] != str(None) and cfg['detect_log'] != model_save_root:
            self.log_dir = os.path.join(cfg['detect_log'], time_str_ymd)
        else:
            self.log_dir = self.model_save_path

    def forward(self, inputs):
        output = self.model(inputs)

        return output

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

    def configure_optimizers(self):
        """定义优化器和学习率调度器"""
        self.scaler = torch.cuda.amp.GradScaler()
        initial_lr = float(self.train_params['initial_lr'])

        # DDP 情况下，只需要拿 self.model.parameters() 即可
        self.optimizer = optimize_lr.optimize_madgrad(
            self.parameters(),
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

        return [self.optimizer], [self.scheduler]

    def calc_loss(self, label, pred):
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

    def read_prev_data(self, data):
        """读取多帧数据示例"""
        prev_images = []
        for i in range(self.model_params['input_frames'] - 1):
            prev_images.append(data[f'prev_image_aligned{i}'])
        return prev_images

    def shared_step(self, batch, stage):
        if self.img_read_method == 'admix':
            images_rgb = batch['image_rgb'].to(self.device)
            images_gray = batch['image_gray'].to(self.device)
            B = images_rgb.size(0)

            # 拼接两个分支（在 batch 维度上拼接）
            if 'image_filter' in batch:
                images_filter = batch['image_filter'].to(self.device)
                input_images = torch.cat([images_rgb, images_gray, images_filter], dim=0)  # shape: [2B,C,H,W]
                # input_images = torch.cat([images_rgb, images_filter], dim=0)  # shape: [2B,C,H,W]
            else:
                input_images = torch.cat([images_rgb, images_gray], dim=0)  # shape: [2B, C, H, W]
        else:
            input_images = batch['image'].to(self.device)
            B = input_images.size(0)

        self.optimizer.zero_grad()

        loss_sim = None
        # 主任务前向：使用 RGB 图像进行预测
        if self.similarity_learning:
            pred, features = self(input_images)
            # 对中间特征也进行分割
            if features.shape[0] == 2 * B:
                features_rgb = features[:B]
                features_2 = features[B:]
                # 相似性损失：比较 RGB 与灰度分支的中间特征
                loss_sim = self.loss_fn['cr_sim']((features_rgb, features_2)) * self.train_params[
                    'loss_scale']['sim']
            else:
                features_rgb = features[:B]
                features_gray = features[B: 2 * B]
                features_filter = features[2 * B:]
                # 相似性损失：比较 RGB 与灰度分支的中间特征
                loss_sim_1 = self.loss_fn['cr_sim']((features_rgb, features_gray))
                loss_sim_2 = self.loss_fn['cr_sim']((features_rgb, features_filter))
                loss_sim_3 = self.loss_fn['cr_sim']((features_gray, features_filter))
                loss_sim = loss_sim_1 + loss_sim_2 + loss_sim_3
        else:
            pred = self(input_images)

        # 将模型输出分割为 RGB 与灰度或者其他部分
        # 对于主任务损失，我们仅使用前 B 个batch的预测
        pred = [tensor[:B] for tensor in pred]  # 例如 [mask, size]
        pred = {
            'mask': pred[0],
            'size': pred[1],
        }

        # 主任务损失（例如 mask、size 损失），用原始标签 data 计算
        loss_main, res_loss_item = self.calc_loss(batch, pred)

        # 总Loss
        if loss_sim is not None:
            loss = loss_main + loss_sim
        else:
            loss = loss_main

        if loss_sim is None:
            loss_sim = 0

        self.log(f'{stage}/{stage}_mask_loss', res_loss_item['mask'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/{stage}_size_loss', res_loss_item['size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/{stage}_sim_loss', loss_sim,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/{stage}_total_loss', res_loss_item['total_loss'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss, pred

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics["train/total_loss"].item()
        self.logger.experiment.add_scalar("epoch/train_total_loss", loss, self.current_epoch)

    def training_step(self, batch):
        train_total_loss, prediction = self.shared_step(batch, stage='train')
        # self.on_after_backward()

        return train_total_loss

    def validation_step(self, batch):
        val_total_loss, prediction = self.shared_step(batch, stage='val')

        return val_total_loss

    # def on_after_backward(self):
    #     # 记录梯度范数
    #     total_norm = torch.norm(
    #         torch.stack([p.grad.norm() for p in self.parameters()]),
    #         p=2
    #     )
    #     self.log("grad_norm", total_norm)


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_yaml = os.path.join(script_dir, 'config/hrnet_train_aot_resnet50_new.yaml')

    # 读取yaml文件
    yaml_list = [train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)
    model = PlModel(cfg_data['model_params'])
    res = model(torch.zeros(2, 2, 512, 640))
