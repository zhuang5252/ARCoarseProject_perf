import gc
import os
import torch
import shutil
import torch.distributed as dist
from air_track.detector.engine.base_trainer import BaseTrainer
from air_track.detector.utils.detect_utils import merge_dataset


class Trainer(BaseTrainer):
    def __init__(self, cfg, yaml_list):
        super().__init__(cfg, yaml_list)

    def calc_loss(self, label, pred, epoch):
        """使用标签和模型输出计算loss"""
        mask = label['mask'].float().to(self.device)
        mask_planned = label['mask_planned'].float().to(self.device)
        label_cls = label['cls'].float().to(self.device)
        reg_size = label['reg_size'].float().to(self.device)
        reg_size_mask = label['reg_size_mask'].float().to(self.device)

        # mask损失
        loss_mask_all = self.loss_fn['mask_loss'](pred['mask'], 1.0 - mask)
        loss_mask_planned = self.loss_fn['mask_loss'](pred['mask'], 1.0 - mask_planned)
        if epoch > 75:
            loss_mask = torch.minimum(loss_mask_all, loss_mask_planned).mean()
        else:
            loss_mask = loss_mask_all.mean()

        # 类别预测（每个维度代表一个类别）
        loss_cls = self.loss_fn['cr_mse'](pred['cls'], label_cls).mean()

        # 尺寸预测
        reg_size_mask = reg_size_mask[:, None, :, :]
        mask_size = reg_size_mask.sum() + 0.1
        loss_size = self.loss_fn['cr_mse'](
            pred['size'] * reg_size_mask,
            reg_size * reg_size_mask
        ).sum() / mask_size

        losses = [
            ('mask', loss_mask),
            ('cls', loss_cls),
            ('size', loss_size)
        ]
        loss = sum([loss_value * self.loss_scale[loss_name] for loss_name, loss_value in losses])

        mask_loss_item = loss_mask.detach().item()
        cls_loss_item = loss_cls.detach().item()
        size_loss_item = loss_size.detach().item()
        total_loss_item = loss.detach().item()

        res_loss_item = {
            'mask': mask_loss_item,
            'cls': cls_loss_item,
            'size': size_loss_item,
            'total_loss': total_loss_item
        }
        return loss, res_loss_item

    def train(self, epoch):
        # 每一个epoch开始前，重置数据集采样器
        self.dataset_train.inner_dataset.reset_epoch()
        # 检验是否需要动态解冻网络
        self.check_unfreeze_stage(epoch)

        if self.distributed:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        num_batches = len(self.train_dl)
        self.plot_tensorboard_feature_period = max(min(self.plot_tensorboard_feature_period, num_batches - 1), 1)
        train_mask_loss = 0
        train_cls_loss = 0
        train_size_loss = 0
        train_sim_loss = 0
        train_total_loss = 0
        batch_size = self.cfg['train_data_loader']['batch_size']

        # 仅 rank=0 打印
        if self.is_rank0 and self.logger:
            size = len(self.train_dl.dataset)
            self.logger.info(f'训练集大小:{size}\t 训练batch数量:{num_batches}')

        for index, batch in enumerate(self.train_dl):
            with torch.set_grad_enabled(True):
                # 新旧数据合并
                data = merge_dataset(batch)

                # 输入数据前处理
                input_images = self.process_input(data)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(False):
                    # 主任务前向：使用 RGB 图像进行预测
                    if self.return_feature_map:
                        predictions, features = self.model(input_images)
                        loss_sim = self.calc_sim_loss(batch_size, features)
                    else:
                        predictions = self.model(input_images)
                        loss_sim = None

                    # 对于主任务损失，我们仅使用前 B 个batch的预测
                    predictions = [tensor[:batch_size] for tensor in predictions]  # 例如 [mask, size]
                    predictions = {
                        'mask': predictions[0],
                        'size': predictions[1],
                        'cls': predictions[2]
                    }

                    # 主任务损失（例如 mask、size 损失），用原始标签 data 计算
                    loss_main, res_loss_item = self.calc_loss(data, predictions, epoch)

                    # 总Loss
                    if loss_sim is not None:
                        loss = loss_main + loss_sim
                    else:
                        loss = loss_main

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if loss_sim is None:
                    loss_sim = 0

                total_loss = loss.detach().item()
                train_mask_loss += res_loss_item['mask']
                train_cls_loss += res_loss_item['cls']
                train_size_loss += res_loss_item['size']
                train_sim_loss += loss_sim
                train_total_loss += total_loss

                if self.is_rank0 and self.logger and (index % 10 == 0):
                    self.logger.info(
                        f'Epoch:{epoch} 第[{index}/{num_batches}]次迭代loss:'
                        f'mask_loss={res_loss_item["mask"]}, cls_loss={res_loss_item["cls"]}, '
                        f'size_loss={res_loss_item["size"]}, '
                        f'sim_loss={loss_sim}, total_loss={total_loss}'
                    )
                if self.is_rank0 and self.logger and self.plot_tensorboard_feature \
                        and (index % self.plot_tensorboard_feature_period == 0):
                    # 记录特征图
                    self.plot_tensorboard_feature_map('train', predictions, epoch)

                # 内存管理：删除不再需要的变量，并清理缓存
                if self.return_feature_map:
                    del data, input_images, predictions, features, loss_main, loss_sim, loss, res_loss_item
                else:
                    del data, input_images, predictions, loss_main, loss_sim, loss, res_loss_item
                torch.cuda.empty_cache()
                gc.collect()

        # 计算 epoch 均值
        train_mask_loss /= num_batches
        train_cls_loss /= num_batches
        train_size_loss /= num_batches
        train_sim_loss /= num_batches
        train_total_loss /= num_batches

        if self.distributed:
            # 同步所有进程的损失
            for name in ['train_mask_loss', 'train_cls_loss', 'train_size_loss', 'train_sim_loss',
                         'train_total_loss']:
                tensor = torch.tensor(locals()[name], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                locals()[name] = tensor.item() / self.world_size

        # 内存管理：删除不再需要的变量，并清理缓存
        gc.collect()

        train_loss_mean = [
            ('train_mask_loss', train_mask_loss),
            ('train_cls_loss', train_cls_loss),
            ('train_size_loss', train_size_loss),
            ('train_sim_loss', train_sim_loss),
            ('train_total_loss', train_total_loss)
        ]

        # 仅 rank=0 写 tensorboard
        if self.is_rank0 and self.tensorboard_logger:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            tensorboard_log = [
                ("train/train_mask_loss", train_mask_loss),
                ("train/train_cls_loss", train_cls_loss),
                ("train/train_size_loss", train_size_loss),
                ('train/train_sim_loss', train_sim_loss),
                ("train/train_total_loss", train_total_loss),
                ("train/lr", lr)
            ]
            self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        return train_loss_mean

    def valid(self, epoch):
        # 每一个epoch开始前，重置数据集采样器
        self.dataset_valid.inner_dataset.reset_epoch()
        if self.distributed:
            self.valid_sampler.set_epoch(epoch)

        self.model.eval()

        num_batches = len(self.valid_dl)
        self.plot_tensorboard_feature_period = max(min(self.plot_tensorboard_feature_period, num_batches - 1), 1)
        test_mask_loss = 0
        test_cls_loss = 0
        test_size_loss = 0
        test_sim_loss = 0
        test_total_loss = 0
        batch_size = self.cfg['val_data_loader']['batch_size']

        if self.is_rank0 and self.logger:
            size = len(self.valid_dl.dataset)
            self.logger.info(f'验证集大小:{size}\t 验证batch数量:{num_batches}')

        for index, batch in enumerate(self.valid_dl):
            with torch.set_grad_enabled(False):
                # 新旧数据合并
                data = merge_dataset(batch)

                # 输入数据前处理
                input_images = self.process_input(data)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(False):
                    # 主任务前向：使用 RGB 图像进行预测
                    if self.return_feature_map:
                        predictions, features = self.model(input_images)
                        loss_sim = self.calc_sim_loss(batch_size, features)
                    else:
                        predictions = self.model(input_images)
                        loss_sim = None

                    # 对于主任务损失，我们仅使用前 B 个batch的预测
                    predictions = [tensor[:batch_size] for tensor in predictions]  # 例如 [mask, size]
                    predictions = {
                        'mask': predictions[0],
                        'size': predictions[1],
                        'cls': predictions[2]
                    }

                    # 主任务损失（例如 mask、size 损失），用原始标签 data 计算
                    loss_main, res_loss_item = self.calc_loss(data, predictions, epoch)

                    # 总Loss
                    if loss_sim is not None:
                        loss = loss_main + loss_sim
                    else:
                        loss = loss_main

                if loss_sim is None:
                    loss_sim = 0

                total_loss = loss.detach().item()
                test_mask_loss += res_loss_item['mask']
                test_cls_loss += res_loss_item['cls']
                test_size_loss += res_loss_item['size']
                test_sim_loss += loss_sim
                test_total_loss += total_loss

                if self.is_rank0 and self.logger and (index % 10 == 0):
                    self.logger.info(
                        f'Epoch:{epoch} 第[{index}/{num_batches}]次验证loss:'
                        f'mask_loss={res_loss_item["mask"]}, cls_loss={res_loss_item["cls"]}, '
                        f'size_loss={res_loss_item["size"]}, '
                        f'sim_loss={loss_sim}, total_loss={total_loss}'
                    )
                if self.is_rank0 and self.logger and self.plot_tensorboard_feature \
                        and (index % self.plot_tensorboard_feature_period == 0):
                    # 记录特征图
                    self.plot_tensorboard_feature_map('test', predictions, epoch)

        test_mask_loss /= num_batches
        test_cls_loss /= num_batches
        test_size_loss /= num_batches
        test_sim_loss /= num_batches
        test_total_loss /= num_batches

        if self.distributed:
            # 同步所有进程的损失
            for name in ['test_mask_loss', 'test_cls_loss', 'test_size_loss', 'test_sim_loss', 'test_total_loss']:
                tensor = torch.tensor(locals()[name], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                locals()[name] = tensor.item() / self.world_size

        gc.collect()

        test_loss_mean = [
            ('test_mask_loss', test_mask_loss),
            ('test_cls_loss', test_cls_loss),
            ('test_size_loss', test_size_loss),
            ('test_sim_loss', test_sim_loss),
            ('test_total_loss', test_total_loss)
        ]

        if self.is_rank0 and self.tensorboard_logger:
            tensorboard_log = [
                ("test/test_mask_loss", test_mask_loss),
                ("test/test_cls_loss", test_cls_loss),
                ("test/test_size_loss", test_size_loss),
                ("test/test_sim_loss", test_sim_loss),
                ("test/test_total_loss", test_total_loss)
            ]
            self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        # 仅 rank=0 进程保存模型
        if self.is_rank0:
            # 保存第一个epoch模型
            if epoch == 1:
                for yaml_file in self.yaml_list:
                    shutil.copy(
                        yaml_file,
                        os.path.join(self.model_save_path, os.path.basename(yaml_file))
                    )

            # 简单 early stopping，保存loss最小的模型
            valid_loss_min = test_total_loss
            if valid_loss_min < self.epoch_minimum_loss - self.min_delta:
                self.epoch_minimum_loss = valid_loss_min
                self.no_improvement_count = 0
                if self.logger:
                    self.logger.info(
                        f'minimum_validate_loss = {self.epoch_minimum_loss:.5f}, '
                        f'saving model to minimum_loss.pt'
                    )

                self.save_model('minimum_loss.pt', epoch)
            else:
                self.no_improvement_count += 1
                if self.logger:
                    self.logger.info(f"No improvement count: {self.no_improvement_count}")

            if self.no_improvement_count >= self.patience:
                if self.logger:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                self.early_stop = True

        # 同步early_stop标志
        if self.distributed:
            early_stop_tensor = torch.tensor([self.early_stop], device=self.device)
            dist.broadcast(early_stop_tensor, src=0)
            self.early_stop = early_stop_tensor.item()

        return test_loss_mean
