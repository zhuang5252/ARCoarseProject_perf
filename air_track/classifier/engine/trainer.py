import copy
import os
import shutil
import time
import torch
from air_track.classifier.model.model import Model
from air_track.utils import loss, col_data_save_csv
from torch.utils.data import DataLoader, random_split
from air_track.classifier.utils import calculate_correct_num
from air_track.classifier.data.dataset import ClassifierDataset
from air_track.schedulers import optimize_lr, CosineAnnealingWarmRestarts
from air_track.utils.train_log import log_configurator, TensorboardLogger


class Trainer:
    def __init__(self, cfg, yaml_list):
        self.cfg = cfg
        self.yaml_list = yaml_list
        self.model_params = cfg['model_params']
        self.dataset_params = cfg['dataset_params']
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
        self.best_acc = 0
        self.copy_split_data_flag = self.dataset_params.get('copy_split_data', False)
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

        if cfg['classify_log'] != str(None) and cfg['classify_log'] != model_save_root:
            log_dir = os.path.join(cfg['classify_log'], time_str_ymd)
        else:
            log_dir = self.model_save_path
        self.logger, self.log_file = log_configurator(log_dir)
        self.tensorboard_logger = TensorboardLogger(log_dir=log_dir, log_hist=False)
        print('Model Save Path: ', self.model_save_path)

    def copy_split_data(self, dataset, folder='val', img_format='.jpg'):
        """拷贝拆分后的数据"""
        npy_paths = dataset.dataset.npy_paths
        save_dir = os.path.join(self.cfg['data_dir'], folder)
        img_files, cls_list, neg_list, pos_list = [], [], [], []
        for i, idx in enumerate(dataset.indices):
            label = copy.deepcopy(dataset.dataset.labels[idx])
            npy_path = npy_paths[idx]
            img_path = npy_path.replace('/npy', '/img')
            img_path = img_path.replace('.npy', img_format)
            save_img_dir = os.path.join(save_dir, 'img')
            os.makedirs(save_img_dir, exist_ok=True)
            base_dir = os.path.dirname(os.path.dirname(img_path)).split('/')[-1]
            base_name = os.path.basename(img_path).split('.')[0]
            file_name = f'{i + 1}_' + base_dir + '_' + base_name + img_format
            save_path = os.path.join(save_img_dir, file_name)
            shutil.copy2(img_path, save_path)

            label[0] = file_name
            img_files.append(label[0])
            cls_list.append(label[1])
            neg_list.append(label[2])
            pos_list.append(label[3])

        # 添加到CSV数据
        csv_data = {
            'index': img_files,
            'cls_name': cls_list,
            'neg': neg_list,
            'pos': pos_list
        }

        col_data_save_csv(f'{save_dir}/label.csv', csv_data)

    def set_dataset(self):
        """构建数据"""
        if self.dataset_params['scale_factor'] != 'None':
            dataset = ClassifierDataset(stage=self.cfg['stage_train'], cfg=self.cfg)
            train_size = int(len(dataset) * self.dataset_params['scale_factor'])
            valid_size = len(dataset) - train_size
            print('train_size: {}, valid_size: {}'.format(train_size, valid_size))
            dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size])
        else:
            dataset_train = ClassifierDataset(stage=self.cfg['stage_train'], cfg=self.cfg)
            dataset_valid = ClassifierDataset(stage=self.cfg['stage_valid'], cfg=self.cfg)

        # 拷贝拆分后的训练集、验证集
        if self.copy_split_data_flag:
            # self.copy_split_data(dataset_train, folder='train', img_format=self.cfg['img_format'])
            self.copy_split_data(dataset_valid, folder='val', img_format=self.cfg['img_format'])

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
        self.model = Model(self.model_params, pretrained=self.model_params['pretrained'])

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
        cr_mae = torch.nn.L1Loss(reduction='mean')  # 绝对值损失
        cr_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')  # 结合 Sigmoid 函数和二进制交叉熵损失函数
        cr_wbce = loss.WeightBCEWithLogitsLoss(neg_weight=self.train_params['neg_weight'],
                                               pos_weight=self.train_params['pos_weight'],
                                               reduction='mean')  # 结合 Sigmoid 函数和二进制交叉熵自实现损失函数
        cr_fl = loss.FocalLossV2(alpha=self.train_params['alpha'],
                                 gamma=self.train_params['gamma'], reduction='mean')  # FocalLoss：平衡正负样本和难易样本

        self.loss_fn = {  # 将整个loss存入字典
            'cr_mse': cr_mse,
            'cr_mae': cr_mae,
            'cr_bce': cr_bce,
            'cr_wbce': cr_wbce,
            'cr_fl': cr_fl,
        }

        return self.loss_fn

    def calc_loss(self, pred, label):
        """使用标签和模型输出计算loss"""
        loss = self.loss_fn[self.train_params['loss_type']](pred, label)
        loss_item = loss.detach().item()

        return loss, loss_item

    def train(self, epoch):
        self.model.train()

        index = 0
        train_loss = 0
        train_num_correct = 0

        batch_size = self.cfg['train_data_loader']['batch_size']
        size = len(self.train_dl.dataset)  # 训练集的大小
        num_batches = len(self.train_dl)  # 批次
        self.logger.info('训练集大小: {}\t 训练batch数量: {}'.format(size, num_batches))

        for data in self.train_dl:
            with torch.set_grad_enabled(True):
                input_data = data['input'].to(self.device)
                label = data['label'].to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(False):  # 混合精度
                    output = self.model(input_data)
                    # 计算损失
                    loss, loss_item = self.calc_loss(output, label)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # 梯度裁剪，以防止梯度爆炸
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                num_correct = calculate_correct_num(label, output)
                train_num_correct += num_correct
                train_loss += loss_item

                if index % 10 == 0:
                    self.logger.info(
                        'Epoch:{:2d} 第[{}/{}]次迭代, train_loss: {}, acc: {:.4f}\t '
                        .format(epoch, index, num_batches, loss_item, num_correct / batch_size))
                index += 1

        train_loss /= num_batches
        acc = train_num_correct / size

        # 获取当前的学习率
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # tensorboard可视化
        tensorboard_log = [
            ("train/train_loss", train_loss),
            ("train/lr", lr),
            ("train/acc", acc)
        ]
        self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        return train_loss, acc

    def valid(self, epoch):
        self.model.eval()

        index = 0
        val_loss = 0
        val_num_correct = 0

        batch_size = self.cfg['val_data_loader']['batch_size']
        size = len(self.valid_dl.dataset)  # 验证集的大小
        num_batches = len(self.valid_dl)  # 批次
        self.logger.info('验证集大小: {}\t 验证batch数量: {}'.format(size, num_batches))

        for data in self.valid_dl:
            with torch.set_grad_enabled(False):
                input_data = data['input'].to(self.device)
                label = data['label'].to(self.device)

                with torch.cuda.amp.autocast(False):  # 混合精度
                    output = self.model(input_data)
                    # 计算损失
                    loss, loss_item = self.calc_loss(output, label)

                num_correct = calculate_correct_num(label, output)
                val_num_correct += num_correct
                val_loss += loss_item

                if index % 10 == 0:
                    self.logger.info(
                        'Epoch:{:2d} 第[{}/{}]次迭代, val_loss: {}, acc: {:.4f}\t '
                        .format(epoch, index, num_batches, loss_item, num_correct / batch_size))
                index += 1

        val_loss /= num_batches
        acc = val_num_correct / size

        # tensorboard可视化
        tensorboard_log = [
            ("test/test_loss", val_loss),
            ("test/acc", acc)
        ]
        self.tensorboard_logger.list_of_scalars_summary(tensorboard_log, epoch)

        # 保存最新的模型
        self.logger.info(f'epoch = {epoch}, saving model to epoch_{epoch}.pt')
        self.save_model(f'epoch_{epoch}.pt', epoch)

        # 保存第一个epoch模型
        if epoch == 1:
            for yaml_file in self.yaml_list:
                shutil.copy(yaml_file, os.path.join(self.model_save_path, os.path.basename(yaml_file)))

        # 简单 early stopping，保存loss最小的模型
        valid_loss_min = val_loss
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

        if acc > self.best_acc:
            self.best_acc = acc
            self.logger.info(
                'best_validate_acc = {:.5f}, saving model to best_acc.pt'.format(self.best_acc))
            self.save_model('best_acc.pt', epoch)

        return val_loss, acc

    def load_checkpoint(self, checkpoint_path):
        """用于继续训练"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint['epoch'] > 0:
            print('Continue model: ', checkpoint_path)

            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        else:
            self.set_model()

        return checkpoint['epoch']

    def save_model(self, model_name, epoch):
        epoch_model = copy.deepcopy(self.model)

        torch.save({
            "epoch": epoch,
            "model_state_dict": epoch_model.module.state_dict() if self.distributed else epoch_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        },
            f"{self.model_save_path}/{model_name}", )
