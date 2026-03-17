import os
from air_track.utils import reprod_init, combine_load_cfg_yaml, auto_import_module


def train(cfg, yaml_list, checkpoint_path=None):
    # 获取模型输出头名称和数量
    head_name = cfg['model_params']['head_name']
    head_nums = cfg['model_params']['head_nums']
    if str(head_nums) not in head_name:
        raise ValueError(f"head_nums '{head_nums}' is not in head_name '{head_name}'")
    # 根据输出头数量选择训练器
    Trainer = auto_import_module(
        f'air_track.detector.engine.trainer_{head_nums}_output', 'Trainer')

    # 固定随机数种子
    reprod_init(seed=cfg['seed'])

    # 训练迭代次数
    train_params = cfg["train_params"]
    nb_epochs = train_params['nb_epochs']

    # 创建训练器
    trainer = Trainer(cfg=cfg, yaml_list=yaml_list)

    # 构建数据集
    trainer.set_dataset()

    # 构建模型
    model = trainer.set_model()
    print('The Model is: ', model.__class__.__name__)

    # 创建优化器和学习率调度器
    trainer.set_optimizer_and_scheduler()

    # 定义损失函数
    trainer.set_loss_fn()
    trainer.set_loss_scale()

    # 继续训练
    continue_epoch = 0
    if checkpoint_path is not None:
        continue_epoch = trainer.load_checkpoint(checkpoint_path)

    # 初始化训练和测试Loss
    train_loss = []
    validate_loss = []
    # 循环为正式开始训练
    for epoch in range(continue_epoch + 1, continue_epoch + nb_epochs + 1):
        trainer.model.train()
        """
        dataloader, optimizer, model, loss_fn, train_params, scaler, grad_clip_value, epoch
        """
        epoch_train_loss = trainer.train(epoch)

        trainer.scheduler.step()
        trainer.model.eval()
        """
        dataloader, model, loss_fn, train_params, epoch
        """
        epoch_validate_loss = trainer.valid(epoch)
        # 获取当前的学习率
        lr = trainer.optimizer.state_dict()['param_groups'][0]['lr']

        if trainer.is_rank0:
            trainer.logger.info('Epoch:{:2d}\t train_loss={:.5f}\t test_loss={:.5f}\t, Lr={:.2E}'
                                .format(epoch, epoch_train_loss[-1][1], epoch_validate_loss[-1][1], lr))

        train_loss.append(epoch_train_loss[-1][1])
        validate_loss.append(epoch_validate_loss[-1][1])

        if trainer.early_stop:
            break

    # 循环结束后，计算并输出Loss的统计量
    train_loss_minimum = min(train_loss)
    train_loss_minimun_epoch = train_loss.index(train_loss_minimum) + 1
    validate_loss_minimum = min(validate_loss)
    validate_loss_minimun_epoch = validate_loss.index(validate_loss_minimum) + 1
    if trainer.is_rank0:
        trainer.logger.info('finish training!')
        trainer.logger.info('训练损失最小值：{} 对应的epoch：{}'.format(train_loss_minimum, train_loss_minimun_epoch))
        trainer.logger.info(
            '评估损失最小值：{} 对应的epoch：{}'.format(validate_loss_minimum, validate_loss_minimun_epoch))

    return os.path.join(trainer.model_save_path, 'minimum_loss.pt')


if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_yaml = os.path.join(script_dir, 'config/dataset_zbzx.yaml')
    train_yaml = os.path.join(script_dir, 'config/detect_train_zbzx.yaml')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    train(cfg=cfg_data, yaml_list=yaml_list)
