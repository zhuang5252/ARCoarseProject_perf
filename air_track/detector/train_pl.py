import re
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from air_track.detector.data.dataset import PlDetectDataset
from air_track.detector.engine.trainer_pl import PlModel
from air_track.utils import reprod_init, combine_load_cfg_yaml
from air_track.utils.train_log import PlTensorBoardLogger


def train(cfg):
    # 固定随机数种子
    reprod_init(seed=cfg['seed'])

    # 训练迭代次数
    train_params = cfg['train_params']
    nb_epochs = train_params['nb_epochs']
    if isinstance(cfg['cuda_device'], str):
        cuda_devices = list(map(int, re.findall(r'\d+', cfg['cuda_device'])))
    elif isinstance(cfg['cuda_device'], int):
        cuda_devices = [cfg['cuda_device']]
    elif isinstance(cfg['cuda_device'], list):
        cuda_devices = cfg['cuda_device']
    else:
        raise ValueError('cuda_device must be str or int')

    model = PlModel(cfg=cfg, pretrained=False)
    model.set_loss_fn()

    checkpoint_callback = ModelCheckpoint(
        dirpath=model.model_save_path,  # 模型保存路径
        filename='minimum_loss',
        monitor='val/val_total_loss',  # 监控的指标
        save_top_k=3,  # 保存最佳3个模型
        mode='min',  # 指标越小越好
    )

    # logger = TensorBoardLogger(
    #     save_dir=model.model_save_path,
    #     name="exp_name",
    #     version="v1",
    #     default_hp_metric=False  # 关闭自动记录的hp_metric（避免干扰）
    # )

    logger = PlTensorBoardLogger(
        save_dir=model.model_save_path,
        name="exp_name",
        version="v1",
        default_hp_metric=False  # 关闭自动记录的hp_metric（避免干扰）
    )

    # 创建训练器
    trainer = pl.Trainer(
        logger=[logger],
        default_root_dir=model.model_save_path,
        callbacks=[checkpoint_callback],
        devices=cuda_devices,
        max_epochs=nb_epochs,
        num_sanity_val_steps=0,
        # log_every_n_steps=int(1e+10),
    )

    # 构建数据集
    train_dataset = PlDetectDataset(stage=cfg_data['stage_train'], cfg_data=cfg_data)
    train_dl = train_dataset.set_dataloader()
    valid_dataset = PlDetectDataset(stage=cfg_data['stage_valid'], cfg_data=cfg_data)
    valid_dl = valid_dataset.set_dataloader()

    trainer.fit(model, train_dl, valid_dl)


if __name__ == "__main__":
    import os

    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_yaml = os.path.join(script_dir, 'config/dataset_cola.yaml')
    train_yaml = os.path.join(script_dir, 'config/detect_train_cola.yaml')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    train(cfg=cfg_data)
