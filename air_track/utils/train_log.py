import os
import time
import logging
import datetime
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
# from pytorch_lightning.loggers.logger import Logger
# from pytorch_lightning.utilities import rank_zero_only


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def log_configurator(log_root):
    """训练过程保存到log日志"""
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    log_file = os.path.join(log_root, time_str + '_train.log')
    logger = get_logger(log_file)
    # print(f'训练日志保存在{log_file}')
    return logger, log_file


class TensorboardLogger(object):
    """tensorboard曲线"""
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        # Check a new folder for each log should be dreated
        if log_hist:
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def add_images(self, tag, img_tensor, step):
        self.writer.add_images(tag, img_tensor, step)

    def add_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)


# class PlTensorBoardLogger(Logger):
#     def __init__(
#             self,
#             save_dir: str = "logs",
#             name: str = "experiment",
#             version: Optional[str] = None,
#             log_hist: bool = True,
#             **kwargs
#     ):
#         super().__init__()
#         self._save_dir = save_dir
#         self._name = name
#         self._log_hist = log_hist
#
#         # 自动生成版本号
#         self._version = version or self._generate_version()
#
#         # 创建完整日志路径
#         self._log_dir = self._create_log_dir()
#
#         # 延迟初始化 SummaryWriter
#         self._experiment: Optional[SummaryWriter] = None
#
#     @property
#     def experiment(self) -> SummaryWriter:
#         """返回 TensorBoard SummaryWriter 实例"""
#         if self._experiment is None:
#             self._experiment = SummaryWriter(log_dir=self.log_dir)
#         return self._experiment
#
#     @property
#     def log_dir(self) -> str:
#         """返回完整日志目录路径"""
#         return self._log_dir
#
#     @property
#     def name(self) -> str:
#         return self._name
#
#     @property
#     def version(self) -> str:
#         return self._version
#
#     @rank_zero_only
#     def log_metrics(
#             self,
#             metrics: Dict[str, float],
#             step: Optional[int] = None
#     ) -> None:
#         """记录标量指标"""
#         for metric_name, value in metrics.items():
#             self.experiment.add_scalar(metric_name, value, step)
#
#     @rank_zero_only
#     def log_hyperparams(
#             self,
#             params: Dict[str, Any],
#             metrics: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         """记录超参数（可选）"""
#         # 将超参数转换为 Markdown 表格格式
#         params_table = "\n".join([f"| {k} | {v} |" for k, v in params.items()])
#         params_table = "| Parameter | Value |\n|----------|-------|\n" + params_table
#
#         # 添加到 TensorBoard
#         self.experiment.add_text("Hyperparameters", params_table)
#
#     @rank_zero_only
#     def log_image(
#             self,
#             tag: str,
#             image_tensor,
#             global_step: Optional[int] = None
#     ) -> None:
#         """记录图像数据"""
#         self.experiment.add_image(tag, image_tensor, global_step)
#
#     @rank_zero_only
#     def log_figure(
#             self,
#             tag: str,
#             figure,
#             global_step: Optional[int] = None,
#             close: bool = True
#     ) -> None:
#         """记录 Matplotlib 图形"""
#         self.experiment.add_figure(tag, figure, global_step)
#         if close:
#             import matplotlib.pyplot as plt
#             plt.close(figure)
#
#     def _create_log_dir(self) -> str:
#         """创建日志目录结构"""
#         if self._log_hist:
#             version_path = os.path.join(self.save_dir, self.name, self.version)
#         else:
#             version_path = os.path.join(self.save_dir, self.name)
#
#         os.makedirs(version_path, exist_ok=True)
#         return version_path
#
#     def _generate_version(self) -> str:
#         """生成时间戳版本号"""
#         return datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
#
#     @rank_zero_only
#     def save(self) -> None:
#         """保存元数据（如需要）"""
#         # 可在此保存附加信息
#         pass
#
#     @rank_zero_only
#     def finalize(self, status: str) -> None:
#         """关闭资源"""
#         if self._experiment:
#             self._experiment.flush()
#             self._experiment.close()
#
#     @property
#     def save_dir(self) -> str:
#         """返回基础保存目录"""
#         return self._save_dir
#
#     def __getstate__(self):
#         """序列化时排除 SummaryWriter"""
#         state = self.__dict__.copy()
#         state["_experiment"] = None  # 不序列化 SummaryWriter
#         return state


# class TensorboardLogger(object):
#     """支持追加写入的TensorBoard日志记录器"""
#
#     def __init__(self, log_dir, new_log_dir=True, log_hist=True, max_retries=3):
#         """
#         Args:
#             log_dir (str): 根日志目录
#             new_log_dir (bool):
#                 True - 每次创建带时间戳的新目录（默认安全模式）
#                 False - 直接使用指定目录（支持追加写入）
#             max_retries (int): 目录创建失败时的重试次数
#         """
#         self._validate_dir(log_dir)
#
#         if new_log_dir:
#             # 带时间戳的新目录模式
#             self.log_path = self._create_timestamp_dir(log_dir, max_retries)
#         else:
#             # 直接使用现有目录模式
#             self.log_path = log_dir
#             os.makedirs(self.log_path, exist_ok=True)
#
#         self.writer = SummaryWriter(self.log_path, flush_secs=30)
#
#     def _validate_dir(self, path):
#         """验证目录合法性"""
#         if not isinstance(path, str):
#             raise TypeError(f"log_dir必须为字符串类型，当前类型：{type(path)}")
#         if not os.path.isabs(path):
#             raise ValueError(f"请使用绝对路径，相对路径可能存在歧义。当前路径：{path}")
#
#     def _create_timestamp_dir(self, base_dir, max_retries):
#         """创建带时间戳的子目录"""
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         for i in range(max_retries):
#             try:
#                 new_dir = os.path.join(base_dir, timestamp)
#                 if i > 0:  # 重试时添加序号
#                     new_dir += f"_{i}"
#                 os.makedirs(new_dir, exist_ok=False)
#                 return new_dir
#             except FileExistsError:
#                 continue
#         raise RuntimeError(f"无法在{max_retries}次重试内创建唯一目录")
#
#     def scalar_summary(self, tag, value, step):
#         """记录标量数据"""
#         self.writer.add_scalar(tag, value, step)
#
#     def list_of_scalars_summary(self, tag_value_pairs, step):
#         """批量记录标量数据"""
#         for tag, value in tag_value_pairs:
#             self.scalar_summary(tag, value, step)
#
#     def add_images(self, tag, img_tensor, step, dataformats="NCHW"):
#         """记录图像数据"""
#         self.writer.add_images(tag, img_tensor, step, dataformats=dataformats)
#
#     def add_figure(self, tag, figure, step, close_figure=True):
#         """记录matplotlib图像"""
#         self.writer.add_figure(tag, figure, step)
#         if close_figure:
#             import matplotlib.pyplot as plt
#             plt.close(figure)
#
#     def flush(self):
#         """立即写入缓冲区数据"""
#         self.writer.flush()
#
#     def close(self):
#         """安全关闭写入器"""
#         self.flush()
#         self.writer.close()
