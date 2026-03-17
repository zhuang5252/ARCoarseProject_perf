import torch
import madgrad
from torch.optim import lr_scheduler


def optimize_SGD(parameters, lr, mu=0.9):
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=mu)
    return optimizer


def optimize_Adam(parameters, lr):
    optimizer = torch.optim.Adam(parameters, lr=lr)
    return optimizer


def optimize_AdamW(parameters, lr):
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    return optimizer


def optimize_madgrad(parameters, lr):
    """
    MadGrad 优化器

    parameters: 神经网络模型可训练参数
    lr: 学习率
    """

    optimizer = madgrad.MADGRAD(parameters, lr=lr)
    return optimizer


def LR_scheduler_exponential(optimizer, gamma=0.98):
    LR_Schedule = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return LR_Schedule


def LR_scheduler_CosWarmUp(optimizer, T_0=10, T_mult=2):
    """
    加WarmUp的余弦退火，T_0表示每隔几个epoch，LR跳回到初始值，T_mult可以改变T_0，T_0=T_0+T_0*T_mult
    """
    LR_Schedule = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    return LR_Schedule


def LR_scheduler_Cosine(optimizer, T_max=8):
    """
    不加WarmUp的余弦退火，T_max为LR变化满一次余弦函数的半个周期，单位为epoch
    """
    LR_Schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return LR_Schedule
