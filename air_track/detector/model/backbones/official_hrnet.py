import timm
import torch
from torch import nn
import torch.nn.functional as F
from air_track.utils.registry import BACKBONES


@BACKBONES.register('OfficialHRNet_2_Stage')
class OfficialHRNet_2_Stage(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        base_model_name = cfg['base_model_name'].lower()  # hrnet_w48
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)  # 512

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,  # 只要模型的特征提取部分
                                            out_indices=(0, 1),  # 指定模型哪些层的输出作为特征
                                            in_chans=input_ch,  # 输入通道数（2）
                                            pretrained=pretrained)  # 是否使用预训练权重

        # 获取每层的通道数组成列表
        backbone_depths = list(self.base_model.feature_info.channels())  # [64, 128, 256, 512, 1024]
        self.output_channels = sum(backbone_depths)  # 1984

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        stages_output = self.base_model(inputs)
        """
        torch.Size([5, 64, 512, 512])
        torch.Size([5, 128, 256, 256])
        torch.Size([5, 256, 128, 128])
        torch.Size([5, 512, 64, 64])
        torch.Size([5, 1024, 32, 32])
        """

        # 对应hrnet网咯四层的特征
        x = [
            stages_output[0],  # 512 torch.Size([2, 64, 512, 512])
            # 上采样，特征每个维度×2，bilinear使用双线性插值
            F.interpolate(stages_output[1], size=output_size, mode="bilinear"),  # torch.Size([2, 128, 512, 512])
        ]

        # 特征拼接
        x = torch.cat(x, dim=1)

        return x


@BACKBONES.register('OfficialHRNet_5_Stage')
class OfficialHRNet_5_Stage(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        base_model_name = cfg['base_model_name'].lower()  # hrnet_w48
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)  # 512

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,  # 只要模型的特征提取部分
                                            out_indices=(0, 1, 2, 3, 4),  # 指定模型哪些层的输出作为特征
                                            in_chans=input_ch,  # 输入通道数（2）
                                            pretrained=pretrained)  # 是否使用预训练权重

        # 获取每层的通道数组成列表
        backbone_depths = list(self.base_model.feature_info.channels())  # [64, 128, 256, 512, 1024]
        self.output_channels = sum(backbone_depths)  # 1984

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        stages_output = self.base_model(inputs)
        """
        torch.Size([5, 64, 512, 512])
        torch.Size([5, 128, 256, 256])
        torch.Size([5, 256, 128, 128])
        torch.Size([5, 512, 64, 64])
        torch.Size([5, 1024, 32, 32])
        """

        # 对应hrnet网咯四层的特征
        x = [
            stages_output[0],  # 512 torch.Size([2, 64, 512, 512])
            # 上采样，特征每个维度×2，bilinear使用双线性插值
            F.interpolate(stages_output[1], size=output_size, mode="bilinear"),  # torch.Size([2, 128, 512, 512])
            F.interpolate(stages_output[2], size=output_size, mode="bilinear"),  # torch.Size([2, 256, 512, 512])
            F.interpolate(stages_output[3], size=output_size, mode="bilinear"),  # torch.Size([2, 512, 512, 512])
            F.interpolate(stages_output[4], size=output_size, mode="bilinear"),  # torch.Size([2, 1024, 512, 512])
        ]

        # 特征拼接
        x = torch.cat(x, dim=1)

        return x
