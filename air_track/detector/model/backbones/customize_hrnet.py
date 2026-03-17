import torch
from torch import nn
import torch.nn.functional as F
from air_track.utils.registry import BACKBONES


#########################################
# HRNet系列 HRNet18, HRNet32, HRNet48
#########################################
@BACKBONES.register('CustomizeHRNet')
class CustomizeHRNet(nn.Module):
    """
    通用的简化版HRNet骨干网络，通过参数 variant 指定基础通道数：
      - variant="W18": 基础通道设为18（实际网络中通常乘以倍数，这里为了简单设定）
      - variant="W32": 基础通道设为32
      - variant="W48": 基础通道设为48
    网络流程：
      1. Stem：对单通道输入做下采样（1/2分辨率）。
      2. 两路分支：
         - 高分辨率分支：保持分辨率，输出通道为 base_ch * 2
         - 低分辨率分支：进一步下采样一倍，输出通道为 base_ch * 4
      3. 将低分辨率分支上采样到高分辨率后，与高分辨率分支拼接，
         再通过1×1卷积映射到 final_channels（默认256）。
    """
    _VALID_BASE_MODELS = {'hrnet_w9': 9, 'hrnet_w18': 18, 'hrnet_w32': 32, 'hrnet_w48': 48}

    def __init__(self, cfg, pretrained=False):
        super(CustomizeHRNet, self).__init__()
        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.base_model_name = cfg['base_model_name'].lower()  # hrnet_w48
        self.down_scale = int(cfg['down_scale'])
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)  # 512

        try:
            base_ch = self._VALID_BASE_MODELS[self.base_model_name]
        except:
            raise ValueError(f"Unsupported base_model_name: '{self.base_model_name}'")

        self.output_channels = 256

        # Stem：对输入进行初步下采样（1/2分辨率）
        self.stem = nn.Sequential(
            nn.Conv2d(input_ch, base_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        # 高分辨率分支：保持分辨率，输出通道 = base_ch * 2
        self.hr_branch = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )
        # 低分辨率分支：进一步下采样一倍，输出通道 = base_ch * 4
        self.lr_branch = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True)
        )
        # 融合：将低分辨率分支上采样到与高分辨率分支相同大小后拼接，再用1×1卷积映射到 final_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2 + base_ch * 4, self.output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True)
        )
        self.low_reduce = nn.Sequential(
            nn.Conv2d(base_ch * 2, self.output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, pool_size=None, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        x = self.stem(x)  # 下采样到 1/2 分辨率, 输出: [B, base_ch, H, W]
        hr_feat = self.hr_branch(x)  # 高分辨率分支, 输出: [B, base_ch*2, H, W] 例如36通道
        lr_feat = self.lr_branch(x)  # 低分辨率分支, 输出: [B, base_ch*4, H/2, W/2] 例如72通道（W18）
        lr_feat_up = F.interpolate(lr_feat, size=hr_feat.shape[2:], mode='bilinear', align_corners=False)
        fused_feat = torch.cat([hr_feat, lr_feat_up], dim=1)  # [B, base_ch*2 + base_ch*4, H, W] 例如108通道
        out = self.fuse_conv(fused_feat)  # [B, final_channels, H, W]，例如256通道
        # 对低层特征做额外映射：
        hr_feat_mapped = self.low_reduce(hr_feat)  # [B, final_channels, H, W]

        if pool_size is not None:
            out = F.adaptive_avg_pool2d(out, pool_size)
            hr_feat_mapped = F.adaptive_avg_pool2d(hr_feat_mapped, pool_size)

        # 上采样到标签尺寸
        if out.shape[2:] != output_size:
            out = F.interpolate(out, size=output_size, mode='bilinear')
            hr_feat_mapped = F.interpolate(hr_feat_mapped, size=output_size, mode='bilinear')

        if return_multi:
            return out, hr_feat_mapped

        return out
