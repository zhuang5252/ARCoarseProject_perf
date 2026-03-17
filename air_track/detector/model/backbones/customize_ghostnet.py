import torch
import torch.nn as nn
import torch.nn.functional as F
from air_track.utils.registry import BACKBONES


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super(GhostConv, self).__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        return torch.cat([primary, cheap], dim=1)


class GhostBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(GhostBlock, self).__init__()
        self.conv1 = GhostConv(in_planes, out_planes, kernel_size=3, stride=stride, relu=True)
        self.conv2 = GhostConv(out_planes, out_planes, kernel_size=3, stride=1, relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.relu(out)


def _make_layer(block, in_planes, out_planes, num_blocks, stride=1):
    layers = [block(in_planes, out_planes, stride)]
    for _ in range(1, num_blocks):
        layers.append(block(out_planes, out_planes, stride=1))
    return nn.Sequential(*layers)

@BACKBONES.register("GhostNet12")
class GhostNet12(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super(GhostNet12, self).__init__()

        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.output_channels = 256

        self.layer1 = _make_layer(GhostBlock, input_ch, 64, num_blocks=1, stride=1)
        self.layer2 = _make_layer(GhostBlock, 64, 128, num_blocks=1, stride=2)
        self.layer3 = _make_layer(GhostBlock, 128, 256, num_blocks=1, stride=2)
        self.layer4 = _make_layer(GhostBlock, 256, 512, num_blocks=1, stride=2)
        self.high_mapper = nn.Conv2d(512, self.output_channels, kernel_size=1)

    def forward(self, x, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_feat = self.high_mapper(x)

        # 上采样到标签尺寸
        if high_feat.shape[2:] != output_size:
            high_feat = F.interpolate(high_feat, size=output_size, mode='bilinear')

        return high_feat

@BACKBONES.register("GhostNet12_Lite")
class GhostNet12Lite(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super(GhostNet12Lite, self).__init__()

        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.output_channels = 256

        # ↓ 更小的通道数
        self.layer1 = _make_layer(GhostBlock, input_ch, 32, num_blocks=1, stride=1)
        self.layer2 = _make_layer(GhostBlock, 32, 64, num_blocks=1, stride=2)
        self.layer3 = _make_layer(GhostBlock, 64, 128, num_blocks=1, stride=2)
        self.layer4 = _make_layer(GhostBlock, 128, 256, num_blocks=1, stride=2)
        self.high_mapper = nn.Conv2d(256, self.output_channels, kernel_size=1)

    def forward(self, x, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_feat = self.high_mapper(x)

        # 上采样到标签尺寸
        if high_feat.shape[2:] != output_size:
            high_feat = F.interpolate(high_feat, size=output_size, mode='bilinear')

        return high_feat
