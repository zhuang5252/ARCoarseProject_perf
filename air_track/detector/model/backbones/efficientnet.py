import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Union
from air_track.utils import combine_load_cfg_yaml
from air_track.utils.registry import BACKBONES


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    Ensure all layers have channels divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels / squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


class MBConvConfig:
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConvBlock(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        # expansion phase
        if expanded_channels != cnf.input_channels:
            layers.append(
                nn.Conv2d(cnf.input_channels, expanded_channels, 1, bias=False)
            )
            layers.append(norm_layer(expanded_channels))
            layers.append(nn.SiLU(inplace=True))

        # depthwise convolution
        layers.append(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                cnf.kernel,
                cnf.stride,
                (cnf.kernel - 1) // 2,
                groups=expanded_channels,
                bias=False,
            )
        )
        layers.append(norm_layer(expanded_channels))
        layers.append(nn.SiLU(inplace=True))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels))

        # projection phase
        layers.append(
            nn.Conv2d(expanded_channels, cnf.out_channels, 1, bias=False)
        )
        layers.append(norm_layer(cnf.out_channels))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            input_channel: int = 3,
            inverted_residual_setting: Optional[List[MBConvConfig]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # expand_ratio, kernel, stride, input_channels, out_channels, num_layers
                MBConvConfig(1, 3, 1, 32, 16, 1, width_mult, depth_mult),
                MBConvConfig(6, 3, 2, 16, 24, 2, width_mult, depth_mult),
                MBConvConfig(6, 5, 2, 24, 40, 2, width_mult, depth_mult),
                MBConvConfig(6, 3, 2, 40, 80, 3, width_mult, depth_mult),
                MBConvConfig(6, 5, 1, 80, 112, 3, width_mult, depth_mult),
                MBConvConfig(6, 5, 2, 112, 192, 4, width_mult, depth_mult),
                MBConvConfig(6, 3, 1, 192, 320, 1, width_mult, depth_mult),
            ]

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.features = [nn.Sequential(
            nn.Conv2d(input_channel, firstconv_output_channels, 3, 2, 1, bias=False),
            norm_layer(firstconv_output_channels),
            nn.SiLU(inplace=True)
        )]

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers = []
            for _ in range(cnf.num_layers):
                layers.append(MBConvBlock(cnf, norm_layer))
            self.features.append(nn.Sequential(*layers))

        self.features = nn.Sequential(*self.features)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in [2, 4, 6, 9]:  # Selected layers for multi-scale features
                outputs.append(x)

        return outputs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_impl(x)


@BACKBONES.register('OfficialEfficientNet')
class OfficialEfficientNet(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        input_ch = cfg['input_channel'] * cfg['input_frames']
        self.down_scale = int(cfg['down_scale'])
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)

        # Get model configuration from cfg or use default
        width_mult = cfg.get('width_mult', 1.0)
        depth_mult = cfg.get('depth_mult', 1.0)

        self.base_model = EfficientNet(
            input_channel=input_ch,
            width_mult=width_mult,
            depth_mult=depth_mult
        )

        # 动态计算通道数
        with torch.no_grad():
            img_size = cfg.get('img_size', 512)
            dummy_input = torch.randn(1, input_ch, img_size, img_size)
            stages_output = self.base_model(dummy_input)

            self.output_channels = 0
            for item in stages_output:
                self.output_channels += item.size(1)

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        stages_output = self.base_model(inputs)

        # 上采样所有特征到相同尺寸并拼接
        x = []
        for feature in stages_output:
            x.append(F.interpolate(feature, size=output_size, mode="bilinear"))

        x = torch.cat(x, dim=1)

        return x


if __name__ == '__main__':
    # 测试代码
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_yaml = os.path.join(script_dir, 'config/detect_train_test.yaml')
    yaml_list = [train_yaml]
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    model = OfficialEfficientNet(cfg_data['model_params'])
    output = model(torch.zeros((2, 2, 16, 16)))
    print(output.shape)