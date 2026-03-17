import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Union
from air_track.utils import combine_load_cfg_yaml
from air_track.utils.registry import BACKBONES


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw - 扩展通道
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

        # dw - 深度可分离卷积
        # 关键修改：groups总是等于当前输入通道数
        layers.append(ConvBNReLU(
            hidden_dim, hidden_dim,
            stride=stride,
            groups=hidden_dim,  # 分组数等于输入通道数
            norm_layer=norm_layer
        ))

        # pw-linear - 压缩通道
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            input_channel: int = 3,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.input_channel = input_channel

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list")

        # building first layer
        input_channel = _make_divisible(input_channel * 32, round_nearest)
        self.first_conv = ConvBNReLU(self.input_channel, 32, stride=2, norm_layer=norm_layer)
        features: List[nn.Module] = [self.first_conv]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * 32, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t, norm_layer))
                input_channel = output_channel

        self.features = nn.Sequential(*features)

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
        x = self.first_conv(x)
        for i, module in enumerate(self.features[1:]):
            x = module(x)
            if i in [2, 5, 12, 16]:  # 这些索引对应特定的中间层输出
                outputs.append(x)
        return outputs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_impl(x)


@BACKBONES.register('OfficialMobileNetV2')
class OfficialMobileNetV2(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        input_ch = cfg['input_channel'] * cfg['input_frames']
        self.down_scale = int(cfg['down_scale'])
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)

        self.base_model = MobileNetV2(input_channel=input_ch)

        # 动态计算通道数
        with torch.no_grad():
            img_size = cfg.get('img_size', 512)
            dummy_input = torch.randn(1, input_ch, img_size, img_size)
            stages_output = self.base_model(dummy_input)

            self.output_channels = 0
            for item in stages_output:
                self.output_channels += item.size(1)

    def forward(self, inputs):
        if inputs.size(1) == 1 and self.base_model.input_channel == 3:
            inputs = inputs.repeat(1, 3, 1, 1)

        b, c, h, w = inputs.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        stages_output = self.base_model(inputs)

        x = []
        for feature in stages_output:
            x.append(F.interpolate(feature, size=output_size, mode="bilinear"))

        x = torch.cat(x, dim=1)

        return x


if __name__ == '__main__':
    # 测试代码
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_yaml = os.path.join(script_dir, 'config/detect_train_cola.yaml')
    yaml_list = [train_yaml]
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 测试3通道输入
    print("Testing 3-channel input...")
    model = OfficialMobileNetV2(cfg_data['model_params'])
    output = model(torch.zeros((2, 3, 512, 640)))
    print(f"Output shape for 3-channel input: {output.shape}")

    # 测试1通道输入
    print("\nTesting 1-channel input...")
    cfg_data['model_params']['input_channel'] = 1
    model_grayscale = OfficialMobileNetV2(cfg_data['model_params'])
    output = model_grayscale(torch.zeros((2, 1, 512, 640)))
    print(f"Output shape for 1-channel input: {output.shape}")
