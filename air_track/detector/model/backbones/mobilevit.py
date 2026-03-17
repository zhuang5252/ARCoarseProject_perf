import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, List, Tuple
from air_track.utils import combine_load_cfg_yaml
from air_track.utils.registry import BACKBONES


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            bias: bool = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvLayer(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    act_layer=nn.SiLU,
                )
            )
        layers.extend([
            # dw
            ConvLayer(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer,
                act_layer=nn.SiLU,
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            n_transformer_blocks: int = 2,
            head_dim: int = 32,
            attn_dropout: float = 0.1,
            dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            patch_h: int = 8,
            patch_w: int = 8,
            conv_ksize: int = 3,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_h * self.patch_w

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = transformer_dim // head_dim
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize
        self.dilation = dilation

        # Local representation
        self.conv_3x3 = ConvLayer(
            in_channels,
            in_channels,
            kernel_size=conv_ksize,
            stride=1,
            groups=in_channels,
            dilation=dilation,
            norm_layer=norm_layer,
            act_layer=nn.SiLU,
        )
        self.conv_1x1 = nn.Conv2d(in_channels, transformer_dim, 1, bias=False)

        # Global representations
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                transformer_dim,
                self.n_heads,
                head_dim,
                self.ffn_dim,
                attn_dropout,
                dropout,
                ffn_dropout,
            )
            for _ in range(self.n_blocks)
        ])

        # Fusion
        self.conv_proj = ConvLayer(
            transformer_dim,
            in_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=nn.SiLU,
        )

        # Unfold
        self.unfold = nn.Unfold(
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
        )

        # Fold
        self.fold = nn.Fold(
            output_size=(self.patch_h, self.patch_w),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        # Local representation
        x = self.conv_3x3(x)
        x = self.conv_1x1(x)

        # Convert feature map to patches
        batch_size, in_channels, orig_h, orig_w = x.shape
        patches = self.unfold(x)
        patches = patches.view(
            batch_size, in_channels, self.patch_h, self.patch_w, -1
        )
        patches = patches.permute(0, 4, 1, 2, 3)
        patches = patches.reshape(
            batch_size * self.patch_area, in_channels, self.patch_h, self.patch_w
        )

        # Learn global representations
        patches = self.transformer(patches)

        # Reshape patches to feature maps
        patches = patches.reshape(
            batch_size, self.patch_area, in_channels, self.patch_h, self.patch_w
        )
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, in_channels * self.patch_h * self.patch_w, -1)
        x = self.fold(patches)

        # Fusion
        x = self.conv_proj(x)
        x += res

        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            ffn_dim: int,
            attn_dropout: float = 0.1,
            dropout: float = 0.1,
            ffn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(ffn_dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B*N, C, H, W]
        batch_size, channels, height, width = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B*N, H*W, C]

        # Self-attention
        attn_out = self.attn(x, x, x)[0]
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        # Reshape back
        x = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        return x


class MobileViT(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            width_multiplier: float = 1.0,
            output_layers: Optional[List[int]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.output_layers = output_layers if output_layers is not None else [2, 4, 6]

        # MobileNet-like layers
        self.layer1 = nn.Sequential(
            ConvLayer(
                in_channels,
                _make_divisible(16 * width_multiplier, 8),
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                act_layer=nn.SiLU,
            ),
            InvertedResidual(
                _make_divisible(16 * width_multiplier, 8),
                _make_divisible(16 * width_multiplier, 8),
                stride=1,
                expand_ratio=1,
                norm_layer=norm_layer,
            ),
        )

        self.layer2 = nn.Sequential(
            InvertedResidual(
                _make_divisible(16 * width_multiplier, 8),
                _make_divisible(24 * width_multiplier, 8),
                stride=2,
                expand_ratio=4,
                norm_layer=norm_layer,
            ),
            InvertedResidual(
                _make_divisible(24 * width_multiplier, 8),
                _make_divisible(24 * width_multiplier, 8),
                stride=1,
                expand_ratio=2,
                norm_layer=norm_layer,
            ),
        )

        self.layer3 = nn.Sequential(
            InvertedResidual(
                _make_divisible(24 * width_multiplier, 8),
                _make_divisible(48 * width_multiplier, 8),
                stride=2,
                expand_ratio=4,
                norm_layer=norm_layer,
            ),
            MobileViTBlock(
                _make_divisible(48 * width_multiplier, 8),
                transformer_dim=_make_divisible(64 * width_multiplier, 8),
                ffn_dim=_make_divisible(128 * width_multiplier, 8),
                n_transformer_blocks=2,
                head_dim=32,
                patch_h=2,
                patch_w=2,
                norm_layer=norm_layer,
            ),
        )

        self.layer4 = nn.Sequential(
            InvertedResidual(
                _make_divisible(48 * width_multiplier, 8),
                _make_divisible(64 * width_multiplier, 8),
                stride=2,
                expand_ratio=4,
                norm_layer=norm_layer,
            ),
            MobileViTBlock(
                _make_divisible(64 * width_multiplier, 8),
                transformer_dim=_make_divisible(80 * width_multiplier, 8),
                ffn_dim=_make_divisible(160 * width_multiplier, 8),
                n_transformer_blocks=4,
                head_dim=32,
                patch_h=2,
                patch_w=2,
                norm_layer=norm_layer,
            ),
        )

        self.layer5 = nn.Sequential(
            InvertedResidual(
                _make_divisible(64 * width_multiplier, 8),
                _make_divisible(80 * width_multiplier, 8),
                stride=2,
                expand_ratio=4,
                norm_layer=norm_layer,
            ),
            MobileViTBlock(
                _make_divisible(80 * width_multiplier, 8),
                transformer_dim=_make_divisible(96 * width_multiplier, 8),
                ffn_dim=_make_divisible(192 * width_multiplier, 8),
                n_transformer_blocks=3,
                head_dim=32,
                patch_h=2,
                patch_w=2,
                norm_layer=norm_layer,
            ),
        )

        self.layer6 = nn.Sequential(
            InvertedResidual(
                _make_divisible(80 * width_multiplier, 8),
                _make_divisible(96 * width_multiplier, 8),
                stride=2,
                expand_ratio=4,
                norm_layer=norm_layer,
            ),
            MobileViTBlock(
                _make_divisible(96 * width_multiplier, 8),
                transformer_dim=_make_divisible(120 * width_multiplier, 8),
                ffn_dim=_make_divisible(240 * width_multiplier, 8),
                n_transformer_blocks=3,
                head_dim=32,
                patch_h=2,
                patch_w=2,
                norm_layer=norm_layer,
            ),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.layer1(x)
        if 1 in self.output_layers:
            features.append(x)

        x = self.layer2(x)
        if 2 in self.output_layers:
            features.append(x)

        x = self.layer3(x)
        if 3 in self.output_layers:
            features.append(x)

        x = self.layer4(x)
        if 4 in self.output_layers:
            features.append(x)

        x = self.layer5(x)
        if 5 in self.output_layers:
            features.append(x)

        x = self.layer6(x)
        if 6 in self.output_layers:
            features.append(x)

        return features


@BACKBONES.register('OfficialMobileViT')
class OfficialMobileViT(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        input_ch = cfg['input_channel'] * cfg['input_frames']
        self.down_scale = int(cfg['down_scale'])
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)

        # MobileViT configuration
        width_multiplier = cfg.get('width_multiplier', 1.0)
        output_layers = cfg.get('output_layers', [2, 4, 6])  # Default to 3 stages

        self.base_model = MobileViT(
            in_channels=input_ch,
            width_multiplier=width_multiplier,
            output_layers=output_layers,
        )

        # Calculate output channels
        with torch.no_grad():
            img_size = cfg.get('img_size', 256)
            dummy_input = torch.randn(1, input_ch, img_size, img_size)
            stages_output = self.base_model(dummy_input)

            self.output_channels = 0
            for item in stages_output:
                self.output_channels += item.size(1)

        # Store feature scales
        self.feature_scales = [2 ** (i + 1) for i, _ in enumerate(output_layers)]

    def forward(self, inputs):
        B, C, H, W = inputs.shape

        # Forward through MobileViT
        stages_output = self.base_model(inputs)

        # Upsample all features to the same size
        target_size = (H // self.down_scale, W // self.down_scale)
        upsampled_features = []
        for i, feature in enumerate(stages_output):
            upsampled = F.interpolate(
                feature,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        # Concatenate all features along channel dimension
        x = torch.cat(upsampled_features, dim=1)

        return x


if __name__ == '__main__':
    # 测试代码
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_yaml = os.path.join(script_dir, 'config/detect_train.yaml')
    yaml_list = [train_yaml]
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    model = OfficialMobileViT(cfg_data['model_params'])
    output = model(torch.zeros((2, 2, 256, 256)))  # MobileViT默认输入256x256
    print(output.shape)