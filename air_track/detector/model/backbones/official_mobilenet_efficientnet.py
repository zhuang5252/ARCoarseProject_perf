# official_mobilenet_efficientnet.py
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from air_track.utils import combine_load_cfg_yaml
from air_track.utils.registry import BACKBONES


# --------------------------- 工具函数 --------------------------- #
def _override_first_conv(module: nn.Module, in_ch: int) -> None:
    """
    将 torchvision BackBone 的首层 Conv2d 输入通道数改为 in_ch。
    （Mobilenet / EfficientNet 首层卷积皆位于 features[0][0] ）
    """
    if isinstance(module, models.MobileNetV2):
        old_conv: nn.Conv2d = module.features[0][0]
        module.features[0][0] = nn.Conv2d(
            in_ch, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
    elif isinstance(module, models.EfficientNet):
        old_conv: nn.Conv2d = module.features[0][0]
        module.features[0][0] = nn.Conv2d(
            in_ch, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
    else:
        raise ValueError("Unsupported backbone type when overriding first conv.")


def _forward_collect_stages(x, feature_blocks: nn.Sequential,
                            out_indices: List[int]) -> List[torch.Tensor]:
    """沿着 Sequential 逐层前向，并在 out_indices 记录输出"""
    outs, cur = [], x
    for idx, blk in enumerate(feature_blocks):
        cur = blk(cur)
        if idx in out_indices:
            outs.append(cur)
    return outs


# --------------------------- MobileNetV2 --------------------------- #
@BACKBONES.register("OfficialMobileNetV2")
class OfficialMobileNetV2(nn.Module):
    """
    base_model_name: mobilenet_v2
    """
    def __init__(self, cfg, pretrained: bool = False):
        super().__init__()

        # ---- config ---- #
        self.down_scale = int(cfg["down_scale"])
        self.combine_outputs_dim = cfg.get("combine_outputs_dim", -1)
        input_ch = cfg["input_channel"] * cfg["input_frames"]
        base_model_name = cfg["base_model_name"].lower()

        assert base_model_name == "mobilenet_v2", \
            f"base_model_name 应设为 mobilenet_v2, got {base_model_name}"

        # ---- backbone ---- #
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        _override_first_conv(self.backbone, input_ch)

        # 输出 1/4, 1/8, 1/16, 1/32 四个尺度特征
        # 对应 feature 索引: 3,6,10,17
        self.out_indices = [3, 6, 10, 17]

        # ---- 计算输出通道 ---- #
        with torch.no_grad():
            img_size = cfg.get("img_size", 512)
            dummy = torch.randn(1, input_ch, img_size, img_size)
            feats = self._extract_stages(dummy)
            self.output_channels = sum(f.size(1) for f in feats)

    # --------- 内部函数 --------- #
    def _extract_stages(self, x):
        return _forward_collect_stages(x, self.backbone.features, self.out_indices)

    # --------- Forward --------- #
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        out_h, out_w = h // self.down_scale, w // self.down_scale
        feats = self._extract_stages(x)
        feats = [F.interpolate(f, size=(out_h, out_w), mode="bilinear") for f in feats]
        return torch.cat(feats, dim=1)


# --------------------------- EfficientNet --------------------------- #
_EFFNET_FACTORY = {
    f"efficientnet_b{i}": getattr(models, f"efficientnet_b{i}") for i in range(0, 8)
}


@BACKBONES.register("OfficialEfficientNet")
class OfficialEfficientNet(nn.Module):
    """
    base_model_name: efficientnet_b0 ~ efficientnet_b7
    """
    _def_out_indices = {
        # 索引经实验挑选，覆盖 4 个分辨率层次
        # (low  -> high channels)
        "efficientnet_b0": [2, 3, 4, 6],
        "efficientnet_b1": [2, 3, 5, 7],
        "efficientnet_b2": [2, 4, 6, 8],
        "efficientnet_b3": [2, 4, 6, 9],
        "efficientnet_b4": [3, 5, 7, 11],
        "efficientnet_b5": [3, 5, 8, 12],
        "efficientnet_b6": [3, 6, 9, 14],
        "efficientnet_b7": [4, 7, 10, 16],
    }

    def __init__(self, cfg, pretrained: bool = False):
        super().__init__()

        # ---- config ---- #
        self.down_scale = int(cfg["down_scale"])
        self.combine_outputs_dim = cfg.get("combine_outputs_dim", -1)
        input_ch = cfg["input_channel"] * cfg["input_frames"]
        self.base_model_name = cfg["base_model_name"].lower()

        assert self.base_model_name in _EFFNET_FACTORY, \
            f"Unsupported base_model_name {self.base_model_name}"

        # ---- backbone ---- #
        self.backbone = _EFFNET_FACTORY[self.base_model_name](pretrained=pretrained)
        _override_first_conv(self.backbone, input_ch)
        self.out_indices = self._def_out_indices[self.base_model_name]

        # ---- 计算输出通道 ---- #
        with torch.no_grad():
            img_size = cfg.get("img_size", 512)
            dummy = torch.randn(1, input_ch, img_size, img_size)
            feats = self._extract_stages(dummy)
            self.output_channels = sum(f.size(1) for f in feats)

    # --------- 内部函数 --------- #
    def _extract_stages(self, x):
        return _forward_collect_stages(x, self.backbone.features, self.out_indices)

    # --------- Forward --------- #
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        out_h, out_w = h // self.down_scale, w // self.down_scale
        feats = self._extract_stages(x)
        feats = [F.interpolate(f, size=(out_h, out_w), mode="bilinear") for f in feats]
        return torch.cat(feats, dim=1)


# --------------------------- Demo --------------------------- #
if __name__ == "__main__":
    # 如同 original_resnet.py 的自检
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yaml_path = os.path.join(repo_root, "config/detect_train_test.yaml")
    cfg_data = combine_load_cfg_yaml([yaml_path])["model_params"]

    # MobileNetV2 Test
    cfg_data["base_model_name"] = "mobilenet_v2"
    model = OfficialMobileNetV2(cfg_data, pretrained=False)
    out = model(torch.zeros(2, cfg_data["input_channel"] * cfg_data["input_frames"], 512, 640))
    print("MobileNetV2 Output:", out.shape, "Total C =", model.output_channels)

    # EfficientNet‑B0 Test
    cfg_data["base_model_name"] = "efficientnet_b0"
    model = OfficialEfficientNet(cfg_data, pretrained=False)
    out = model(torch.zeros(2, cfg_data["input_channel"] * cfg_data["input_frames"], 128, 128))
    print("EfficientNet‑B0 Output:", out.shape, "Total C =", model.output_channels)
