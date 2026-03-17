import os
import torch
from typing import List, Tuple
import torch.nn.functional as F
from torch import Tensor

from air_track.detector.model.model import Model as DetectModel
from air_track.detector.utils.detect_utils import combine_images
from air_track.utils import combine_load_cfg_yaml


class Model2ONNX(DetectModel):
    """用于将双输入模型转换为ONNX格式的派生类，添加预处理和后处理"""

    def __init__(self, cfg, pretrained=False):
        super().__init__(cfg, pretrained=pretrained)

        # 定义裁剪参数
        self.original_size = (768, 1024)  # 原始输入尺寸 (H, W)
        self.crop_size = (512, 640)  # 中心裁剪尺寸 (H, W)
        self.model_input_size = (256, 320)  # 模型期望输入尺寸 (H, W)

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        """中心裁剪输入张量"""
        _, _, H, W = x.shape
        crop_h, crop_w = self.crop_size

        # 计算裁剪起始位置
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2

        # 执行中心裁剪
        return x[:, :, 128:128 + 512, 192:192 + 640]

    def _extract_first_channel(self, x: torch.Tensor) -> torch.Tensor:
        """提取第一个通道"""
        return x[:, 0:1, :, :]  # 保持4D张量形状 (B,1,H,W)

    def _resize_to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """调整到模型输入尺寸"""
        return F.interpolate(x, size=self.model_input_size, mode='bilinear', align_corners=False)

    # def _restore_original_size(self, x: torch.Tensor) -> torch.Tensor:
    #     """将输出恢复到原始尺寸"""
    #     _, _, H, W = x.shape
    #     target_h, target_w = self.original_size
    #
    #     # 计算需要填充的尺寸
    #     pad_h = (target_h - H) // 2
    #     pad_w = (target_w - W) // 2
    #
    #     # 执行对称填充 (上下左右填充0)
    #     return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

    def _restore_original_size(self, x: torch.Tensor) -> torch.Tensor:
        """将输出恢复到原始输入尺寸的一半（384x512）"""
        _, _, H, W = x.shape
        target_h = self.original_size[0] // 2  # 768 / 2 = 384
        target_w = self.original_size[1] // 2  # 1024 / 2 = 512

        # 计算需要填充的像素数（对称填充）
        pad_h = (target_h - H) // 2
        pad_w = (target_w - W) // 2

        # 执行填充（上下左右填充0）
        return F.pad(
            x,
            (pad_w, pad_w, pad_h, pad_h),  # 顺序：左、右、上、下
            mode='constant',
            value=0
        )

    def forward(self, *inputs) -> List[Tensor]:
        """重写前向传播方法，添加预处理和后处理"""
        # 1. 处理输入
        if len(inputs) == 1:
            # 单输入模式
            cur_frame = inputs[0]
            prev_frames = []
        elif len(inputs) == 2:
            # 双输入模式
            prev_frames, cur_frame = inputs
            if not isinstance(prev_frames, List):
                prev_frames = [prev_frames]
        else:
            raise ValueError(f"Expected 1 or 2 inputs, got {len(inputs)}")

        # 2. 预处理当前帧
        # 中心裁剪 -> 取第一个通道 -> 调整到模型输入尺寸
        cur_frame = self._center_crop(cur_frame)
        cur_frame = self._extract_first_channel(cur_frame)
        # cur_frame = self._resize_to_model_input(cur_frame)

        # 3. 预处理前一帧（如果存在）
        processed_prev_frames = []
        for prev_frame in prev_frames:
            prev_frame = self._center_crop(prev_frame)
            prev_frame = self._extract_first_channel(prev_frame)
            # prev_frame = self._resize_to_model_input(prev_frame)
            processed_prev_frames.append(prev_frame)

        # 4. 合并输入并调用父类forward
        if processed_prev_frames:
            combined_input = combine_images(processed_prev_frames, cur_frame)
        else:
            combined_input = cur_frame

        mask, size = super().forward(combined_input)

        # 5. 后处理：将输出恢复到原始尺寸的一半
        mask = self._restore_original_size(mask)
        size = self._restore_original_size(size)

        return [mask, size]


def export_to_onnx(model_params: dict, checkpoint_path: str, output_path: str,
                   input_names: List, output_names: List, input_data,
                   opset_version: int = 11):
    """导出模型到ONNX格式的辅助函数

    Args:
        model_params: 模型配置参数
        checkpoint_path: 预训练权重路径
        output_path: ONNX输出路径
        input_names: ONNX输入名称列表
        output_names: ONNX输出名称列表
        input_data: 输入数据
        opset_version: ONNX opset版本
    """
    # 创建模型实例
    model = Model2ONNX(cfg=model_params, pretrained=False)

    # 加载预训练权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 这里严格要求权重与模型定义一致
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Warning: No checkpoint provided, using random weights")

    # 导出模型
    torch.onnx.export(
        model,
        input_data,
        output_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
    )
    print(f"Successfully exported ONNX model to {output_path}")


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 配置文件路径
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')

    # 读取yaml文件
    yaml_list = [pred_yaml]
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 配置模型参数
    model_params = cfg_data['model_params']
    model_params.update({
        'input_frames': 1,  # 双输入模式
        'input_channel': 1,  # 单通道输入
        'backbone_type': 'official_resnet',
        'backbone_name': 'OfficialResNet',
        'base_model_name': 'ResNet18',
        'head_name': 'Head_2_Output',
        'return_feature_map': False,
        'nb_classes': 19
    })

    # 设置路径
    model_dir = '/media/linana/1276341C76340351/csz/github/AirTrack/model_saved/official_resnet/OfficialResNet/ResNet18/0812_T18_ir_640_512'
    checkpoint_path = os.path.join(model_dir, 'minimum_loss.pt')
    output_path = os.path.join(model_dir, '0812_T18_ir_epoch_15_fix.onnx')

    # 设置输入输出名称
    input_names = ['cur_frame']
    output_names = ['mask', 'size']

    # input_shape = (1, 3, 576, 768)
    input_shape = (1, 3, 768, 1024)
    # 创建示例输入
    batch_size, channels, height, width = input_shape
    cur_img = torch.rand(batch_size, channels, height, width)

    input_data = cur_img

    # 导出ONNX模型
    export_to_onnx(
        model_params=model_params,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        input_data=input_data,
        opset_version=11
    )
