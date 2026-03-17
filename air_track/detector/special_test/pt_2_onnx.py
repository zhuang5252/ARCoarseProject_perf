import os
import torch
from typing import List
from air_track.detector.model.model import Model as DetectModel
from air_track.detector.utils.detect_utils import combine_images
from air_track.utils import combine_load_cfg_yaml


class Model2ONNX(DetectModel):
    """用于将双输入模型转换为ONNX格式的派生类

    继承自原始Model类，但修改forward方法以接受两个输入:
    - prev_frame: 前一帧图像
    - cur_frame: 当前帧图像

    在内部将两个输入合并为一个后再传递给父类的forward方法
    """

    def __init__(self, cfg, pretrained=False):
        """初始化方法

        Args:
            cfg (Dict): 配置字典，包含模型结构参数
            pretrained (bool): 是否加载预训练权重
        """
        super().__init__(cfg, pretrained=pretrained)

    def forward(self, *inputs) -> torch.Tensor:
        """重写前向传播方法

        本质支持两种调用方式:
        1. 单输入: forward(tensor) - 与基类兼容
        2. 双输入: forward(prev_frame, cur_frame)
        3. 多输入：forward(prev_frames, cur_frame)

        Args:
            *inputs: 可以是一个张量(单输入)或两个张量(双输入)

        Returns:
            torch.Tensor: 模型输出
        """
        # 处理输入
        if len(inputs) == 1:
            # 单输入模式，直接调用父类方法
            return super().forward(inputs[0])
        elif len(inputs) == 2:
            # 双输入模式，用于ONNX导出
            prev_frames, cur_frame = inputs

            if not isinstance(prev_frames, List):
                prev_frames = [prev_frames]

            # 验证输入形状
            for prev_frame in prev_frames:
                if prev_frame.dim() != 4 or cur_frame.dim() != 4:
                    raise ValueError(
                        f"Input tensors must be 4D (B,C,H,W), got shapes {prev_frame.shape} and {cur_frame.shape}")

            # 合并两个输入为一个张量
            combined_input = combine_images(prev_frames, cur_frame)

            # 调用父类的forward方法
            return super().forward(combined_input)
        else:
            raise ValueError(f"Expected 1 or 2 inputs, got {len(inputs)}")


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
        'nb_classes': 5
    })

    # 设置路径
    model_dir = '/home/csz_xishu2/PycharmProjects/github/zhuang5252/AirTrack/model_saved/detect_model/official_resnet/OfficialResNet/ResNet18/zbzx_coarse_visible_20260317'
    checkpoint_path = os.path.join(model_dir, 'minimum_loss.pt')
    output_path = os.path.join(model_dir, 'zbzx_coarse_visible_20260317.onnx')

    # 设置输入输出名称
    input_names = ['cur_frame']
    output_names = ['mask', 'size']

    # input_shape = (1, 3, 576, 768)
    input_shape = (1, 1, 1024, 1024)
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
