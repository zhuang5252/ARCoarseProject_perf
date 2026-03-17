import os
import torch
from typing import List
from air_track.utils import combine_load_cfg_yaml
from air_track.classifier.model.model import Model as ClassifyModel


class Model2ONNX(ClassifyModel):
    """用于将二级分类模型转换为ONNX格式的派生类

    继承自原始Model类，但修改forward方法以接受两个输入:
    - prev_frame: 前一帧图像（为未来预留）
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
        2. 双输入: forward(prev_frame, cur_frame)（为未来预留）
        3. 多输入：forward(prev_frames, cur_frame)（为未来预留）

        Args:
            *inputs: 可以是一个张量(单输入)或两个张量(双输入)（为未来预留）

        Returns:
            torch.Tensor: 模型输出
        """
        # 处理输入
        if len(inputs) == 1:
            # 单输入模式，直接调用父类方法
            return super().forward(inputs[0])
        else:
            raise ValueError(f"Expected 1 input, got {len(inputs)}")


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
        'model_mode': 'predict',
        'model_cls': 'binary_resnet',
        'base_model_name': 'BinaryResNet',
        'input_channel': 1,  # 单通道输入
        'nb_classes': 2
    })

    # 设置路径
    model_dir = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/model_saved/second_classifier/binary_resnet/BinaryResNet/251203_FlySubjectAirToGround_64_64'
    checkpoint_path = os.path.join(model_dir, 'epoch_18.pt')
    output_path = os.path.join(model_dir, '251203_FlySubjectAirToGround_64_64.onnx')

    # 设置输入输出名称
    input_names = ['input']
    output_names = ['ouput']

    input_shape = (1, 1, 64, 64)
    # 创建示例输入
    batch_size, channels, height, width = input_shape
    data = torch.rand(batch_size, channels, height, width)

    input_data = data

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
