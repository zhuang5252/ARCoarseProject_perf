"""
导出仓库中 `OfficialResNet` 为 ONNX 的小脚本

用法示例：
  python tools/export_official_resnet_onnx.py --out resnet18.onnx --img-size 512 --batch 1

输出是网络最后 `torch.cat` 后的特征图（full resolution）。
"""
import argparse
import torch
from collections import defaultdict

from air_track.detector.model.backbones.official_resnet import OfficialResNet


def make_cfg(img_size=512, input_channel=3, input_frames=1):
    return {
        'base_model_name': 'resnet18',
        'input_channel': input_channel,
        'input_frames': input_frames,
        'down_scale': 1,
        'nb_classes': 1000,
        'img_size': img_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='输出 ONNX 文件路径')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--opset', type=int, default=13)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()

    cfg = make_cfg(img_size=args.img_size)
    model = OfficialResNet(cfg, pretrained=False)
    model.eval()

    dummy_input = torch.randn(args.batch, cfg['input_channel'] * cfg['input_frames'], args.img_size, args.img_size)

    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=['input'],
        output_names=['output'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 'output': {0: 'batch', 2: 'height', 3: 'width'}},
    )

    print('Exported', args.out)


if __name__ == '__main__':
    main()
