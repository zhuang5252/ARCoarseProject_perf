"""
导出完整 `Model`（backbone+fc_comb+head）为 ONNX，并生成一个轻量化变体的 ONNX。

流程：
 1. 读取配置文件（默认使用 `air_track/detector/config/detect_train_zbzx.yaml`）
 2. 构建 `Model`，导出原始 ONNX
 3. 替换 `model.fc_comb` 为轻量化模块（先降通道，再 depthwise，再 pointwise），导出轻量化 ONNX
 4. 若检测到 TensorRT+pycuda，可调用 `tools/benchmark_tensorrt.py` 对两个 ONNX 做基准；否则输出运行说明

用法示例：
  python tools/export_model_full_and_light.py --cfg air_track/detector/config/detect_train_zbzx.yaml \
      --out_orig /tmp/orig.onnx --out_light /tmp/light.onnx --reduce_ch 128 --opset 13

"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from air_track.utils import combine_load_cfg_yaml
from air_track.detector.model.model import Model


class LightCombine(torch.nn.Module):
    """轻量化的 combine 模块：先 1x1 降通道 -> depthwise 3x3 -> pointwise 1x1 恢复/输出"""

    def __init__(self, in_ch, reduce_ch=128, out_ch=512):
        super().__init__()
        self.reduce = torch.nn.Conv2d(in_ch, reduce_ch, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.dw = torch.nn.Conv2d(reduce_ch, reduce_ch, kernel_size=3, padding=1, groups=reduce_ch)
        self.pw = torch.nn.Conv2d(reduce_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.reduce(x)
        x = self.relu(x)
        x = self.dw(x)
        x = self.relu(x)
        x = self.pw(x)
        return x


def export_model(model, dummy_input, out_path, opset=13, device='cpu'):
    model.eval()
    model.to(device)
    dummy = dummy_input.to(device)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 'output': {0: 'batch', 2: 'height', 3: 'width'}},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='air_track/detector/config/detect_train_zbzx.yaml')
    parser.add_argument('--out_orig', default='/tmp/orig_model_full.onnx')
    parser.add_argument('--out_light', default='/tmp/light_model_full.onnx')
    parser.add_argument('--reduce_ch', type=int, default=128)
    parser.add_argument('--reduce_stage_ch', type=int, default=None, help='optional: override cfg reduce_stage_ch to project each stage before upsample')
    parser.add_argument('--opset', type=int, default=13)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f'cfg file not found: {args.cfg}')

    cfg_data = combine_load_cfg_yaml(yaml_paths_list=[args.cfg])
    cfg = cfg_data['model_params']

    # allow CLI override for reduce_stage_ch
    if args.reduce_stage_ch is not None:
        cfg['reduce_stage_ch'] = int(args.reduce_stage_ch)

    # build model (Model will build backbone+fc_comb+head)
    print('Building Model from config...')
    model = Model(cfg, pretrained=False)

    # determine input shape from dataset config
    ds = cfg_data.get('dataset_params', {})
    img_size = ds.get('img_size', [640, 512])
    # dataset img_size stored as [w,h]
    if isinstance(img_size, list) and len(img_size) >= 2:
        W, H = img_size[0], img_size[1]
    else:
        W, H = 640, 512

    in_ch = cfg.get('input_channel', 1) * cfg.get('input_frames', 1)
    dummy = torch.randn(1, in_ch, H, W)

    print('Exporting original full model to', args.out_orig)
    export_model(model, dummy, args.out_orig, opset=args.opset, device=args.device)

    # create light variant by replacing fc_comb
    if model.fc_comb is None:
        print('Model has no fc_comb, creating a light combine and inserting...')
    base_ch = getattr(model.base_model, 'output_channels', None)
    if base_ch is None:
        raise RuntimeError('Cannot determine base_model.output_channels')

    reduce_ch = args.reduce_ch
    out_ch = cfg.get('combine_outputs_dim', 512)
    light = LightCombine(base_ch, reduce_ch=reduce_ch, out_ch=out_ch)
    model_light = Model(cfg, pretrained=False)
    model_light.fc_comb = light

    print('Exporting light model to', args.out_light)
    export_model(model_light, dummy, args.out_light, opset=args.opset, device=args.device)

    # attempt to run TRT benchmark if available
    try:
        import tensorrt as trt  # noqa: F401
        import subprocess
        print('TensorRT detected; running benchmark on both ONNX (fp16)...')
        cmd = f'python tools/benchmark_tensorrt.py --onnx {args.out_orig} --precision fp16 --batch 1 --iters 200'
        subprocess.run(cmd, shell=True, check=False)
        cmd2 = f'python tools/benchmark_tensorrt.py --onnx {args.out_light} --precision fp16 --batch 1 --iters 200'
        subprocess.run(cmd2, shell=True, check=False)
    except Exception:
        print('TensorRT or pycuda not available in this environment; skipped runtime benchmark.')
        print('You can run the benchmark on Orin with:')
        print(f'  python tools/benchmark_tensorrt.py --onnx {args.out_orig} --precision fp16 --batch 1 --iters 200')
        print(f'  python tools/benchmark_tensorrt.py --onnx {args.out_light} --precision fp16 --batch 1 --iters 200')


if __name__ == '__main__':
    main()
