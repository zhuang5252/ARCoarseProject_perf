import os
import torch
import timeit
import numpy as np
from torch import nn
import onnxruntime as ort
from air_track.aligner.utils.transform_utils import gen_transform
from air_track.aligner.model import resnet34_return_dx_dy_angle as resnet34
from air_track.utils import combine_load_cfg_yaml, numpy_2_tensor, load_model


class ONNXPredictor:
    """ONNX模型推理器"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化ONNX推理器

        参数:
            model_path: ONNX模型路径
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.device = device.lower()
        self.session = self._load_onnx_model(model_path)
        self.input_names = [_input.name for _input in self.session.get_inputs()]
        self.output_names = [_output.name for _output in self.session.get_outputs()]

    def _load_onnx_model(self, model_path: str) -> ort.InferenceSession:
        """加载ONNX模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {model_path}")

        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

    def predict(self, prev_img: np.ndarray, cur_img: np.ndarray):
        """
        ONNX模型推理

        参数:
            prev_img: 前一帧图像 (numpy数组)
            cur_img: 当前帧图像 (numpy数组)

        返回:
            (dx, dy, angle) 模型输出
        """
        inputs = {
            self.input_names[0]: prev_img,
            self.input_names[1]: cur_img
        }
        item_dx_dy_angle = self.session.run(self.output_names, inputs)
        dx, dy, angle = item_dx_dy_angle[0][0]

        return dx, dy, angle


class Predictor:
    def __init__(self, yaml_list: list):
        # 合并若干个yaml的配置文件内容
        self.cfg = combine_load_cfg_yaml(yaml_paths_list=yaml_list)
        self.model_path = self.cfg['model_path']
        self.dataset_params = self.cfg['dataset_params']
        self.img_size_w, self.img_size_h = self.dataset_params['img_size']
        self.max_pixel = self.cfg['max_pixel']
        self.device = self.cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.onnx_predictor = None
        self.model_format = None  # 'pt系列' 或 'onnx'

    def set_model(self, model_path=None):
        """定义并load模型"""
        if model_path is not None:
            self.model_path = model_path
        print('The model path is: ', self.model_path)
        self.model_format = self.model_path.split('.')[-1].lower()

        if self.model_format == 'onnx':
            self.onnx_predictor = ONNXPredictor(self.model_path, device=self.device)
            return self.onnx_predictor
        else:
            model_params = self.cfg['model_params']
            self.model: nn.Module = resnet34.__dict__[model_params['model_cls']](cfg=model_params,
                                                                                 pretrained=model_params['pretrained'])
            self.model = load_model(self.model, self.model_path, device=self.device)

            return self.model

    def process_input(self, prev_img, cur_img):
        """输入数据前处理"""
        if len(cur_img.shape) == 2 and len(prev_img.shape) == 2:
            cur_tensor_img = numpy_2_tensor(cur_img, self.max_pixel).unsqueeze(0).unsqueeze(0).to(self.device)
            prev_tensor_img = numpy_2_tensor(prev_img, self.max_pixel).unsqueeze(0).unsqueeze(0).to(self.device)
        elif len(cur_img.shape) == 3 and len(prev_img.shape) == 3:
            cur_tensor_img = numpy_2_tensor(cur_img, self.max_pixel).permute(2, 0, 1).unsqueeze(0).to(self.device)
            prev_tensor_img = numpy_2_tensor(prev_img, self.max_pixel).permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:  # len(cur_img.shape) == 4 and len(prev_img.shape) == 4
            cur_tensor_img = numpy_2_tensor(cur_img, self.max_pixel).permute(0, 3, 1, 2).to(self.device)
            prev_tensor_img = numpy_2_tensor(prev_img, self.max_pixel).permute(0, 3, 1, 2).to(self.device)

        return prev_tensor_img, cur_tensor_img

    def process_output(self, dx, dy, angle, shape):
        """输出数据后处理"""
        dx, dy, angle = dx.item(), dy.item(), angle.item()

        transform = gen_transform(shape, dx, dy, angle)

        return [dx, dy, angle], transform

    def predict(self, prev_img, cur_img):
        """模型预测"""
        # 输入数据前处理
        prev_frame, cur_frame = self.process_input(prev_img, cur_img)

        start_time = timeit.default_timer()
        if self.model_format == 'onnx':
            prev_frame = prev_frame.cpu().numpy()
            cur_frame = cur_frame.cpu().numpy()
            dx, dy, angle = self.onnx_predictor.predict(prev_frame, cur_frame)
        else:
            self.model.eval()
            with torch.no_grad():
                dx, dy, angle = self.model(prev_frame, cur_frame)

        end_time = timeit.default_timer()
        # 推理耗时
        inference_time = (end_time - start_time) * 1000

        # 输出数据后处理
        shape = (prev_frame.shape[-1], prev_frame.shape[-2])
        item_dx_dy_angle, transform = self.process_output(dx, dy, angle, shape)

        return item_dx_dy_angle, transform, inference_time
