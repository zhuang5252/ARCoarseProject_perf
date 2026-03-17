import os
import torch
import timeit
import numpy as np
import onnxruntime as ort
from typing import List, Optional

from air_track.classifier.model.model import Model
from air_track.utils import combine_load_cfg_yaml, load_model, check_and_change_img_size, auto_import_module


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

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        ONNX模型推理

        参数:
            input_data: 输入数据 (numpy数组)

        返回:
            模型输出
        """
        input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]

        outputs = self.session.run(output_names, {input_name: input_data})

        return outputs[0]


class Predictor:
    def __init__(self, yaml_list: List[str]):
        # 合并若干个yaml的配置文件内容
        self.cfg = combine_load_cfg_yaml(yaml_paths_list=yaml_list)
        self.dataset_params = self.cfg['dataset_params']
        self.model_params = self.cfg['model_params']
        self.model_path = self.cfg['model_path']

        self.classes = list(self.cfg['classes'])
        self.max_pixel = self.cfg['max_pixel']
        self.normalize = self.dataset_params['normalize']
        self.input_channel = self.model_params['input_channel']
        self.img_read_method = self.dataset_params['img_read_method'].lower()
        self.img_size_w, self.img_size_h = self.dataset_params['img_size']
        self.return_torch_tensors = self.dataset_params['return_torch_tensors']
        self.device = self.cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.onnx_predictor = None
        self.model_format = None  # 'pt系列' 或 'onnx'

    def set_model(self, model_path: Optional[str] = None):
        """set model"""
        if model_path is not None:
            self.model_path = model_path

        print('The model path is: ', self.model_path)
        self.model_format = self.model_path.split('.')[-1]

        if self.model_format == 'onnx':
            self.onnx_predictor = ONNXPredictor(self.model_path, device=self.device)
        else:
            model = Model(self.model_params, pretrained=self.model_params['pretrained'])
            # 获取模型权重
            self.model = load_model(model, self.model_path, device=self.device)

        return self.model if self.model_format != 'onnx' else self.onnx_predictor

    def process_single_input(self, data):
        """process single input"""
        # 输入为CHW需转换为HWC，才能resize
        if len(data.shape) == 3 and data.shape[0] == self.input_channel:
            data = np.transpose(data, (1, 2, 0))
        if self.normalize:
            data = data / self.max_pixel

        # 校验并调整图像尺寸
        data = check_and_change_img_size(data, img_size_w=self.img_size_w, img_size_h=self.img_size_h)
        # HWC需转换为CHW，才能输入模型
        if len(data.shape) == 3 and data.shape[-1] == self.input_channel:
            data = np.transpose(data, (2, 0, 1))
        else:
            data = np.expand_dims(data, axis=0)

        # 针对onnx和pt的不同处理
        if self.model_format != 'onnx':
            if type(data) == np.ndarray:
                data = torch.from_numpy(data).to(self.device)
            else:
                data = data.to(self.device)
        else:
            if type(data) != np.ndarray:
                data = data.cpu().numpy()

        return data

    def process_input(self, data):
        """process input"""
        input_data = []
        if len(data.shape) == 2:
            input_data = self.process_single_input(data)
            if self.model_format != 'onnx':
                input_data = input_data.unsqueeze(0).unsqueeze(0).float().to(self.device)
            else:
                input_data = np.expand_dims(input_data, 0).astype(np.float32)
        elif len(data.shape) == 3:
            input_data = self.process_single_input(data)
            if self.model_format != 'onnx':
                input_data = input_data.unsqueeze(0).float().to(self.device)
            else:
                input_data = np.expand_dims(input_data, 0).astype(np.float32)
        elif len(data.shape) == 4:
            for item in data:
                temp = self.process_single_input(item)
                if len(temp.shape) == 2:
                    temp = temp.unsqueeze(0)
                input_data.append(temp)
            if self.model_format != 'onnx':
                input_data = torch.stack(input_data, dim=0).float().to(self.device)
            else:
                input_data = np.stack(input_data, axis=0).astype(np.float32)
        else:
            raise ValueError('The input data shape is not supported.')

        return input_data

    def process_single_output(self, output):
        """process output"""
        if self.model_format == 'onnx':
            cls_idx = np.argmax(output, axis=1)
        else:
            cls_idx = output.argmax(dim=1)
        cls_name = self.classes[cls_idx.item()]

        return cls_name

    def process_output(self, output):
        """process output"""
        cls_name_list = []
        for output_item in output:
            cls_idx = np.argmax(output_item, axis=0)
            cls_name = self.classes[cls_idx.item()]
            cls_name_list.append(cls_name)

        return cls_name_list

    def predict(self, data):
        """model predict"""
        batch = data.shape[0]
        start_time = timeit.default_timer()

        if self.model_format == 'onnx':
            outputs = []
            if batch != 1:
                for item in data:
                    output = self.onnx_predictor.predict(np.expand_dims(item, axis=0))
                    outputs.append(output)
                outputs = np.concatenate(np.array(outputs), axis=0)
            else:
                outputs = self.onnx_predictor.predict(data)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(data)
            outputs = outputs.cpu().detach().numpy()

        end_time = timeit.default_timer()
        # 推理耗时
        inference_time = (end_time - start_time) * 1000

        cls_name_batch = self.process_output(outputs)

        return outputs, cls_name_batch, inference_time / batch


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)

    predictor.set_model()
