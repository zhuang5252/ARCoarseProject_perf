import os
import cv2
import torch
import timeit
import numpy as np
import onnxruntime as ort
from typing import Tuple, Union, List, Optional
from air_track.detector.model.model import Model
from air_track.detector.utils.detect_utils import pred_to_detections_2_output, pred_to_detections_3_output, \
    pred_to_detections_5_output, pred_to_detections_6_output, combine_images
from air_track.utils import combine_load_cfg_yaml, load_model, numpy_2_tensor, check_image_bit_depth, \
    check_and_change_img_size


def process_2_output(cfg, predictions, down_scale, target_min_size):
    """process output"""
    pred = {
        'mask': predictions[0],
        'size': predictions[1],
    }

    # pred结果包含5部分输出：size、offset、distance、tracking、mask
    pred['mask'] = pred['mask'][0]  # torch.Size([1, 256, 320])

    keys_list = list(pred.keys())
    for k in list(keys_list):
        if isinstance(pred[k], np.ndarray):
            pred[k] = pred[k][0].astype(np.float32)
        else:
            pred[k] = pred[k][0].float().cpu().detach().numpy()

    # 对模型输出的pred转为坐标
    detected_objects = pred_to_detections_2_output(
        classes=cfg['classes'],
        comb_pred=pred['mask'],
        size=pred['size'],
        conf_threshold=cfg['conf_threshold'],
        iou_threshold=cfg['iou_threshold'],
        down_scale=down_scale,
        target_min_size=target_min_size,
        x_pad=cfg['x_pad'],
        y_pad=cfg['y_pad'],
    )

    # 保留最大的max_nb_objects个目标
    detected_objects = detected_objects[:cfg['max_nb_objects']]

    return detected_objects


def process_3_output(cfg, predictions, down_scale, target_min_size):
    """process output"""
    pred = {
        'mask': predictions[0],
        'size': predictions[1],
        'cls': predictions[2],
    }

    # pred结果包含5部分输出：size、offset、distance、tracking、mask
    pred['mask'] = pred['mask'][0]  # torch.Size([1, 256, 320])

    keys_list = list(pred.keys())
    for k in list(keys_list):
        if isinstance(pred[k], np.ndarray):
            pred[k] = pred[k][0].astype(np.float32)
        else:
            pred[k] = pred[k][0].float().cpu().detach().numpy()

    # 对模型输出的pred转为坐标
    detected_objects = pred_to_detections_3_output(
        classes=cfg['classes'],
        comb_pred=pred['mask'],
        cls=pred['cls'],
        size=pred['size'],
        conf_threshold=cfg['conf_threshold'],
        iou_threshold=cfg['iou_threshold'],
        down_scale=down_scale,
        target_min_size=target_min_size,
        x_pad=cfg['x_pad'],
        y_pad=cfg['y_pad'],
    )

    # 保留最大的max_nb_objects个目标
    detected_objects = detected_objects[:cfg['max_nb_objects']]

    return detected_objects


def process_5_output(cfg, predictions, down_scale, target_min_size):
    """process output"""
    pred = {
        'mask': predictions[0],
        'size': predictions[1],
        'offset': predictions[2],
        'distance': predictions[3],
        'tracking': predictions[4],
    }

    # pred结果包含5部分输出：size、offset、distance、tracking、mask
    pred['mask'] = pred['mask'][0]  # torch.Size([1, 256, 320])

    keys_list = list(pred.keys())
    for k in list(keys_list):
        if isinstance(pred[k], np.ndarray):
            pred[k] = pred[k][0].astype(np.float32)
        else:
            pred[k] = pred[k][0].float().cpu().detach().numpy()

    # 对模型输出的pred转为坐标
    detected_objects = pred_to_detections_5_output(
        classes=cfg['classes'],
        comb_pred=pred['mask'],
        offset=pred['offset'],
        size=pred['size'],
        tracking=pred['tracking'],
        distance=pred['distance'][0],
        offset_scale=cfg['offset_scale'],
        conf_threshold=cfg['conf_threshold'],
        iou_threshold=cfg['iou_threshold'],
        down_scale=down_scale,
        target_min_size=target_min_size,
        x_pad=cfg['x_pad'],
        y_pad=cfg['y_pad'],
    )

    # 保留最大的max_nb_objects个目标
    detected_objects = detected_objects[:cfg['max_nb_objects']]

    return detected_objects


def process_6_output(cfg, predictions, down_scale, target_min_size):
    """process output"""
    pred = {
        'mask': predictions[0],
        'size': predictions[1],
        'offset': predictions[2],
        'distance': predictions[3],
        'tracking': predictions[4],
        'cls': predictions[5],
    }

    # pred结果包含5部分输出：size、offset、distance、tracking、mask
    pred['mask'] = pred['mask'][0]  # torch.Size([1, 256, 320])

    keys_list = list(pred.keys())
    for k in list(keys_list):
        if isinstance(pred[k], np.ndarray):
            pred[k] = pred[k][0].astype(np.float32)
        else:
            pred[k] = pred[k][0].float().cpu().detach().numpy()

    # 对模型输出的pred转为坐标
    detected_objects = pred_to_detections_6_output(
        classes=cfg['classes'],
        comb_pred=pred['mask'],
        cls=pred['cls'],
        offset=pred['offset'],
        size=pred['size'],
        tracking=pred['tracking'],
        distance=pred['distance'][0],
        offset_scale=cfg['offset_scale'],
        conf_threshold=cfg['conf_threshold'],
        iou_threshold=cfg['iou_threshold'],
        down_scale=down_scale,
        target_min_size=target_min_size,
        x_pad=cfg['x_pad'],
        y_pad=cfg['y_pad'],
    )

    # 保留最大的max_nb_objects个目标
    detected_objects = detected_objects[:cfg['max_nb_objects']]

    return detected_objects


class PtPredictor:
    def __init__(self, yaml_list: list):
        # 合并若干个yaml的配置文件内容
        self.cfg = combine_load_cfg_yaml(yaml_paths_list=yaml_list)
        self.dataset_params = self.cfg['dataset_params']
        self.model_params = self.cfg['model_params']
        self.model_path = self.cfg['model_path']

        self.head_nums = int(self.model_params['head_nums'])
        self.max_pixel = self.cfg['max_pixel']
        self.img_read_method = self.dataset_params['img_read_method'].lower()
        self.down_scale = self.model_params['down_scale']
        self.target_min_size = self.cfg['target_min_size']
        self.return_feature_map = self.model_params.get('return_feature_map', False)
        self.device = self.cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = None

        # 初始化后处理函数
        self._init_process_funcs()

    def _init_process_funcs(self):
        """延迟初始化函数映射（避免未实现的函数报错）"""
        self.process_funcs = {
            2: process_2_output,
            3: process_3_output,
            5: process_5_output,
            6: process_6_output,
        }
        assert self.head_nums in self.process_funcs, \
            f"Unsupported head_nums: {self.head_nums}"

    def set_model(self, model_path=None):
        """
        set model
        model_path=None，则使用self.cfg['model_path']
        """
        if model_path is not None:
            self.model_path = model_path
        print('The model path is: ', self.model_path)
        model_params = self.cfg['model_params']
        model = Model(cfg=model_params, pretrained=model_params['pretrained'])
        self.model = load_model(model, self.model_path, device=self.device)

        return self.model

    def get_prev_step_images(self,
                             path, file_names, index,
                             img_size_w=None, img_size_h=None,
                             frame_step=None, input_frames=None,
                             max_pixel=None
                             ):
        """
        从file_names中取出prev的数据

        存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
        """
        if not max_pixel:
            max_pixel = self.max_pixel
        if not frame_step:
            frame_step = self.dataset_params['frame_step']
        if not input_frames:
            input_frames = self.dataset_params['input_frames']
        if not img_size_w and not img_size_h:
            img_size_w, img_size_h = self.dataset_params['img_size']

        prev_images = []
        for i in range(frame_step, input_frames, frame_step):
            interval_index = i * frame_step

            file_name = file_names[index - interval_index]
            prev_img_path = os.path.join(path, file_name)

            if self.img_read_method == 'gray':
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
            elif self.img_read_method == 'unchanged':
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_UNCHANGED)
            else:
                prev_img = cv2.imread(prev_img_path)

            # 校验图像深度与配置中的max_pixel是否一致
            check_image_bit_depth(prev_img, prev_img_path, max_pixel)
            # 校验图像大小，如果图像大小与配置中的img_size不一致，则resize
            prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

            prev_images.append(prev_img)

        return prev_images

    def process_multi_frame_input(self, prev_images, cur_image):
        """process input"""
        # 当前帧
        cur_tensor_img = self.process_single_frame_input(cur_image)

        # prev_images可能不止一帧
        prev_tensor_images = []
        for prev_img in prev_images:
            prev_tensor_img = self.process_single_frame_input(prev_img)
            prev_tensor_images.append(prev_tensor_img)

        # 将cur数据与prev数据合并
        input_data = combine_images(prev_tensor_images, cur_tensor_img)

        input_data = input_data.float().to(self.device)

        return input_data

    def process_single_frame_input(self, img):
        """process input"""
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                img = numpy_2_tensor(img, self.max_pixel).unsqueeze(-1).unsqueeze(0)
            else:
                img = numpy_2_tensor(img, self.max_pixel).unsqueeze(0)
            img = img.permute(0, 3, 1, 2)
        elif isinstance(img, torch.Tensor):
            pass
        else:
            raise TypeError('img must be numpy.ndarray or torch.Tensor')
        input_data = img.float().to(self.device)

        return input_data

    def predict(self, data):
        """model predict"""
        self.model.eval()

        start_time = timeit.default_timer()
        # False禁用梯度跟踪
        with torch.set_grad_enabled(False):
            if self.return_feature_map:
                predictions, features = self.model(data)
            else:
                predictions = self.model(data)

        end_time = timeit.default_timer()
        # 推理耗时
        inference_time = (end_time - start_time) * 1000

        # 输出数据后处理(detected_objects为检测输出、classify_objects为二级分类器的输入候选目标)
        detected_objects = self.process_funcs[self.head_nums](
            self.cfg, predictions, self.down_scale, self.target_min_size
        )

        if self.return_feature_map:  # 返回特征图
            return features, predictions, detected_objects, inference_time
        else:
            return predictions, detected_objects, inference_time


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

        # 获取模型输入信息
        self.input_info = self._get_model_input_info()

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

    def _get_model_input_info(self) -> dict:
        """获取模型输入信息"""
        input_info = {}
        for _input in self.session.get_inputs():
            # 获取输入形状 (通常是 [batch, channel, height, width])
            shape = _input.shape

            # 确保形状是4维的(假设是图像输入)
            if len(shape) != 4:
                raise ValueError(f"输入形状应为4维 [batch, channel, height, width], 但得到的是 {shape}")

            # 获取通道数(第2维)
            channel = shape[1] if shape[1] is not None else 3  # 默认为3通道

            input_info[_input.name] = {
                'shape': shape,
                'channel': channel,
            }

        return input_info

    def predict(self, input_data: np.ndarray):
        """
        ONNX模型推理

        参数:
            input_data: 输入数据 (numpy数组), 形状应为 [batch, frame*channel, height, width]

        返回:
            outputs: 模型输出
        """
        # 检查输入数据维度
        if input_data.ndim != 4:
            raise ValueError(f"输入数据应为4维 [batch, frame * channel, height, width], 但得到的是 {input_data.shape}")

        # 准备ONNX输入数据
        onnx_input_data = {}
        output_names = [_output.name for _output in self.session.get_outputs()]

        # 计算模型所需总通道数
        all_channels = sum(info['channel'] for info in self.input_info.values())
        # 输入数据总通道数与ONNX模型输入总通道数不符，抛出异常
        if all_channels != input_data.shape[1]:
            raise ValueError(
                f"输入数据总通道数与ONNX模型输入总通道数不符，"
                f"ONNX模型所需总通道数为：{all_channels}，而输入数据总通道数为：{input_data.shape[1]}"
            )

        # 输入数据分配为onnx模型所支持的dict输入
        start_channel = 0
        for input_name, info in self.input_info.items():
            # 获取该输入需要的通道数
            required_channels = info['channel']

            end_channel = start_channel + required_channels
            try:
                onnx_input_data[input_name] = input_data[:, start_channel:end_channel, :, :]
            except IndexError:
                raise ValueError(
                    f"无法分配 {required_channels} 通道给输入 '{input_name}' "
                    f"(当前分配范围: {start_channel}:{end_channel})"
                )
            start_channel = end_channel

        # 执行推理
        start_time = timeit.default_timer()
        outputs = self.session.run(output_names, onnx_input_data)
        end_time = timeit.default_timer()

        inference_time = (end_time - start_time) * 1000  # ms

        return outputs, inference_time


class Predictor(PtPredictor):
    """混合模型预测器，支持PT和ONNX模型"""

    def __init__(self, yaml_list: List[str]):
        """
        初始化混合预测器

        参数:
            yaml_list: 配置文件列表
        """
        super().__init__(yaml_list)
        self.onnx_predictor = None
        self.model_format = None

    def set_model(self, model_path: Optional[str] = None) -> None:
        """
        加载模型

        参数:
            model_path: 模型路径 (None表示使用配置文件中的路径)
        """
        if model_path is not None:
            self.model_path = model_path
        self.model_format = self.model_path.split('.')[-1]

        if self.model_format == 'onnx':
            print('The model path is: ', self.model_path)
            self.onnx_predictor = ONNXPredictor(self.model_path, device=self.device)
        else:
            super().set_model(self.model_path)

    def predict(self, input_data: Union[torch.Tensor, np.ndarray]):
        """
        执行预测 (支持PT和ONNX模型)

        参数:
            input_data: 输入数据

        返回:
            pt:
                if self.return_feature_map:  # 返回特征图
                    return features, predictions, detected_objects, inference_time
                else:
                    return predictions, detected_objects, inference_time
            onnx:
                return predictions, detected_objects, inference_time
        """
        if self.model_format == 'onnx':
            # ONNX 模型推理
            if not isinstance(input_data, np.ndarray):
                input_data = input_data.cpu().numpy() if isinstance(input_data, torch.Tensor) else np.array(input_data)

            # onnx 模型始终不返回特征图
            predictions, inference_time = self.onnx_predictor.predict(input_data)

            # 检测结果处理 (保持与父类一致)
            detected_objects = self.process_funcs[self.head_nums](
                self.cfg, predictions, self.down_scale, self.target_min_size
            )

            return predictions, detected_objects, inference_time
        else:
            # PT模型推理 (使用父类方法)
            return super().predict(input_data)
