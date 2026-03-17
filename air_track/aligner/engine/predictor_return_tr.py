import torch
import timeit
from torch import nn
from air_track.aligner.model import resnet34_return_tr as resnet34
from air_track.utils import combine_load_cfg_yaml, numpy_2_tensor, load_model


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

    def set_model(self):
        """定义并load模型"""
        print('The model path is: ', self.model_path)
        model_params = self.cfg['model_params']
        self.model: nn.Module = resnet34.__dict__[model_params['model_cls']](cfg=model_params, pretrained=model_params['pretrained'])
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

    def predict(self, prev_img, cur_img):
        """模型预测"""
        self.model.eval()
        # 输入数据前处理
        prev_frame, cur_frame = self.process_input(prev_img, cur_img)

        start_time = timeit.default_timer()
        with torch.no_grad():
            transform = self.model(prev_frame, cur_frame)

        end_time = timeit.default_timer()
        # 推理耗时
        inference_time = (end_time - start_time) * 1000

        # 输出数据后处理
        transform = transform.detach().cpu().numpy()[0]

        return transform, inference_time
