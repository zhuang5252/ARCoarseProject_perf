import timm
import torch
import torch.nn as nn


def tsm(tensor, duration, dilation=1):
    """
    时间分割注意力机制（TSM）

    tensor: [N*T, C, H, W] N 是批量大小，T 是时间步数（帧数），C 是通道数，H 是高度，W 是宽度
    duration: 时间步数 T，即每个样本在时间维度上的长度
    dilation: 膨胀系数，用于控制时间分割的间隔，默认值为 1
    """
    size = tensor.size()  # torch.Size([4, 64, 256, 256]) 4 为 N*T
    tensor = tensor.view((-1, duration) + size[1:])  # tensor [N, T, C, H, W]
    # 分割通道
    shift_size = size[1] // 8  # C // 8
    # mix_tensor, peri_tensor = C*1/8, C*7/8
    mix_tensor, peri_tensor = tensor.split([shift_size, 7 * shift_size], dim=2)
    # 在时间维度 T 上进行移位，具体是将时间维度的第一个元素与第二个元素交换位置
    mix_tensor = mix_tensor[:, (1, 0), :, :]

    # 合并并恢复到[N*T, C, H, W]
    return torch.cat((mix_tensor, peri_tensor), dim=2).view(size)


def add_tsm_to_module(obj, duration, dilation=1):
    """
    在原本obj之前加入一层时间分割注意力机制（TSM）
    """

    orig_forward = obj.forward

    def updated_forward(*args, **kwargs):
        a = (tsm(args[0], duration=duration, dilation=dilation),) + args[1:]  # args[1:] 为 ()空
        return orig_forward(*a, **kwargs)

    obj.forward = updated_forward

    return obj


class TrEstimatorTsm(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_channel']
        duration = cfg['duration']  # 时间步数
        self.weight_scale = cfg['weight_scale']

        # self.base_model共有4个layer
        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            in_chans=input_depth,
                                            pretrained=pretrained)

        # 对每一个layer中每一层的conv1加入时间分割注意力机制（TSM）
        if 'resnet' in base_model_name:
            for l in self.base_model.layer1:
                add_tsm_to_module(l.conv1, duration)

            for l in self.base_model.layer2:
                add_tsm_to_module(l.conv1, duration)

            for l in self.base_model.layer3:
                add_tsm_to_module(l.conv1, duration)

            for l in self.base_model.layer4:
                add_tsm_to_module(l.conv1, duration)

        out_ch = self.base_model.feature_info.channels()[-1]  # 512

        # 特征融合与回归头
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(out_ch * 2, 512, 3, padding=1),  # 前后帧特征融合
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 参数回归
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出dx, dy, angle
        )

        # 参数初始化
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, prev_frame, cur_frame):
        # 输入处理 (batch_size, 2, C, H, W)
        if prev_frame.shape[1] != 1:
            prev_frame = prev_frame[:, :1, :, :]
        if cur_frame.shape[1] != 1:
            cur_frame = cur_frame[:, :1, :, :]

        combined = torch.cat([prev_frame, cur_frame], dim=0)  # [N*2, C, H, W]

        # 特征提取
        features = self.base_model(combined)[-1]  # 取最后层特征
        B = prev_frame.size(0)
        prev_feat, cur_feat = features[:B], features[B:]  # 分割前后帧特征

        # 特征融合
        fused = torch.cat([prev_feat, cur_feat], dim=1)  # 通道维度拼接
        fused = self.feature_fusion(fused)

        # 参数回归
        params = self.regressor(fused)

        dx = params[:, 0]
        dy = params[:, 1]
        angle = params[:, 2]

        return dx, dy, angle


def print_summary():
    model = TrEstimatorTsm(cfg={'base_model_name': 'resnet34', 'weight_scale': 3, 'input_channel': 3, 'duration': 2})
    heatmap, offsets = model(torch.zeros((2, 3, 512, 640)), torch.zeros((2, 3, 512, 640)))
    print(heatmap.shape, offsets.shape)


if __name__ == "__main__":
    print_summary()
