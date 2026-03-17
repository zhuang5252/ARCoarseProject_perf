import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from air_track.utils.registry import BACKBONES


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


#########################################
# ResNet系列BackBone: ResNet12 ~ ResNet50
#########################################
@BACKBONES.register('CustomizeResNet12')
class CustomizeResNet12(nn.Module):
    """
    自定义的轻量化 ResNet12：
    - 在 few-shot 学习里常见，也可以用于跟踪场景。
    - 大体包含4个Block，输出通道依次 [64, 128, 256, 512] 。
    - 你可根据实际需求减少通道数，如 [32, 64, 128, 256]。
    """

    def __init__(self, cfg, pretrained=False):
        super(CustomizeResNet12, self).__init__()
        if pretrained:
            # 目前没有官方的预训练权重，可以尝试自行加载自训练的权重
            print("Warning: ResNet12 has no official pretrained weights. 'pretrained=True' is for custom usage.")

        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.output_channels = 256

        # Input stage
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_ch, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # 4段Block，每段可能包含1或2个BasicBlock，这里只演示最简
        self.layer1 = self._make_layer(BasicBlock, 64, stride=1)  # 不下采样
        self.layer2 = self._make_layer(BasicBlock, 128, stride=2)  # 下采样 => 1/2
        self.layer3 = self._make_layer(BasicBlock, 256, stride=2)  # 再下采样
        self.layer4 = self._make_layer(BasicBlock, 512, stride=2)  # 再下采样

        # 你可以在后面加一个 reduce_conv = nn.Conv2d(512, 256, 1)
        self.reduce_conv = nn.Conv2d(512, self.output_channels, kernel_size=1)
        self.low_reduce = nn.Conv2d(64, self.output_channels, kernel_size=1)
        # (因为你项目常见 "返回 high_feat, low_feat" => 都做 256 维)

    def _make_layer(self, block, out_planes, stride=1):
        # 这里仅建一个 block (最小化), 你也可叠多个 block
        layer = block(self.in_planes, out_planes, stride=stride)
        self.in_planes = out_planes
        return layer

    def forward(self, x, pool_size=None, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        # stage0
        out = self.relu(self.bn1(self.conv1(x)))
        # low_feat: 这里将stage0 + layer1合并得到特征
        low_feat = self.layer1(out)

        # layer2,3,4 => high_feat
        out = self.layer2(low_feat)
        out = self.layer3(out)
        out = self.layer4(out)
        high_feat = out

        # reduce channels => 256
        high_feat = self.reduce_conv(high_feat)
        low_feat = self.low_reduce(low_feat)

        # 如果你跟 ResNet18 一样，需要 pool_size
        if pool_size is not None:
            high_feat = F.adaptive_avg_pool2d(high_feat, pool_size)
            low_feat = F.adaptive_avg_pool2d(low_feat, pool_size)

        # 上采样到标签尺寸
        if high_feat.shape[2:] != output_size:
            high_feat = F.interpolate(high_feat, size=output_size, mode='bilinear')
            low_feat = F.interpolate(low_feat, size=output_size, mode='bilinear')

        if return_multi:
            return high_feat, low_feat
        else:
            return high_feat


@BACKBONES.register('CustomizeResNet18')
class CustomizeResNet18(nn.Module):
    """
    修改版的 ResNet18 Backbone：
      - 返回多尺度特征，其中低层特征先经过 1×1 卷积升维到 256，
        高层特征同样经过 reduce_conv 降维到 256，
        然后在上层可以进行融合。
    """

    def __init__(self, cfg, pretrained=False):
        super(CustomizeResNet18, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.output_channels = 256

        if input_ch != 3:
            if pretrained:
                print("Warning: ResNet18官方预训练仅适用3通道，自动跳过第一层权重加载。")
            # 替换第一层 conv1 => input_channels
            old_weight = resnet.conv1.weight.data
            out_ch = resnet.conv1.out_channels
            resnet.conv1 = nn.Conv2d(input_ch, out_ch,
                                     kernel_size=7, stride=2, padding=3, bias=False)
            # 如果想“部分”拷贝 old_weight 的均值到新卷积层，这里可以自定义:
            with torch.no_grad():
                # 简单做法：把 3通道权重的均值作为1通道初始
                resnet.conv1.weight[:, 0, :, :] = old_weight.mean(dim=1)
        # 将 maxpool 替换为 Identity
        resnet.maxpool = nn.Identity()

        # 修改 layer2, layer3, layer4 的下采样：将 stride=2 修改为 stride=1
        def modify_layer(layer):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

        modify_layer(resnet.layer2)
        modify_layer(resnet.layer3)
        modify_layer(resnet.layer4)

        # 低层特征: layer1 输出通道 64
        self.features = nn.Sequential(
            resnet.conv1,  # 通常 kernel=7, stride=2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # 这里已为 Identity
            resnet.layer1  # 输出通道为 64
        )
        # 高层特征：layer2 ~ layer4 输出通道 512，之后 reduce_conv 降至 256
        self.high_level = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.reduce_conv = nn.Conv2d(512, self.output_channels, kernel_size=1)
        # 新增 low_reduce，将低层特征通道从 64 升维至 256
        self.low_reduce = nn.Conv2d(64, self.output_channels, kernel_size=1)

    def forward(self, x, pool_size=None, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        low_feat = self.features(x)  # 低层特征，shape: [B, 64, H, W]
        high_feat = self.high_level(low_feat)  # 高层特征, shape: [B, 512, H', W']
        high_feat = self.reduce_conv(high_feat)  # 降维到 256, shape: [B, 256, H', W']
        low_feat = self.low_reduce(low_feat)  # 升维到 256, shape: [B, 256, H, W]
        if pool_size is not None:
            high_feat = F.adaptive_avg_pool2d(high_feat, pool_size)
            low_feat = F.adaptive_avg_pool2d(low_feat, pool_size)

        # 上采样到标签尺寸
        if high_feat.shape[2:] != output_size:
            high_feat = F.interpolate(high_feat, size=output_size, mode='bilinear')
            low_feat = F.interpolate(low_feat, size=output_size, mode='bilinear')

        if return_multi:
            return high_feat, low_feat
        return high_feat


@BACKBONES.register('CustomizeResNet34')
class CustomizeResNet34(nn.Module):
    """
    使用 torchvision.models.resnet34 构造的 ResNet34 骨干，修改 maxpool 为 Identity，
    并将下采样层的 stride 调整为(1,1)，使得后续特征图尺寸更大，最后统一将高低层特征映射到256通道。
    """

    def __init__(self, cfg, pretrained=False):
        super(CustomizeResNet34, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)

        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.output_channels = 256

        if input_ch != 3:
            if pretrained:
                print("Warning: ResNet34官方预训练仅适用3通道。")
            old_weight = resnet.conv1.weight.data
            out_ch = resnet.conv1.out_channels
            resnet.conv1 = nn.Conv2d(input_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight[:, 0, :, :] = old_weight.mean(dim=1)
        resnet.maxpool = nn.Identity()

        def modify_layer(layer):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

        modify_layer(resnet.layer2)
        modify_layer(resnet.layer3)
        modify_layer(resnet.layer4)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.high_level = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.reduce_conv = nn.Conv2d(512, self.output_channels, kernel_size=1)
        self.low_reduce = nn.Conv2d(64, self.output_channels, kernel_size=1)

    def forward(self, x, pool_size=None, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        low_feat = self.features(x)
        high_feat = self.high_level(low_feat)
        high_feat = self.reduce_conv(high_feat)
        low_feat = self.low_reduce(low_feat)

        if pool_size is not None:
            high_feat = F.adaptive_avg_pool2d(high_feat, pool_size)
            low_feat = F.adaptive_avg_pool2d(low_feat, pool_size)

        # 上采样到标签尺寸
        if high_feat.shape[2:] != output_size:
            high_feat = F.interpolate(high_feat, size=output_size, mode='bilinear')
            low_feat = F.interpolate(low_feat, size=output_size, mode='bilinear')

        if return_multi:
            return high_feat, low_feat
        return high_feat


@BACKBONES.register('CustomizeResNet50')
class CustomizeResNet50(nn.Module):
    """
    使用 torchvision.models.resnet50 构造的 ResNet50 骨干。
    由于 ResNet50 的高层特征通道为2048，需要通过1x1卷积映射到256，
    低层特征（layer1）输出通道为256，直接通过1x1卷积保持统一。
    """

    def __init__(self, cfg, pretrained=False):
        super(CustomizeResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # 输入通道数
        input_ch = cfg['input_channel'] * cfg['input_frames']  # 2
        self.down_scale = int(cfg['down_scale'])
        self.output_channels = 256

        if input_ch != 3:
            if pretrained:
                print("Warning: ResNet50官方预训练仅适用3通道。")
            old_weight = resnet.conv1.weight.data
            out_ch = resnet.conv1.out_channels
            resnet.conv1 = nn.Conv2d(input_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight[:, 0, :, :] = old_weight.mean(dim=1)
        resnet.maxpool = nn.Identity()

        def modify_layer(layer):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

        modify_layer(resnet.layer2)
        modify_layer(resnet.layer3)
        modify_layer(resnet.layer4)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.high_level = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # 注意：ResNet50 高层输出通道为2048
        self.reduce_conv = nn.Conv2d(2048, self.output_channels, kernel_size=1)
        self.low_reduce = nn.Conv2d(256, self.output_channels, kernel_size=1)

    def forward(self, x, pool_size=None, return_multi=False):
        b, c, h, w = x.shape
        h = h // self.down_scale
        w = w // self.down_scale
        output_size = (h, w)

        low_feat = self.features(x)
        high_feat = self.high_level(low_feat)
        high_feat = self.reduce_conv(high_feat)
        low_feat = self.low_reduce(low_feat)

        if pool_size is not None:
            high_feat = F.adaptive_avg_pool2d(high_feat, pool_size)
            low_feat = F.adaptive_avg_pool2d(low_feat, pool_size)

        # 上采样到标签尺寸
        if high_feat.shape[2:] != output_size:
            high_feat = F.interpolate(high_feat, size=output_size, mode='bilinear')
            low_feat = F.interpolate(low_feat, size=output_size, mode='bilinear')

        if return_multi:
            return high_feat, low_feat
        return high_feat
