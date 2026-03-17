import torch
from torch import nn
from air_track.utils.registry import HEADS


@HEADS.register('Head_2_Output')
class Head_2_Output(nn.Module):
    def __init__(self, in_ch, cls_num, mode='train'):
        super().__init__()
        self.mode = mode
        self.cls_num = cls_num

        self.fc_mask = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.fc_size = nn.Conv2d(in_ch, 2, kernel_size=1)

    def forward(self, x):
        size = self.fc_size(x)
        mask = self.fc_mask(x)

        if self.mode == 'predict':
            mask = torch.sigmoid(mask)

        res = [
            mask,
            size,
        ]

        return res


@HEADS.register('Head_3_Output')
class Head_3_Output(nn.Module):
    def __init__(self, in_ch, cls_num, mode='train'):
        super().__init__()
        self.mode = mode
        self.cls_num = cls_num

        self.fc_mask = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.fc_size = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.fc_cls = nn.Conv2d(in_ch, self.cls_num, kernel_size=1)

    def forward(self, x):
        size = self.fc_size(x)
        mask = self.fc_mask(x)
        cls = self.fc_cls(x)

        if self.mode == 'predict':
            mask = torch.sigmoid(mask)

        res = [
            mask,
            size,
            cls,
        ]

        return res


@HEADS.register('Head_5_Output')
class Head_5_Output(nn.Module):
    def __init__(self, in_ch, cls_num, mode='train'):
        super().__init__()
        self.mode = mode
        self.cls_num = cls_num

        self.fc_mask = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.fc_size = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(in_ch, 2, kernel_size=1)

    def forward(self, x):
        size = self.fc_size(x)
        mask = self.fc_mask(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        if self.mode == 'predict':
            mask = torch.sigmoid(mask)

        res = [
            mask,
            size,
            offset,
            distance,
            tracking,
        ]

        return res


@HEADS.register('Head_6_Output')
class Head_6_Output(nn.Module):
    def __init__(self, in_ch, cls_num, mode='train'):
        super().__init__()
        self.mode = mode
        self.cls_num = cls_num

        self.fc_mask = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.fc_size = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.fc_cls = nn.Conv2d(in_ch, self.cls_num, kernel_size=1)
        self.fc_offset = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(in_ch, 2, kernel_size=1)

    def forward(self, x):
        size = self.fc_size(x)
        mask = self.fc_mask(x)
        cls = self.fc_cls(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        if self.mode == 'predict':
            mask = torch.sigmoid(mask)

        res = [
            mask,
            size,
            offset,
            distance,
            tracking,
            cls,
        ]

        return res
