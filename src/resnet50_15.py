import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, filters, downsample=False):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(4 * filters, filters, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.rectifier1 = nn.ReLU()

        # version 1.5 seems to be downsampling here?
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=2 if downsample else 1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.rectifier2 = nn.ReLU()

        self.conv3 = nn.Conv2d(filters, 4 * filters, 1, stride=1)
        self.bn3 = nn.BatchNorm2d(filters * 4)
        # residual connection here
        if downsample:
            self.linProj = nn.Conv2d(4 * filters, 4 * filters, 1, stride=2) # linear projection as explained in the paper
        self.rectifier3 = nn.ReLU()

    def forward(self, x):
        # Given the input [N, K, Y, X] where N is batch size, K is number of filters, Y and X for coordinates on convoluted image
        out = x # number of K must be 4 * filters

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.rectifier1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rectifier2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.linProj(x) if self.downsample else out + x
        out = self.rectifier3(out)

        return out

class Resnet50_15(nn.Module):
    def __init__(self, size=224):
        super().__init__()

        # raw image to conv2d
        # out = (size - 7 + 2*padding) / 2 + 1
        #   let out = size/2
        #   size/2 = (size - 7)/2 + 2*padding/2 + 1
        #   padding = size/2 - (size-7)/2 - 1 [round up]
        self.conv_pre = nn.Conv2d(3, 64, 7, stride=2, padding=3) # 112
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1) # 56

        self.conv = list()

        #conv2_x (no down-sampling here)
        self.conv.append(list())
        self.conv[0].append(nn.Conv2d(64, 256, 1)) # identity to map 64 filters to 256 filters
        for i in range(3):
            self.conv[0].append(BottleneckBlock(64))

        #conv3_x
        self.conv.append(list())
        self.conv[0].append(nn.Conv2d(256, 512, 1)) # identity to map 256 filters to 512 filters
        for i in range(4):
            self.conv[0].append(BottleneckBlock(128, i == 0))

        #conv4_x
        self.conv.append(list())
        self.conv[0].append(nn.Conv2d(512, 1024, 1)) # identity to map 512 filters to 1024 filters
        for i in range(6):
            self.conv[0].append(BottleneckBlock(256, i == 0))

        #conv5_x
        self.conv.append(list())
        self.conv[0].append(nn.Conv2d(1024, 2048, 1)) # identity to map 1024 filters to 2048 filters
        for i in range(3):
            self.conv[0].append(BottleneckBlock(512, i == 0))

        self.avg_pool = nn.AvgPool2d(7) # a single average pooling per filters

    def forward(self, x):
        out = x

        out = self.conv_pre(out)
        out = self.max_pool(out)

        for group in self.conv:
            for block in group:
                out = block(out)    

        out = self.avg_pool(out)
        out = torch.squeeze(out) # [N, filters, 1, 1] -> [N, filters]

        return out
