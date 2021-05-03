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