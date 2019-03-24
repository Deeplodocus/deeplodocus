import torch
import torch.nn as nn


class SegmentationNet(nn.Module):

    def __init__(self, num_channels=3, num_classes=10):
        super(SegmentationNet, self).__init__()
        self.encode_block_1 = DownsampleBlock(num_channels, 32)
        self.encode_block_2 = DownsampleBlock(32, 64)
        self.encode_block_3 = DownsampleBlock(64,  128)
        self.encode_block_4 = DownsampleBlock(128, 256)
        self.decode_block_1 = UpsampleBlock(256, 128)

    def forward(self, x):
        x = self.encode_block_1(x)
        x = self.encode_block_2(x)
        x = self.encode_block_3(x)
        x = self.encode_block_4(x)
        return x


class DownsampleBlock(nn.Module):

    def __init__(self, in_channels=32, out_channels=64):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        # DOWNSAMPLE WITH 1x1 CONVOLUTION WITH A STRIDE OF 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x




