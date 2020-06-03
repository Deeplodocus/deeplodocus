# Backend imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *


class Darknet19(nn.Module):

    """
    Original article: https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """

    def __init__(self, num_channels=3, include_top=True, num_classes=1000, weights=None):
        super(Darknet19, self).__init__()

        # Set the number of input channels
        self.num_channels = int(num_channels)

        # For storing layer outputs (for skip connections in detection)
        self.skip = {}

        # Whether or not to include classifying layers
        self.include_top = include_top
        if include_top:
            self.num_classes = int(num_classes)
            # CLASSIFYING LAYER
            self.classifier = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1000),
                nn.ReLU(),
                nn.Linear(in_features=1000, out_features=self.num_classes),
                nn.Softmax(dim=1)
            )

        # DOWNSAMPLE LAYER
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # CONVBLOCKS
        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.conv1024 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
        )

        if weights is not None:
            self.load_weights(weights)

    def load_weights(self, weights):
        if weights is not None:
            Notification(DEEP_NOTIF_INFO, "Initializing network weights from file")
            try:
                self.load_state_dict(torch.load(weights)['model_state_dict'], strict=True)
                Notification(DEEP_NOTIF_SUCCESS, "Successfully loaded weights from %s" % weights)
            except RuntimeError as e:
                Notification(DEEP_NOTIF_WARNING, " :" .join(str(e).split(":")[1:]).strip())
                Notification(DEEP_NOTIF_WARNING, "Ignoring unexpected key(s)")
                self.load_state_dict(torch.load(weights)['model_state_dict'], strict=False)
                Notification(DEEP_NOTIF_SUCCESS, "Successfully loaded weights from %s" % weights)

    def forward(self, x):
        x = self.conv32(x)
        x = self.downsample(x)
        x = self.conv64(x)
        x = self.downsample(x)
        x = self.conv128(x)
        x = self.downsample(x)
        x = self.conv256(x)
        self.skip[8] = x
        x = self.downsample(x)
        x = self.conv512(x)
        self.skip[13] = x
        x = self.downsample(x)
        x = self.conv1024(x)
        if self.include_top:
            x = F.adaptive_avg_pool2d(x, (1, 1))    # Global Average Pooling
            x = x.view(x.size(0), -1)               # Flatten
            x = self.classifier(x)                  # Classify
        return x


class Darknet53(nn.Module):

    """
    Original article: https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """

    def __init__(self, num_channels=3, include_top=True, num_classes=1000, weights=None):
        super(Darknet53, self).__init__()

        # Set the number of input channels
        self.num_channels = int(num_channels)

        # Whether or not to include classifying layers
        self.include_top = include_top

        # For storing layer outputs (for skip connections in detection)
        self.skip = {}

        # INPUT LAYER
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # DOWNSAMPLE LAYERS
        self.downsample_64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.downsample_128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias=False, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.downsample_256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, bias=False, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.downsample_512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, bias=False, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.downsample_1024 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, bias=False, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # RESIDUAL BLOCKS
        # 64 filters
        self.res_block_64 = ResBlock(64)
        # 128 filters
        self.res_block_128_0 = ResBlock(128)
        self.res_block_128_1 = ResBlock(128)
        # 256 filters
        self.res_block_256_0 = ResBlock(256)
        self.res_block_256_1 = ResBlock(256)
        self.res_block_256_2 = ResBlock(256)
        self.res_block_256_3 = ResBlock(256)
        self.res_block_256_4 = ResBlock(256)
        self.res_block_256_5 = ResBlock(256)
        self.res_block_256_6 = ResBlock(256)
        self.res_block_256_7 = ResBlock(256)
        # 512 filters
        self.res_block_512_0 = ResBlock(512)
        self.res_block_512_1 = ResBlock(512)
        self.res_block_512_2 = ResBlock(512)
        self.res_block_512_3 = ResBlock(512)
        self.res_block_512_4 = ResBlock(512)
        self.res_block_512_5 = ResBlock(512)
        self.res_block_512_6 = ResBlock(512)
        self.res_block_512_7 = ResBlock(512)
        # 1024 filters
        self.res_block_1024_0 = ResBlock(1024)
        self.res_block_1024_1 = ResBlock(1024)
        self.res_block_1024_2 = ResBlock(1024)
        self.res_block_1024_3 = ResBlock(1024)

        if include_top:
            self.num_classes = int(num_classes)
            # Identification layer
            self.classifier = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1000),
                nn.ReLU(),
                nn.Linear(in_features=1000, out_features=self.num_classes),
                nn.Softmax(dim=1)
            )

        if weights is not None:
            self.load_weights(weights)

    def forward(self, x):
        assert not any(map(lambda i: i % 32, x.shape[2:])), \
            "Height and Width must be divisible by 32 : Received  %s" % str(x.shape[2:])
        x = self.input_layer(x)         # b x nc x h x w
        x = self.downsample_64(x)       # b x 64 x h/2 x w/2
        x = self.res_block_64(x)        # b x 64 x h/2 x w/2
        x = self.downsample_128(x)      # b x b x 128 x h/4 x w/4
        x = self.res_block_128_0(x)     # b x 128 x h/4 x w/4
        x = self.res_block_128_1(x)     # b x 128 x h/4 x w/4
        x = self.downsample_256(x)      # b x 256 x h/8 x w/8
        x = self.res_block_256_0(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_1(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_2(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_3(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_4(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_5(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_6(x)     # b x 256 x h/8 x w/8
        x = self.res_block_256_7(x)     # b x 256 x h/8 x w/8
        self.skip[36] = x               # Store activation for skip connection
        x = self.downsample_512(x)      # b x 512 x h/16 x w/16
        x = self.res_block_512_0(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_1(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_2(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_3(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_4(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_5(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_6(x)     # b x 512 x h/16 x w/16
        x = self.res_block_512_7(x)     # b x 512 x h/16 x w/16
        self.skip[61] = x               # Store activation for skip connection
        x = self.downsample_1024(x)     # b x 1024 x h/32 x w/32
        x = self.res_block_1024_0(x)    # b x 1024 x h/32 x w/32
        x = self.res_block_1024_1(x)    # b x 1024 x h/32 x w/32
        x = self.res_block_1024_2(x)    # b x 1024 x h/32 x w/32
        x = self.res_block_1024_3(x)    # b x 1024 x h/32 x w/32
        if self.include_top:
            x = F.adaptive_avg_pool2d(x, (1, 1))    # Global Average Pooling
            x = x.view(x.size(0), -1)               # Flatten
            x = self.classifier(x)                  # Classify
        return x
    
    def load_weights(self, weights):
        if weights is not None:
            Notification(DEEP_NOTIF_INFO, "Initializing network weights from file")
            try:
                self.load_state_dict(torch.load(weights)['model_state_dict'], strict=True)
                Notification(DEEP_NOTIF_SUCCESS, "Successfully loaded weights from %s" % weights)
            except RuntimeError as e:
                Notification(DEEP_NOTIF_WARNING, " :" .join(str(e).split(":")[1:]).strip())
                Notification(DEEP_NOTIF_WARNING, "Ignoring unexpected key(s)")
                self.load_state_dict(torch.load(weights)['model_state_dict'], strict=False)
                Notification(DEEP_NOTIF_SUCCESS, "Successfully loaded weights from %s" % weights)


class ResBlock(nn.Module):

    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=filters,
                      out_channels=int(filters / 2),
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(int(filters / 2)),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=int(filters / 2),
                      out_channels=filters,
                      kernel_size=3,
                      bias=False,
                      padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        res = x
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x += res
        return x
