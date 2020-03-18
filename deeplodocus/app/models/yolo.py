import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *
from deeplodocus.utils.generic_utils import get_specific_module


skip_dict = {
    "Darknet53": (36, 61),
    "Darknet19": (8, 13)
}


class YOLO(nn.Module):

    """
    Original article: https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """

    def __init__(
            self,
            backbone=None,
            num_classes=10,
            skip_layers=None,
            anchors=(
                ((10, 13), (16, 30), (22, 23)),
                ((30, 61), (62, 45), (59, 119)),
                ((116, 90), (156, 198), (373, 326))
            )
    ):
        super(YOLO, self).__init__()
        self.num_classes = num_classes

        # If no backbone specified, set default backbone arguments
        if backbone is None:
            backbone = {
                "name": "Daknet53",
                "module": "deeplodocus.app.models.darknet",
                "kwargs": {
                    "num_channels": 3,
                    "include_top": False
                }
            }
            skip_layers = (36, 61)

        # If skip layers not specified, select the appropriate indices in accordance with skip_dict
        with contextlib.suppress(KeyError):
            for key, skip in skip_dict.items():
                if backbone["name"].startswith(key) and skip_layers is None:
                    skip_layers = skip
                    Notification(DEEP_NOTIF_INFO, "%s detected : skip layers set to %s" % (key, str(skip)))

        # Get and initialise the backbone module
        backbone_module = get_specific_module(name=backbone["name"], module=backbone["module"], fatal=True)
        self.backbone = backbone_module(**backbone["kwargs"])

        # Indices of the layers to connect to in the backbone
        self.skip_layers = skip_layers

        # CONVOLUTIONAL LAYERS/BLOCKS (ConvBlocks contain 4 conv layers)
        self.conv_1_1 = ConvLayer(in_channels=1024, out_channels=512, kernel_size=1, negative_slope=0.1)
        self.conv_1_2 = ConvBlock(1024)
        self.conv_1_3 = ConvLayer(in_channels=512, out_channels=1024, kernel_size=3, padding=1, negative_slope=0.1)
        self.conv_1_4 = nn.Conv2d(in_channels=1024, out_channels=len(anchors) * (num_classes + 5), kernel_size=1)

        self.conv_2_1 = ConvLayer(in_channels=512, out_channels=256, kernel_size=1, negative_slope=0.1)
        self.conv_2_2 = ConvLayer(in_channels=768, out_channels=256, kernel_size=1, negative_slope=0.1)
        self.conv_2_3 = ConvBlock(512)
        self.conv_2_4 = ConvLayer(in_channels=256, out_channels=512, kernel_size=3, padding=1, negative_slope=0.1)
        self.conv_2_5 = nn.Conv2d(in_channels=512, out_channels=len(anchors) * (num_classes + 5), kernel_size=1)

        self.conv_3_1 = ConvLayer(in_channels=256, out_channels=128, kernel_size=1, negative_slope=0.1)
        self.conv_3_2 = ConvLayer(in_channels=384, out_channels=128, kernel_size=1, negative_slope=0.1)
        self.conv_3_3 = ConvBlock(256)
        self.conv_3_4 = ConvLayer(in_channels=128, out_channels=256, kernel_size=3, padding=1, negative_slope=0.1)
        self.conv_3_5 = nn.Conv2d(in_channels=256, out_channels=len(anchors) * (num_classes + 5), kernel_size=1)

        # UPSAMPLE LAYER
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # YOLO LAYERS
        # Layer 1 - detection on largest scale
        self.yolo_layer_1 = YoloLayer(anchors=anchors[2])
        # Layer 2 - detection on mid scale
        self.yolo_layer_2 = YoloLayer(anchors=anchors[1])
        # Layer 3 - detection on smallest scale
        self.yolo_layer_3 = YoloLayer(anchors=anchors[0])

        # Classify after backbone
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        if not (x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0):
            Notification(
                DEEP_NOTIF_FATAL,
                "in YOLO.forward(x) : input height and width must be divisible by 32 : got %s" % str(x.shape)
            )
        image_shape = x.shape[2:4]

        # BACKBONE - initial feature detection
        x = self.backbone(x)                                            # b x 1024 x h/32 x w/32

        # DETECTION ON LARGEST SCALE
        x = self.conv_1_1(x)                                            # b x 512  x h/32 x w/32
        x = self.conv_1_2(x)                                            # b x 512  x h/32 x w/32
        output_1 = self.conv_1_3(x)                                     # b x 1024 x h/32 x w/32
        output_1 = self.conv_1_4(output_1)                              # b x 255  x h/32 x w/32
        output_1 = self.yolo_layer_1(output_1, image_shape)             # First YOLO layer (large scale)

        # DETECTION ON MID SCALE
        x = self.conv_2_1(x)                                            # b x 256 x h/32 x w/32
        x = self.upsample(x)                                            # b x 256 x h/16 x w/16
        skip = self.backbone.skip[self.skip_layers[1]]                  # Get skip layer from backbone
        x = torch.cat((x, skip), 1)                                     # Concatenate x with second backbone skip layer
        x = self.conv_2_2(x)                                            # b x 256 x h/16 x w/16
        x = self.conv_2_3(x)                                            # b x 256 x h/16 x w/16
        output_2 = self.conv_2_4(x)                                     # b x 512 x h/16 x w/16
        output_2 = self.conv_2_5(output_2)                              # b x a * (num_classes + num_types + 5) x h/16 x w/16
        output_2 = self.yolo_layer_2(output_2, image_shape)             # Second YOLO layer (mid scale)

        # DETECTION ON SMALLEST SCALE
        x = self.conv_3_1(x)                                            # b x 128 x h/16 x w/16
        x = self.upsample(x)                                            # b x 128 x h/8  x w/8
        skip = self.backbone.skip[self.skip_layers[0]]
        x = torch.cat((x, skip), 1)  # Concatenate x with first backbone skip layer
        x = self.conv_3_2(x)                                            # b x 128 x h/8  x w/8
        x = self.conv_3_3(x)                                            # b x 128 x h/8  x w/8
        output_3 = self.conv_3_4(x)                                     # b x 128 x h/8  x w/8
        output_3 = self.conv_3_5(output_3)                              # b x a * (num_classes + num_types + 5) x h/8 x w/8
        output_3 = self.yolo_layer_3(output_3, image_shape)             # Third YOLO layer (small scale)

        return {
            "detections": [output_1, output_2, output_3],
            "scaled_anchors": torch.stack(
                (
                    self.yolo_layer_1.scaled_anchors,
                    self.yolo_layer_2.scaled_anchors,
                    self.yolo_layer_3.scaled_anchors
                ), 0
            ),
            "strides": (
                self.yolo_layer_1.stride,
                self.yolo_layer_2.stride,
                self.yolo_layer_3.stride
            )
        }


class ConvBlock(nn.Module):

    def __init__(self, filters=1024, negative_slope=0.1):
        super(ConvBlock, self).__init__()
        self.conv_1 = ConvLayer(
            in_channels=int(filters / 2),
            out_channels=filters,
            kernel_size=3,
            padding=1,
            negative_slope=negative_slope
        )
        self.conv_2 = ConvLayer(
            in_channels=filters,
            out_channels=int(filters / 2),
            kernel_size=1,
            negative_slope=negative_slope
        )
        self.conv_3 = ConvLayer(
            in_channels=int(filters / 2),
            out_channels=filters,
            kernel_size=3,
            padding=1,
            negative_slope=negative_slope
        )
        self.conv_4 = ConvLayer(
            in_channels=filters,
            out_channels=int(filters / 2),
            kernel_size=1,
            negative_slope=negative_slope
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class ConvLayer(nn.Module):
    """
    Author: Samuel Westlake

    Acts to package:
        > a convolutional layer
        > batch normalisation
        > ReLU activation
    into a single layer (in that order).
    """
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3,
            bias=False,
            stride=1,
            batch_norm=True,
            padding=0,
            negative_slope=0.0
    ):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            stride=stride,
            padding=padding
        )
        self.norm_layer = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation_layer = nn.ReLU() if negative_slope == 0 else nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.activation_layer(x)
        return x


class YoloLayer(nn.Module):

    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.scaled_anchors = None
        self.stride = None
        self.shape = None
        self.x = None
        self.y = None

    def forward(self, x, image_shape):
        (b, _, h, w), a = x.shape, len(self.anchors)
        if self.shape != (h, w):
            self.shape = (h, w)
            self.stride = image_shape[1] / w
            self.x = torch.arange(w, device=x.device).repeat(h, 1).view([1, 1, h, w]).float()
            self.y = torch.arange(h, device=x.device).repeat(w, 1).t().view([1, 1, h, w]).float()
            self.scaled_anchors = torch.tensor(self.anchors, device=x.device, dtype=torch.float32) / self.stride

        # Unpack predictions b x num_anchors x h x w x [num_classes + 5]
        x = x.view(b, a, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()

        # Apply transforms to bx, by, bw, bh
        bx = torch.sigmoid(x[..., 0]) + self.x
        by = torch.sigmoid(x[..., 1]) + self.y
        bw = self.scaled_anchors[:, 0].view(1, a, 1, 1) * torch.exp(x[..., 2])
        bh = self.scaled_anchors[:, 1].view(1, a, 1, 1) * torch.exp(x[..., 3])

        return torch.cat(
            (bx[..., None], by[..., None], bw[..., None], bh[..., None], x[..., 4:]),
            dim=4
        )
