import torch
import torch.nn as nn

from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *
from deeplodocus.utils.generic_utils import get_specific_module

# This enables YOLO to decide which skip connections to expect for each backbone
skip_dict = {
    "Darknet53": (36, 61),
    "Darknet19": (8, 13)
}

# Message strings
MSG_AUTO_SKIP = "%s detected : skip layers set to %s"
MSG_NO_SKIP = "in YOLO.__init__() : Skip layers not specified : could not auto assign skip layers for %s"
MSG_WRONG_DIMS = "in YOLO.forward() : input has incorrect number of dimensions : expected 4 : got %i"
MSG_WRONG_SHAPE = "in YOLO.forward() : input height and width must be divisible by 32 : got %s"


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
        self.anchors = anchors

        # If no backbone specified, set default backbone arguments
        if backbone is None:
            backbone = {
                "name": "Darknet53",
                "module": "deeplodocus.app.models.darknet",
                "kwargs": {
                    "num_channels": 3,
                    "include_top": False
                }
            }
            skip_layers = (36, 61)

        # If skip layers not specified, select the appropriate indices in accordance with skip_dict
        if skip_layers is None:
            try:
                self.skip_layers = skip_dict[backbone["name"]]
                Notification(
                    DEEP_NOTIF_INFO,
                    MSG_AUTO_SKIP % (backbone["name"], str(self.skip_layers))
                )
            except KeyError:
                Notification(
                    DEEP_NOTIF_FATAL,
                    MSG_NO_SKIP % backbone["name"]
                )
        else:
            self.skip_layers = skip_layers

        # Get and initialise the backbone module
        backbone_module = get_specific_module(name=backbone["name"], module=backbone["module"], fatal=True)
        self.backbone = backbone_module(**backbone["kwargs"])

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

        # Classify after backbone
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        im_shape = torch.tensor(x.shape[2:4], dtype=torch.float32)

        # Check input dims
        if x.ndim != 4:
            Notification(
                DEEP_NOTIF_FATAL,
                MSG_WRONG_DIMS % x.ndim
            )
        # Check input height and width are divisible by 32
        if not (x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0):
            Notification(
                DEEP_NOTIF_FATAL,
                MSG_WRONG_SHAPE % str(x.shape)
            )

        # BACKBONE - initial feature detection
        x = self.backbone(x)  # b x 1024 x h/32 x w/32

        # DETECTION ON LARGEST SCALE
        x = self.conv_1_1(x)  # b x 512  x h/32 x w/32
        x = self.conv_1_2(x)  # b x 512  x h/32 x w/32
        output_1 = self.conv_1_3(x)  # b x 1024 x h/32 x w/32
        output_1 = self.conv_1_4(output_1)  # b x 3 * (num cls + 5) x h/32 x w/32
        output_1 = self.unpack(output_1, s=2)  # Unpack predictions across anchors

        # DETECTION ON MID SCALE
        x = self.conv_2_1(x)  # b x 256 x h/32 x w/32
        x = self.upsample(x)  # b x 256 x h/16 x w/16
        skip = self.backbone.skip[self.skip_layers[1]]  # Get skip layer from backbone
        x = torch.cat((x, skip), 1)  # Concatenate x with second backbone skip layer
        x = self.conv_2_2(x)  # b x 256 x h/16 x w/16
        x = self.conv_2_3(x)  # b x 256 x h/16 x w/16
        output_2 = self.conv_2_4(x)  # b x 512  x h/16 x w/16
        output_2 = self.conv_2_5(output_2)  # b x 3 * (num cls + 5) x h/16 x w/16
        output_2 = self.unpack(output_2, s=1)  # Unpack predictions across anchors

        # DETECTION ON SMALLEST SCALE
        x = self.conv_3_1(x)  # b x 128 x h/16 x w/16
        x = self.upsample(x)  # b x 128 x h/8  x w/8
        skip = self.backbone.skip[self.skip_layers[0]]
        x = torch.cat((x, skip), 1)  # Concatenate x with first backbone skip layer
        x = self.conv_3_2(x)  # b x 128 x h/8  x w/8
        x = self.conv_3_3(x)  # b x 128 x h/8  x w/8
        output_3 = self.conv_3_4(x)  # b x 128 x h/8  x w/8
        output_3 = self.conv_3_5(output_3)  # b x 3 * (num cls + 5) x h/8 x w/8
        output_3 = self.unpack(output_3, s=0)  # Unpack predictions across anchors
        return Namespace(
            {
                "inference": [output_3, output_2, output_1],  # Scale: [small, medium scale, large]
                "anchors": torch.tensor(self.anchors, dtype=torch.float32, device=self.device),
                "strides": torch.tensor(
                    (
                        im_shape[0] / output_3.shape[2],
                        im_shape[0] / output_2.shape[2],
                        im_shape[0] / output_1.shape[2],
                    ),
                    dtype=torch.float32,
                    device=self.device
                )
            }
        )

    def unpack(self, x, s):
        a = len(self.anchors[s])
        b, _, h, w = x.shape
        return x.view(b, a, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()


class ConvBlock(nn.Module):

    def __init__(self, filters=1024, negative_slope=0.1):
        """
        Author: SW
        Acts to contain the regular appearing pattern of 5 layers
        (3x3) -> (1x1) -> (3x3) -> (1x1) -> (3x3)
        """
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
    in that order
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


#class YoloLayer(nn.Module):
#
#    def __init__(self, anchors):
#        super(YoloLayer, self).__init__()
#        self.anchors = torch.tensor(anchors, dtype=torch.float32)
#        self.stride = None
#        self.scaled_anchors = None
#
#    def forward(self, x, image_shape):
#        (b, _, h, w) = x.shape
#        a = self.anchors.shape[1]
#        self.anchors = self.anchors.to(x.device)
#
         # Calculate stride and scale anchors w.r.t. cell size
#         self.stride = image_shape[1] / w
#         self.scaled_anchors = self.anchors / self.stride
#
        # Make grids of cell locations
#        cx = torch.arange(w, device=x.device).repeat(h, 1).view([1, 1, h, w]).float()
#        cy = torch.arange(h, device=x.device).repeat(w, 1).t().view([1, 1, h, w]).float()

        # Unpack predictions b x num_anchors x h x w x [num_classes + 5]
#        x = x.view(b, a, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()

        # Apply transforms to bx, by, bw, bh
#        bx = torch.sigmoid(x[..., 0]) + cx
#        by = torch.sigmoid(x[..., 1]) + cy
#        bw = self.scaled_anchors[:, 0].view(1, a, 1, 1) * torch.exp(x[..., 2])
#        bh = self.scaled_anchors[:, 1].view(1, a, 1, 1) * torch.exp(x[..., 3])

#        return torch.cat(
#            (bx[..., None], by[..., None], bw[..., None], bh[..., None], x[..., 4:]),
#            dim=4
#        )
