import cv2
import numpy as np
import torch
import torch.nn as nn

from deeplodocus.utils.generic_utils import get_specific_module
from deeplodocus.app.layers.empty import EmptyLayer


class YOLOv3(nn.Module):

    """
    Original article: https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """

    def __init__(
            self,
            backbone=None,
            num_classes=80,
            skip_layers=(36, 61),
            input_shape=(256, 256),
            anchors=(
                ((116, 90), (156, 198), (373, 326)),
                ((30, 61), (62, 45), (59, 119)),
                ((10, 13), (16, 30), (22, 23))
            ),
            normalized_anchors=False,
            predict=False,
            play=False
    ):
        super(YOLOv3, self).__init__()
        # Default backbone arguments
        if backbone is None:
            backbone = {
                "name": "Daknet53",
                "module": "deeplodocus.app.models.darknet",
                "kwargs": {
                    "num_channels": 3,
                    "include_top": False
                }
            }

        # Scale anchors by image dimensions if the anchors are normalized
        if normalized_anchors:
            anchors = [[(a[0] * input_shape[1], a[1] * input_shape[0]) for a in anchor] for anchor in anchors]

        self.predicting = predict
        self.playing = play
        # Get the number of anchor boxes
        num_anchors = len(anchors)

        # Get and initialise the backbone module
        backbone_module = get_specific_module(name=backbone["name"], module=backbone["module"], fatal=True)
        self.backbone = backbone_module(**backbone["kwargs"])

        # The keys to extract from the list
        self.skip_layers = skip_layers

        # CONVOLUTIONAL LAYERS/BLOCKS (ConvBlocks consist of 4 conv layers)
        self.conv_1_1 = ConvLayer(in_channels=1024, out_channels=512, kernel_size=1, negative_slope=0.1)
        self.conv_1_2 = ConvBlock(1024)
        self.conv_1_3 = ConvLayer(in_channels=512, out_channels=1024, kernel_size=3, padding=1, negative_slope=0.1)
        self.conv_1_4 = nn.Conv2d(in_channels=1024, out_channels=num_anchors * (num_classes + 5), kernel_size=1)

        self.conv_2_1 = ConvLayer(in_channels=512, out_channels=256, kernel_size=1, negative_slope=0.1)
        self.conv_2_2 = ConvLayer(in_channels=768, out_channels=256, kernel_size=1, negative_slope=0.1)
        self.conv_2_3 = ConvBlock(512)
        self.conv_2_4 = ConvLayer(in_channels=256, out_channels=512, kernel_size=3, padding=1, negative_slope=0.1)
        self.conv_2_5 = nn.Conv2d(in_channels=512, out_channels=num_anchors * (num_classes + 5), kernel_size=1)

        self.conv_3_1 = ConvLayer(in_channels=256, out_channels=128, kernel_size=1, negative_slope=0.1)
        self.conv_3_2 = ConvLayer(in_channels=384, out_channels=128, kernel_size=1, negative_slope=0.1)
        self.conv_3_3 = ConvBlock(256)
        self.conv_3_4 = ConvLayer(in_channels=128, out_channels=256, kernel_size=3, padding=1, negative_slope=0.1)
        self.conv_3_5 = nn.Conv2d(in_channels=256, out_channels=num_anchors * (num_classes + 5), kernel_size=1)

        # UPSAMPLE LAYER
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # YOLO LAYERS
        self.yolo_layer_1 = YoloLayer(
            num_classes=num_classes,
            image_shape=input_shape,
            anchors=anchors[0]
        )
        self.yolo_layer_2 = YoloLayer(
            num_classes=num_classes,
            image_shape=input_shape,
            anchors=anchors[1]
        )
        self.yolo_layer_3 = YoloLayer(
            num_classes=num_classes,
            image_shape=input_shape,
            anchors=anchors[2]
        )
        self.predict(predict)

    def forward(self, x):
        # BACKBONE
        if self.playing:
            img = x
        else:
            img = None
        x = self.backbone(x)                                            # b x 1024 x h/32 x w/32

        # DETECTION ON LARGEST SCALE
        x = self.conv_1_1(x)                                            # b x 512 x h/32 x w/32
        x = self.conv_1_2(x)                                            # b x 512 x h/32 x w/32
        output_1 = self.conv_1_3(x)                                     # b x 1024 x h/32 x w/32
        output_1 = self.conv_1_4(output_1)                              # b x 255 x h/32 x w/32
        output_1 = self.yolo_layer_1(output_1)                          # First YOLO layer

        # DETECTION ON MID SCALE
        x = self.conv_2_1(x)                                            #
        x = self.upsample(x)                                            # b x 256 x h/12 x w/16
        x = torch.cat((x, self.backbone.skip[self.skip_layers[1]]), 1)  # Concatenate x with second backbone skip layer
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)
        output_2 = self.conv_2_4(x)
        output_2 = self.conv_2_5(output_2)
        output_2 = self.yolo_layer_2(output_2)

        # DETECTION ON SMALLEST SCALE
        x = self.conv_3_1(x)
        x = self.upsample(x)
        x = torch.cat((x, self.backbone.skip[self.skip_layers[0]]), 1)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        output_3 = self.conv_3_4(x)
        output_3 = self.conv_3_5(output_3)
        output_3 = self.yolo_layer_3(output_3)

        # Return the concatenation of all three yolo layers
        if self.predicting:
            return torch.cat((output_1, output_2, output_3), 1)
        else:
            return output_1, output_2, output_3

    def predict(self, value=True):
        """
        :param value:
        :return:
        """
        if value:
            # Put model into evaluation mode
            self.eval()
        # Set predicting here and for all yolo layers
        self.predicting = value
        self.yolo_layer_1.predicting = value
        self.yolo_layer_2.predicting = value
        self.yolo_layer_3.predicting = value


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
        if batch_norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)
        else:
            self.norm_layer = EmptyLayer()
        if negative_slope == 0:
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        return x


class YoloLayer(nn.Module):

    def __init__(self, anchors, num_classes, image_shape, num_anchors=3, predict=False):
        super(YoloLayer, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.image_shape = image_shape      # (height, width)
        self.num_anchors = num_anchors
        self.anchors = anchors
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss(size_average=True)   # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)   # Objectiveness loss
        self.ce_loss = nn.CrossEntropyLoss()            # Class loss
        self.predicting = predict

    def forward(self, x):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        batch_size, _, h, w = x.shape

        # Stride should be 32, 16 or 8
        stride = self.image_shape[1] / w

        # Unpack predictions b x num_anchors x h x w x [num_classes + 5]
        prediction = x.view(
            batch_size,
            self.num_anchors,
            self.num_classes + 5,
            h,
            w
        ).permute(0, 1, 3, 4, 2).contiguous()

        # Scaled anchor width and height and cell offsets
        scaled_anchors = FloatTensor(self.anchors) / stride
        cx = torch.arange(w).repeat(h, 1).view([1, 1, h, w]).type(FloatTensor)
        cy = torch.arange(h).repeat(w, 1).t().view([1, 1, h, w]).type(FloatTensor)

        # Get all outputs
        bx = torch.sigmoid(prediction[..., 0]) + cx
        by = torch.sigmoid(prediction[..., 1]) + cy
        bw = scaled_anchors[:, 0].view(1, self.num_anchors, 1, 1) * torch.exp(prediction[..., 2])
        bh = scaled_anchors[:, 1].view(1, self.num_anchors, 1, 1) * torch.exp(prediction[..., 3])
        obj = torch.sigmoid(prediction[..., 4])
        cls = torch.sigmoid(prediction[..., 5:])

        # Recombine predictions after activations have been applied
        prediction = torch.cat((
            bx.view(*bx.shape, 1),
            by.view(*by.shape, 1),
            bw.view(*bw.shape, 1),
            bh.view(*bh.shape, 1),
            obj.view(*obj.shape, 1),
            cls
        ), 4)

        if self.predicting:
            # Scale up by stride
            prediction[..., 0:4] *= stride
            # Return flattened predictions
            return prediction.view(batch_size, -1, self.num_classes + 5)
        else:
            # If not in prediction mode, return predictions without offsets and with anchors
            return prediction, scaled_anchors
