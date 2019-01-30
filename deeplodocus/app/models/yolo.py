import torch
import torch.nn as nn

from deeplodocus.app.blocks.convblock import ConvBlock2d
from deeplodocus.app.layers.yolo import YoloLayer
from deeplodocus.app.models.darknet import DarkNet53


class YOLOv3(nn.Module):

    def __init__(self, backbone, num_classes=80, skip_layers=("36", "61"), num_anchors=3, image_shape=(256, 256)):
        super(YOLOv3, self).__init__()
        print(backbone)
        self.backbone = backbone
        self.skip_layers = skip_layers

        # CONVOLUTIONAL LAYERS/BLOCKS
        self.conv_1_1 = ConvBlock2d(in_channels=1024, out_channels=512, kernel_size=1, negative_slope=0.1)
        self.conv_1_2 = YOLOConvBlock(1024)
        self.conv_1_3 = ConvBlock2d(in_channels=512, out_channels=1024, kernel_size=3, negative_slope=0.1)
        self.conv_1_4 = nn.Conv2d(in_channels=1024, out_channels=num_anchors * (num_classes + 5), kernel_size=1)

        self.conv_2_1 = ConvBlock2d(in_channels=512, out_channels=256, kernel_size=1, negative_slope=0.1)
        self.conv_2_2 = ConvBlock2d(in_channels=768, out_channels=256, kernel_size=1, negative_slope=0.1)
        self.conv_2_3 = YOLOConvBlock(512)
        self.conv_2_4 = ConvBlock2d(in_channels=256, out_channels=512, kernel_size=3, negative_slope=0.1)
        self.conv_2_5 = nn.Conv2d(in_channels=512, out_channels=num_anchors * (num_classes + 5), kernel_size=1)

        self.conv_3_1 = ConvBlock2d(in_channels=256, out_channels=128, kernel_size=1, negative_slope=0.1)
        self.conv_3_2 = ConvBlock2d(in_channels=384, out_channels=128, kernel_size=1, negative_slope=0.1)
        self.conv_3_3 = YOLOConvBlock(256)
        self.conv_3_4 = ConvBlock2d(in_channels=128, out_channels=256, kernel_size=3, negative_slope=0.1)
        self.conv_3_5 = nn.Conv2d(in_channels=256, out_channels=num_anchors * (num_classes + 5), kernel_size=1)

        # UPSAMPLE LAYER
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # YOLO LAYERS
        self.yolo_layer_1 = YoloLayer(
            num_classes=num_classes,
            image_shape=image_shape,
            num_anchors=num_anchors
        )
        self.yolo_layer_2 = YoloLayer(
            num_classes=num_classes,
            image_shape=image_shape,
            num_anchors=num_anchors
        )
        self.yolo_layer_3 = YoloLayer(
            num_classes=num_classes,
            image_shape=image_shape,
            num_anchors=num_anchors
        )

    def forward(self, x):
        x = self.backbone(x)                                            # b x 1024 x h/32 x w/32
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        output_1 = self.conv_1_3(x)
        output_1 = self.conv_1_4(output_1)                              # b x 255 x h/32 x w/32
        output_1 = self.yolo_layer_1(output_1)                          # First YOLO layer

        x = self.conv_2_1(x)                                            #
        x = self.upsample(x)                                            # b x 256 x h/12 x w/16
        x = torch.cat((x, self.backbone.skip[self.skip_layers[1]]), 1)  # Concatenate x with second backbone skip layer
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)
        output_2 = self.conv_2_4(x)
        output_2 = self.conv_2_5(output_2)
        output_2 = self.yolo_layer_2(output_2)

        x = self.conv_3_1(x)
        x = self.upsample(x)
        x = torch.cat((x, self.backbone.skip[self.skip_layers[0]]), 1)
        x = self.conv_3_2(x)
        x = self.conv_4_3(x)
        output_3 = self.conv_4_4(x)
        output_3 = self.conv_4_5(output_3)
        output_3 = self.yolo_layer_3(output_3)

        return torch.cat((output_1, output_2, output_3), 1)


class YOLOConvBlock(nn.Module):

    def __init__(self, filters=1024, negative_slope=0.1):
        super(YOLOConvBlock, self).__init__()
        self.conv_1 = ConvBlock2d(
            in_channels=int(filters / 2),
            out_channels=filters,
            kernel_size=3,
            negative_slope=negative_slope
        )
        self.conv_2 = ConvBlock2d(
            in_channels=filters,
            out_channels=int(filters / 2),
            kernel_size=1,
            negative_slope=negative_slope
        )
        self.conv_3 = ConvBlock2d(
            in_channels=int(filters / 2),
            out_channels=filters,
            kernel_size=3,
            negative_slope=negative_slope
        )
        self.conv_4 = ConvBlock2d(
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


if __name__ == "__main__":
    from deeplodocus.app.models.darknet import DarkNet53

    darknet53 = DarkNet53(include_top=False)
    yolo = YOLOv3(backbone=darknet53)
    print(yolo)
