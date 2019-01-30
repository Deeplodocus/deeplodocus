import torch.nn as nn

from deeplodocus.app.layers.empty import EmptyLayer


class ConvBlock2d(nn.Module):

    def __init__(
            self, in_channels, out_channels,
            kernel_size=3,
            bias=False,
            stride=1,
            batch_norm=True,
            negative_slope=0
    ):
        super(ConvBlock2d, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            stride=stride
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

