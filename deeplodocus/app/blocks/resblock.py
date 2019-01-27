# Backend import
import torch.nn as nn

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
