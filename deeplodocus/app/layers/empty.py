import torch.nn as nn


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x
