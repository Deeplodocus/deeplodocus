import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from deeplodocus.utils.generic_utils import get_specific_module
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "U":
            layers += [nn.Upsample(scale_factor=2, mode="nearest")]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    "11_decode": ["U", 512, 512, "U", 512, 512, "U", 256, 256, "U", 128, "U", 64],
    "13_decode": ["U", 512, 512, "U", 512, 512, "U", 256, 256, "U", 128, 128, "U", 64, 64],
    "16_decode": ["U", 512, 512, 512, "U", 512, 512, 512, "U", 256, 256, 256, "U", 128, 128, "U", 64, 64],
    "19_decode": ["U", 512, 512, 512, 512, "U", 512, 512, 512, 512, "U", 256, 256, 256, 256, "U", 128, 128, "U", 64, 64]
}


class VGG(nn.Module):

    def __init__(self, features, model_url=None, num_classes=1000, include_top=True):
        super(VGG, self).__init__()
        self.include_top = include_top
        self.features = features
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            self.avgpool = None
            self.classifier = None
        if model_url is None:
            self.initialize_weights()
        else:
            Notification(DEEP_NOTIF_INFO, "Loading model state dict from : %s" % model_url)
            state_dict = model_zoo.load_url(model_url)
            if not include_top:
                state_dict = {key: value for key, value in state_dict.items() if not key.startswith("classifier")}
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.features(x)
        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG11(VGG):

    def __init__(self, num_classes=1000, in_channels=3, batch_norm=True, pretrained=False, include_top=True):
        if pretrained:
            if batch_norm:
                model_url = model_urls['vgg11_bn']
            else:
                model_url = model_urls['vgg11']
        else:
            model_url = None
        features = make_layers(cfg["A"], batch_norm=batch_norm, in_channels=in_channels)
        super(VGG11, self).__init__(
            features=features,
            model_url=model_url,
            num_classes=num_classes,
            include_top=include_top
        )


class VGG13(VGG):

    def __init__(self, num_classes=1000, batch_norm=True, in_channels=3, pretrained=False, include_top=True):
        if pretrained:
            if batch_norm:
                model_url = model_urls['vgg13_bn']
            else:
                model_url = model_urls['vgg13']
        else:
            model_url = None
        features = make_layers(cfg["B"], batch_norm=batch_norm, in_channels=in_channels)
        super(VGG13, self).__init__(
            features=features,
            model_url=model_url,
            num_classes=num_classes,
            include_top=include_top
        )


class VGG16(VGG):

    def __init__(self, num_classes=1000, batch_norm=True, in_channels=3, pretrained=False, include_top=True):
        if pretrained:
            if batch_norm:
                model_url = model_urls['vgg16_bn']
            else:
                model_url = model_urls['vgg16']
        else:
            model_url = None
        features = make_layers(cfg["D"], batch_norm=batch_norm, in_channels=in_channels)
        super(VGG16, self).__init__(
            features=features,
            model_url=model_url,
            num_classes=num_classes,
            include_top=include_top
        )


class VGG19(VGG):

    def __init__(self, num_classes=1000, batch_norm=True, in_channels=3, pretrained=False, include_top=True):
        if pretrained:
            if batch_norm:
                model_url = model_urls['vgg19_bn']
            else:
                model_url = model_urls['vgg19']
        else:
            model_url = None
        features = make_layers(cfg["E"], batch_norm=batch_norm, in_channels=in_channels)
        super(VGG19, self).__init__(
            features=features,
            model_url=model_url,
            num_classes=num_classes,
            include_top=include_top
        )


class VGG11Decode(nn.Module):

    def __init__(self, batch_norm=True, num_classes=10):
        super(VGG11Decode, self).__init__()
        self.features = make_layers(cfg["11_decode"], batch_norm=batch_norm, in_channels=512)
        self.classifier = nn.Conv2d(in_channels=64, out_channels=num_classes, stride=1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGG13Decode(nn.Module):

    def __init__(self, batch_norm=True, num_classes=10):
        super(VGG13Decode, self).__init__()
        self.features = make_layers(cfg["13_decode"], batch_norm=batch_norm, in_channels=512)
        self.classifier = nn.Conv2d(in_channels=64, out_channels=num_classes, stride=1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGG16Decode(nn.Module):

    def __init__(self, batch_norm=True, num_classes=10):
        super(VGG16Decode, self).__init__()
        self.features = make_layers(cfg["16_decode"], batch_norm=batch_norm, in_channels=512)
        self.classifier = nn.Conv2d(in_channels=64, out_channels=num_classes, stride=1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGG19Decode(nn.Module):

    def __init__(self, batch_norm=True, num_classes=10):
        super(VGG19Decode, self).__init__()
        self.features = make_layers(cfg["19_decode"], batch_norm=batch_norm, in_channels=512)
        self.classifier = nn.Conv2d(in_channels=64, out_channels=num_classes, stride=1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(AutoEncoder, self).__init__()
        if encoder is None:
            encoder = {
                "name": "VGG16",
                "kwargs": {
                    "batch_norm": True,
                    "pretrained": True,
                    "in_channels": 3,
                    "include_top": False
                }
            }
        else:
            if "include_top" not in encoder["kwargs"]:
                Notification(DEEP_NOTIF_WARNING, "include_top not specified for the encoder, should be set to False")
        if decoder is None:
            decoder = {
                "name": "VGG16Decode",
                "kwargs": {
                    "batch_norm": True,
                    "num_classes": 10
                }
            }

        self.encoder = get_specific_module(
            name=encoder["name"],
            module="deeplodocus.app.models.vgg"
        )(**encoder["kwargs"])
        self.decoder = get_specific_module(
            name=decoder["name"],
            module="deeplodocus.app.models.vgg",
        )(**decoder["kwargs"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

