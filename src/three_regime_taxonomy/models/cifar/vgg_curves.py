'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
from . import curves
import torch.nn as nn
import math


__all__ = [
    'vgg11_curve', 'vgg11_bn_curve', 
    'vgg13_curve', 'vgg13_bn_curve', 
    'vgg16_curve', 'vgg16_bn_curve',
    'vgg19_curve', 'vgg19_bn_curve'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, 
                        fix_points=fix_points, stride=stride,
                         padding=1, bias=False)


class VGGCurve(nn.Module):

    def __init__(self, num_classes, fix_points, config='N', batch_norm=True):
        super(VGGCurve, self).__init__()
        print("fix_points", fix_points, config, batch_norm)
        features = make_layers(cfg[config], batch_norm=batch_norm, fix_points=fix_points)
        self.features = features
        #self.classifier = nn.Linear(512, num_classes)
        self.classifier = curves.Linear(512, num_classes, fix_points)   #nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x, coeffs_t):
        for block in self.features:
            if isinstance(block, nn.ReLU) or isinstance(block, nn.MaxPool2d):
                x = block(x)
            else:
                x = block(x, coeffs_t)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x, coeffs_t)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, fix_points=[]):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = curves.Conv2d(in_channels, v, 
                                    kernel_size=3, 
                                    fix_points=fix_points, 
                                    stride=1,
                                    padding=1) #bias=False
            if batch_norm:
                layers += [conv2d, curves.BatchNorm2d(v, fix_points=fix_points), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    # 'E': [64, 128, 'M', 128, 256, 'M', 64, 128, 256, 512, 1024, 'M', 64, 128, 256, 512, 1024, 2048,'M',256, 512, 1024, 512,'M']
}


class vgg11_curve:
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    curve = VGGCurve


class vgg11_bn_curve:
    """VGG 11-layer model (configuration "A") with batch normalization"""
    curve = VGGCurve


class vgg13_curve:
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    curve = VGGCurve


class vgg13_bn_curve:
    """VGG 13-layer model (configuration "B") with batch normalization"""
    # model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    # return model
    curve = VGGCurve


class vgg16_curve:
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    curve = VGGCurve


class vgg16_bn_curve:
    """VGG 16-layer model (configuration "D") with batch normalization"""

    curve = VGGCurve


class vgg19_curve:
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    curve = VGGCurve


class vgg19_bn_curve:
    """VGG 19-layer model (configuration 'E') with batch normalization"""

    curve = VGGCurve
    




