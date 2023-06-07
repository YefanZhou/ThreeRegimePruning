import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import curves



__all__ = ['densenet_curve']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=False)


class BasicBlockCurve(nn.Module):
    def __init__(self, inplanes, fix_points, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlockCurve, self).__init__()
        planes = expansion * growthRate
        self.bn1 = curves.BatchNorm2d(inplanes, fix_points=fix_points)
        self.conv1 = curves.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x, coeffs_t):
        out = self.bn1(x, coeffs_t)
        out = self.relu(out)
        out = self.conv1(out, coeffs_t)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class TransitionCurve(nn.Module):
    def __init__(self, inplanes, outplanes, fix_points):
        super(TransitionCurve, self).__init__()
        self.bn1 = curves.BatchNorm2d(inplanes, fix_points=fix_points)
        self.conv1 = curves.Conv2d(inplanes, outplanes, kernel_size=1, bias=False, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, coeffs_t):
        out = self.bn1(x, coeffs_t)
        out = self.relu(out)
        out = self.conv1(out, coeffs_t)
        out = F.avg_pool2d(out, 2)
        return out



class DenseNetCurve(nn.Module):

    def __init__(self, num_classes, fix_points, depth=22, block=BasicBlockCurve, 
        dropRate=0, growthRate=12, compressionRate=1):
        super(DenseNetCurve, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlockCurve else (depth - 4) // 6
        n = int(n)

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = curves.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False, fix_points=fix_points)
        self.dense1 = self._make_denseblock(block, n, fix_points=fix_points)
        self.trans1 = self._make_transition(compressionRate, fix_points=fix_points)
        self.dense2 = self._make_denseblock(block, n, fix_points=fix_points)
        self.trans2 = self._make_transition(compressionRate, fix_points=fix_points)
        self.dense3 = self._make_denseblock(block, n, fix_points=fix_points)
        self.bn = curves.BatchNorm2d(self.inplanes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = curves.Linear(self.inplanes, num_classes, fix_points=fix_points)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.fill_(1)
                    getattr(m, 'bias_%d' % i).data.zero_()

    def _make_denseblock(self, block, blocks, fix_points):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate, fix_points=fix_points))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, fix_points):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return TransitionCurve(inplanes, outplanes, fix_points=fix_points)


    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        for block in self.dense1:
            x = block(x, coeffs_t)
        x = self.trans1(x, coeffs_t) 

        for block in self.dense2:
            x = block(x, coeffs_t)
        x = self.trans2(x, coeffs_t) 

        for block in self.dense3:
            x = block(x, coeffs_t)

        x = self.bn(x, coeffs_t)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)

        return x


class densenet_curve:
    curve = DenseNetCurve


if __name__ == '__main__':
    dense_curve = DenseNetCurve(num_classes=10,
                    depth=40,
                    growthRate=12,
                    compressionRate=1,
                    dropRate=0,
                    fix_points=[True, False, True])

    m_lst = curves.get_curve_prune_modules(dense_curve)
    print("m_lst", len(m_lst))
    for k, m in enumerate(m_lst):
        print(k, m.fix_points)