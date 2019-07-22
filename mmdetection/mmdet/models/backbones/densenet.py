# ! /usr/bin/env python
# -*- coding:utf-8 -*-

# @author: Edison Jia-hao-Chen
# time: 2019-6-10
# email: JiahaoChen@whu.edu.cn

# A implementation of DenseNet (Pytorch)
# https://github.com/bamos/densenet.pytorch

import logging
import torch

import torch.nn as nn
# import torch.optim as optim

import torch.nn.functional as F

# from torch.autograd import Variable

# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# import torchvision.models as models

# import sys
import math
from ..registry import BACKBONES

import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint


from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


@BACKBONES.register_module
class DenseNet(nn.Module):
    def __init__(self, growthRate=32, depth=100, reduction=0.5, nClasses=1, bottleneck=True):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2 # 16

        nChannels = 2 * growthRate # 24
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate # 216
        nOutChannels = int(math.floor(nChannels * reduction)) # 108
        self.trans1 = Transition(nChannels, nOutChannels)

        print('nOutChannels={}'.format(nOutChannels))

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate # 300
        nOutChannels = int(math.floor(nChannels * reduction)) # 150
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate # 


        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        # raise Exception('fc channels [{}, {}]'.format(nChannels, nClasses))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def init_weights(self, pretrained=None):
        # to do
        print('|||========== no pretrained... =============|||')


    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
        

    def forward(self, x):
        outs = []
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        for i in range(4):
            outs.append(out)
        return tuple(outs)
