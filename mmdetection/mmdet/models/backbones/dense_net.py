# ! /usr/bin/env python
# -*- coding:utf-8 -*-

# @author: Edison Jia-hao-Chen
# time: 2019-6-10
# email: JiahaoChen@whu.edu.cn

# A implementation of DenseNet (Pytorch)


import logging
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from ..registry import BACKBONES
import torch.utils.checkpoint as cp


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# class DenseBlock(nn.Module):
#     pass


# DenseNet By Edison
@BACKBONES.register_module
class DenseNet_CJH(nn.Module):
    arch_settings = {
        121: (DenseBlock, (6, 12, 24, 16)),
        169: (DenseBlock, (6, 12, 32, 32)),
        201: (DenseBlock, (6, 12, 48, 32)),
        161: (DenseBlock, (6, 12, 36, 24))
    }

    def __init__(self,
                 idx_version=121, # (6, 12, 24, 16)
                 growth_rate=32,
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0):
        super(DenseNet_CJH, self).__init__()
        if idx_version not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(idx_version))

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features # 64

        self.block, stage_blocks = self.arch_settings[idx_version]
        self.stage_blocks = stage_blocks

        self.layers_block = []
        self.layers_trans = []

        for i, num_layers in enumerate(self.stage_blocks):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            # self.features.add_module('denseblock%d' % (i + 1), block)
            layer_block_name = 'denseblock%d' % (i + 1)
            self.add_module(layer_block_name, block)
            self.layers_block.append(layer_block_name)

            num_features = num_features + num_layers * growth_rate 

            if i != len(self.stage_blocks) - 1 :  # -1
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                # self.features.add_module('transition%d' % (i + 1), trans)
                layer_trans_name = 'transition%d' % (i + 1)
                self.add_module(layer_trans_name, trans)
                self.layers_trans.append(layer_trans_name)

                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        self.conv_final = nn.Conv2d(num_features, num_features * 2, kernel_size=1, stride=1, bias=False)
     
        

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        # out = self.classifier(out)
        # return out
        out = []
        for idx, layer_block_name in enumerate(self.layers_block):
            block_layer = getattr(self, layer_block_name)
            x = block_layer(x)

            if idx in range(3):
                trans_layer = getattr(self, self.layers_trans[idx])
                out.append(x)
                # print((x.size()))
                x = trans_layer(x)
                
                # print('='* 100)
                
                # print('trans\n\n')
                # print('='* 100)
        x = self.conv_final(x)
        out.append(x)
        # print((x.size()))
        # print('='* 100)

        # raise Exception('fuck out[{}]'.format(len(out)))
        # raise Exception('fuck cxy - {}'.format(x.size()))
        return tuple(out)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            # if self.dcn is not None:
            #
            #     # raise Exception('Dcn!!!!!!: {}'.format(dcn))
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck) and hasattr(
            #                 m, 'conv2_offset'):
            #             constant_init(m.conv2_offset, 0)

            # if self.zero_init_residual:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck):
            #             constant_init(m.norm3, 0)
            #         elif isinstance(m, BasicBlock):
            #             constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')
