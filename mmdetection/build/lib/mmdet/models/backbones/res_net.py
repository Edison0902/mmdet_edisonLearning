# ! /usr/bin/env python
# -*- coding:utf-8 -*-

# @author: Edison Jia-hao-Chen
# time: 2019-6-13
# email: JiahaoChen@whu.edu.cn

# A implementation of ResNet (Pytorch)

import logging
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from ..registry import BACKBONES
import torch.utils.checkpoint as cp


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.with_cp = with_cp

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            return out

        if self.with_cp and x.requires_grad:
            # out = _inner_forward(x)
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        

        out = _inner_forward(x)
        
        # raise Exception('x:{}   out:{}'.format(x.size(), out.size()))
        out = self.relu(out)
        return out


# ResNet By Edison
@BACKBONES.register_module
class ResNet_CJH(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 norm_eval=True,
                 frozen_stages=-1,
                 zero_init_residual=True):
        super(ResNet_CJH, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        self.depth = depth
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self.in_channels = 64

        # ResNet includes: layer_head + layers_res(0, 1, 2, 3)
        # the first layer: layer_head
        self.layer_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # four res layers
        self.layers_res = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            out_channels = 64 * 2**i

            res_layer = self._make_res_layer(
                self.block,
                self.in_channels,
                out_channels,
                num_blocks,
                stride=stride,
                dilation=dilation
            )
            self.in_channels = out_channels * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.layers_res.append(layer_name)

        # self._freeze_stages()

    # @property
    # def norm1(self):
    #     return getattr(self, self.norm1_name)

    # @property
    # def norm2(self):
    #     return getattr(self, self.norm2_name)

    # @property
    # def norm3(self):
    #     return getattr(self, self.norm3_name)


    # res layer: 0, 1, 2, 3
    def _make_res_layer(self,
                        block,
                        in_channels,
                        out_channels,
                        blocks,
                        stride=1,
                        dilation=1,
                        style='pytorch'):
        downsample = None

        # print('fuck make layers: stride={}, inchannels={}, outchannels={}'.format(stride, in_channels, out_channels))
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        # print('fuck downsample:\n{}'.format(downsample))
        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                )
        )

        in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    dilation=dilation)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_head(x)

        outs = []
        for i, layer_name in enumerate(self.layers_res):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in range(self.num_stages):
                outs.append(x)
        return tuple(outs)

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

            if self.zero_init_residual:
                # Todo
                pass
                # for m in self.modules():
                #     if isinstance(m, Bottleneck):
                #         constant_init(m.norm3, 0)
                #     elif isinstance(m, BasicBlock):
                #         constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         self.norm1.eval()
    #         for m in [self.conv1, self.norm1]:
    #             for param in m.parameters():
    #                 param.requires_grad = False
    #
    #     for i in range(1, self.frozen_stages + 1):
    #         m = getattr(self, 'layer{}'.format(i))
    #         m.eval()
    #         for param in m.parameters():
    #             param.requires_grad = False

    # def train(self, mode=True):
    #     super(ResNet, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()
