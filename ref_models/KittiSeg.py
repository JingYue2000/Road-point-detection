#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-12-13 17:21:16
# @Author  : jingyue zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict
from resnet import conv3x3, conv1x1, BasicBlock, Bottleneck

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class FusionModule(nn.Module):
    """docstring for FusionModule"""
    def __init__(self, in_planes, out_planes, deconv_kernel, stride=2):
        super(FusionModule, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.deconv_layer = self._make_deconv_layer(deconv_kernel)
        self.conv = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, deconv_kernel):
        layers = []
        kernel, padding, output_padding = self._get_deconv_cfg(deconv_kernel)

        layers.append(
            nn.ConvTranspose2d(
                in_channels=self.in_planes,
                out_channels=self.out_planes,
                kernel_size=kernel,
                stride=self.stride,
                padding=padding,
                output_padding=output_padding,
                bias=False))
        layers.append(nn.BatchNorm2d(self.out_planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, in_x, up_x):
        up_x = self.deconv_layer(up_x)
        in_x = self.conv(in_x)
        x = in_x + up_x
        return x
        

class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.num_classes = cfg.MODEL.NUM_JOINTS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.hall_layer1 = self._hall_layer1(2048, self.num_classes)
        self.hall_layer2 = self._hall_layer2(self.num_classes, self.num_classes)

        # used for deconv layers

        self.deconv_layer1 = FusionModule(self.num_classes, extra.NUM_DECONV_FILTERS[0], extra.NUM_DECONV_KERNELS[0])
        self.deconv_layer2 = FusionModule(extra.NUM_DECONV_FILTERS[0], extra.NUM_DECONV_FILTERS[1], extra.NUM_DECONV_KERNELS[1])
        self.deconv_layer3 = FusionModule(extra.NUM_DECONV_FILTERS[1], extra.NUM_DECONV_FILTERS[2], extra.NUM_DECONV_KERNELS[2])


        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _hall_layer1(self, in_planes, out_planes):
        return conv1x1(in_planes, out_planes)

    def _hall_layer2(self, in_planes, out_planes):
        return conv3x3(in_planes, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)        

        x5 = self.hall_layer1(x4)
        x6 = self.hall_layer2(x5)
        x7 = x5 + x6
        x7 = self.conv2(x7)        

        x8 = self.deconv_layer1(x3, x7)
        x9 = self.deconv_layer2(x2, x8)
        x10 = self.deconv_layer3(x1, x9)
        x11 = self.final_layer(x10)

        return x11

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train=False, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

if __name__ == '__main__':
    import numpy as np
    from easydict import EasyDict as edict
    from torch.autograd import Variable

    config = edict()
    # pose_resnet related params
    POSE_RESNET = edict()
    POSE_RESNET.NUM_LAYERS = 50
    POSE_RESNET.DECONV_WITH_BIAS = False
    POSE_RESNET.NUM_DECONV_LAYERS = 3
    POSE_RESNET.NUM_DECONV_FILTERS = [1024, 512, 256]
    POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
    POSE_RESNET.FINAL_CONV_KERNEL = 1
    POSE_RESNET.TARGET_TYPE = 'gaussian'
    POSE_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
    POSE_RESNET.SIGMA = 2

    MODEL_EXTRAS = {
        'pose_resnet': POSE_RESNET,
    }

    # common params for NETWORK
    config.MODEL = edict()
    config.MODEL.NAME = 'pose_resnet'
    config.MODEL.INIT_WEIGHTS = True
    config.MODEL.PRETRAINED = ''
    config.MODEL.NUM_JOINTS = 16
    config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
    config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

    config.MODEL.STYLE = 'pytorch'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_pose_net(config).to(device)
    x = Variable(torch.randn((1,3,512,512))).to(device)
    y = model(x)
