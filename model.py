#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jingyue zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class BasicConv(nn.Module):
    """docstring for BasicConv"""
    def __init__(self, in_planes, out_planes, kernel_size, padding='same', bn=True):
        super(BasicConv, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.bn = bn
        if padding == 'same':
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU())

    def forward(self, x):
        if self.bn:
            return self.conv1(x)
        else:
            return self.conv2(x)

class BasicRPNet(nn.Module):
    """docstring for BasicRPNet: Road Point Network"""
    def __init__(self, outNode=14, outPairNode=None, iters=3, cfg=None, fusion_mode='cat', w_part=True):
        super(BasicRPNet, self).__init__()
        self.outNode = outNode
        if outPairNode is None:
            self.outPairNode = outNode - 1
        else:
            self.outPairNode = outPairNode
        if cfg is None:
            self.cfg = [64, 64, 'M', 64, 128, 'M', 128, 128]
        else:
            self.cfg = cfg
        self.iters = iters
        self.fusion_mode = fusion_mode # 'cat' or 'add' or 'mul'
        self.w_part = w_part

        self.layer1 = self._make_basic_layer()
        self.layer2 = nn.Sequential(
            BasicConv(128, 256, 9),
            BasicConv(256, 512, 9),
            BasicConv(512, 256, 1, bn=False),
            BasicConv(256, 256, 1, bn=False))
        self.conv1x1_layer1 = BasicConv(384, 384, 1, bn=False)
        self.conv1x1_layer2 = BasicConv(128, 256, 1, bn=False)
        self.layer3 = nn.Sequential(
            BasicConv(384, 64, 7),
            BasicConv(64, 64, 13),
            BasicConv(64, 128, 13),
            BasicConv(128, 256, 1, bn=False))
        self.add_layer2 = nn.Sequential(
            BasicConv(128, 256, 9),
            BasicConv(256, 512, 9),
            BasicConv(512, 256, 1, bn=False),
            BasicConv(256, 128, 1, bn=False))
        self.add_layer3 = nn.Sequential(
            BasicConv(128, 64, 7),
            BasicConv(64, 64, 13),
            BasicConv(64, 128, 13),
            BasicConv(128, 128, 1, bn=False))
        self.heatmap1 = self._make_heatmap(256, self.outNode)
        self.heatmap2 = self._make_heatmap(256, self.outPairNode)
        self._init_weights()

    def _make_heatmap(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1),
            nn.Softmax2d())

    def _make_basic_layer(self):
        layers = []
        in_channels = 3
        for i, e in enumerate(self.cfg):
            if e == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(BasicConv(in_channels, e, kernel_size=3))
                in_channels = e
        return nn.Sequential(*layers)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name and 'convlayer' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'convlayer' in name and '0' in name:
                nn.init.xavier_normal_(param)
                #nn.init.kaiming_normal(w, mode='fan_out')
            
    def _cat_forward(self, x):
        # Feedforward
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # Recurrent
        for i in range(self.iters):
            x3 = torch.cat((x1, x2), 1)
            x2 = self.layer3(x3)
        return x2
        
    def _add_forward(self, x):
        # Feedforward
        x1 = self.layer1(x)
        x2 = self.add_layer2(x1)
        # Recurrent
        for i in range(self.iters):
            x3 = x1 + x2
            x2 = self.add_layer3(x3)
        x2 = self.conv1x1_layer2(x2)
        return x2

    def _mul_forward(self, x):
        # Feedforward
        x1 = self.layer1(x)
        x2 = self.add_layer2(x1)
        # Recurrent
        for i in range(self.iters):
            x3 =  torch.mul(x1, x2)
            x2 = self.add_layer3(x3)
        x2 = self.conv1x1_layer2(x2)
        return x2

    def forward(self,x):
        if self.fusion_mode == 'add':
            x = self._add_forward(x)
        elif self.fusion_mode == 'mul':
            x = self._mul_forward(x)
        else:
            x = self._cat_forward(x)
            
        heatmap = self.heatmap1(x)
        if self.w_part:
            partmap = self.heatmap2(x)
            heatmap = torch.cat((heatmap, partmap), 1)
        return heatmap

class keypointNet(nn.Module):
    def __init__(self, outNode=14, outPairNode=13):
        super(keypointNet, self).__init__()

        self.outNode = outNode
        self.outPairNode = outNode - 1
        self.feature1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                )
        self.feature2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=9, padding=4),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=9, padding=4),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                )
        self.feature3 = nn.Sequential(
                nn.Conv2d(384, 64, kernel_size=7, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=13, padding=6),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(64, 128, kernel_size=13, padding=6),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=1),
                nn.ReLU(),
                )
        self.heatmap1 = nn.Sequential(
                nn.Conv2d(256, self.outNode, kernel_size = 1),
                #nn.CELU(alpha=1.0, inplace=False)
                #nn.SELU(inplace=False),
                #Swish(),
                #nn.Tanh(),
                #nn.LeakyReLU(),
                nn.Softmax2d(),
                )
        self.heatmap2 = nn.Sequential(
                nn.Conv2d(256, self.outPairNode, kernel_size = 1),
                #nn.CELU(alpha=1.0, inplace=False)
                #nn.SELU(inplace=False),
                #Swish(),
                #nn.Tanh(),
                #nn.ReLU(), # converge at high loss
                #nn.Sigmoid(), # loss_coor increase
                #nn.LeakyReLU(), # converge faster than no activate function
                nn.Softmax2d(), # converge most rapidly by now
                )
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name and 'convlayer' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'convlayer' in name and '0' in name:
                nn.init.xavier_normal_(param)
                #nn.init.kaiming_normal(w, mode='fan_out')
            
    def forward(self, x):
        # Feedforward
        x1 = self.feature1(x)
        x2 = self.feature2(x1)
        # Recurrent-1
        x3 = torch.cat((x1, x2), 1)
        x4 = self.feature3(x3)
        # Recurrent-2
        x5 = torch.cat((x1, x4), 1)
        x6 = self.feature3(x5)
        # Recurrent-3
        x7 = torch.cat((x1, x6), 1)
        x8 = self.feature3(x7)
        
        pointmap1 = self.heatmap1(x2)
        pointmap2 = self.heatmap1(x4)
        pointmap3 = self.heatmap1(x6)
        pointmap4 = self.heatmap1(x8)
        
        partmap1 = self.heatmap2(x2)
        partmap2 = self.heatmap2(x4)
        partmap3 = self.heatmap2(x6)
        partmap4 = self.heatmap2(x8)
        heatmap1 = torch.cat((pointmap1, partmap1), 1)
        heatmap2 = torch.cat((pointmap2, partmap2), 1)
        heatmap3 = torch.cat((pointmap3, partmap3), 1)
        heatmap4 = torch.cat((pointmap4, partmap4), 1)
        
        #return pointmap1, pointmap2, pointmap3, pointmap4
        return heatmap1#, heatmap2, heatmap3, heatmap4
