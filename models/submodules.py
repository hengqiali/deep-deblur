#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import torch.nn as nn
import torch
import numpy as np
from config import cfg


def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        #nn.ReLU(inplace=True)
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True)
    )

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),  # ...
        #nn.ReLU(inplace=True)
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class se_layer(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # [b, c, 1, 1]
        self.fc1 = nn.Linear(channel, int(channel/ratio))
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True)
        self.fc2 = nn.Linear(int(channel/ratio), channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()  # 2 512 64 64
        y = self.avg_pool(x)  # [b, c, 1, 1] 2 512 1 1
        y = y.view(b, -1) # [b, c*1*1]  2 512
        y = self.relu(self.fc1(y)) #[b,c/16]  2 32
        y = self.sigmoid(self.fc2(y)) # [b,c] 2 512
        y = y.view(b, c, 1, 1)  # [b,c,1,1] 2 512 1 1
        return x * y.expand_as(x)

def ms_dilate_block(in_channels, kernel_size=3, dilation=[1,1,1,1], bias=True):
    return MSDilateBlock(in_channels, kernel_size, dilation, bias)

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[3], bias=bias)
        self.se = se_layer(in_channels*4)   
        self.convi =  nn.Conv2d(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        cat = self.se(cat)
        out = self.convi(cat) + x

        # out = x + torch.max(torch.cat([conv1.unsqueeze(0),conv2.unsqueeze(0),conv3.unsqueeze(0),conv4.unsqueeze(0)],0),0)[0] # max version
        # out = x + torch.sum(torch.cat([conv1.unsqueeze(0),conv2.unsqueeze(0),conv3.unsqueeze(0),conv4.unsqueeze(0)],0),0) / 2 # avg version
        return out

                                 
def cat_with_crop(target, input): #(1) target = conv2, input = [conv2[1, 64, 128, 128], upconv2[1, 64, 128, 128]]
    output = []                   #(2) target = conv1, input = [conv1[1, 32, 256, 256], upconv1[1, 32, 256, 256]]
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output
