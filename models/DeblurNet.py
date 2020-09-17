#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

# from models.submodules import *
import numpy 
from models.submodules import *
class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        # conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True)

        # encoder
        ks = 3
        # Conv1 stride = 1 whiteConv
        self.conv1_1 = conv(3, 32, kernel_size=ks, stride=1) 
        # Res2-4 grayResnet_block
        self.conv1_2 = resnet_block(32, kernel_size=ks)
        self.conv1_3 = resnet_block(32, kernel_size=ks)
        self.conv1_4 = resnet_block(32, kernel_size=ks)

        # Conv5 stride = 2 blueConv
        self.conv2_1 = conv(32, 64, kernel_size=ks, stride=2)
        # Res6-8 grayResnet_block
        self.conv2_2 = resnet_block(64, kernel_size=ks)
        self.conv2_3 = resnet_block(64, kernel_size=ks)
        self.conv2_4 = resnet_block(64, kernel_size=ks)

        # Conv9 stride = 2 blueConv
        self.conv3_1 = conv(64, 128, kernel_size=ks, stride=2)
        # Res10-12grayResnet_block
        self.conv3_2 = resnet_block(128, kernel_size=ks)
        self.conv3_3 = resnet_block(128, kernel_size=ks)
        self.conv3_4 = resnet_block(128, kernel_size=ks)

        # Res_d13-14 yellowResblock
        dilation = [1,2,3,4]
        self.convd_1 = resnet_block(128, kernel_size=ks, dilation = [2, 1])
        self.convd_2 = resnet_block(128, kernel_size=ks, dilation = [3, 1])
        #Context15 redContextModule
        self.convd_3 = ms_dilate_block(128, kernel_size=ks, dilation = dilation)

        # decoder
        # Conv16  single DeblurNet donot use Fuse vector,whiteConv
        self.upconv3_i = conv(128, 128, kernel_size=ks,stride=1)
        # Res17-19
        self.upconv3_3 = resnet_block(128, kernel_size=ks)
        self.upconv3_2 = resnet_block(128, kernel_size=ks)
        self.upconv3_1 = resnet_block(128, kernel_size=ks)

        # Upconv20 green Deconv
        self.upconv2_u = upconv(128, 64)
        self.attentionBlock1 = Attention_block(F_g=64, F_l=64, F_int=32)
        # Conv21 whiteConv
        self.upconv2_i = conv(128, 64, kernel_size=ks,stride=1)
        # Res22-24 grayResnet_block
        self.upconv2_3 = resnet_block(64, kernel_size=ks)
        self.upconv2_2 = resnet_block(64, kernel_size=ks)
        self.upconv2_1 = resnet_block(64, kernel_size=ks)

        # new added for Per-pixel kernels
        self.conv6 = conv(64, 64, kernel_size=ks, stride=1)
        #self.resize_image = only_upconv(64,64)
        self.resize_image = upconv(64,64)
        self.conv7 = conv(64, 32, kernel_size=ks, stride=1)
        #self.conv8 = only_conv(32, 9, kernel_size=ks, stride=1)
        self.conv8 = conv(32, 9, kernel_size=ks, stride=1)

        # Upconv25 green Deconv
        self.upconv1_u = upconv(64, 32)
        self.attentionBlock2 = Attention_block(F_g=32, F_l=32, F_int=16)
        # Conv26 whiteConv
        self.upconv1_i = conv(64, 32, kernel_size=ks,stride=1)
        # Res27-29 grayResnet_block
        self.upconv1_3 = resnet_block(32, kernel_size=ks)
        self.upconv1_2 = resnet_block(32, kernel_size=ks)
        self.upconv1_1 = resnet_block(32, kernel_size=ks)
        # Conv30 whiteConv
        #self.img_prd = conv(32, 3, kernel_size=ks, stride=1)

        # new added for Residual image and Blending weight map w
        #self.img_prd = only_conv(32, 3, kernel_size=ks, stride=1)
        #self.w_pred = only_conv(32, 1, kernel_size=ks, stride=1)
        self.img_prd = conv(32, 3, kernel_size=ks, stride=1)
        self.w_pred = conv(32, 1, kernel_size=ks, stride=1)     
        self.chanAttentionTopixel = se_layer(channel=ks*ks, ratio=2)


    #nohup python runner.py --phase 'train' > result_bs2_lrdecay0.1_400500550.out 2>&1
    def forward(self, x):
        conv1 = self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(x))))
        conv2 = self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(conv1))))
        conv3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(conv2))))
        convd = self.convd_3(self.convd_2(self.convd_1(conv3)))

        # decoder
        cat3 = self.upconv3_i(convd)
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3))))
        aconv2 = self.attentionBlock1(g=upconv2, x=conv2)
        cat2 = self.upconv2_i(cat_with_crop(aconv2, [aconv2, upconv2]))
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2))))
        aconv1 = self.attentionBlock2(g=upconv1, x=conv1)
        cat1 = self.upconv1_i(cat_with_crop(aconv1, [aconv1, upconv1]))

        #img_prd = self.img_prd(self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1)))) 

        # new added
        residual_img = self.img_prd(self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1))))  # img_prd equals Residual image in 《Per-Pixel Adaptive Kernels paper》
        w = self.w_pred(self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1))))
        conv6 = self.conv6(self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2))))
        resize_image = self.resize_image(conv6)
        conv7 = self.conv7(resize_image)
        conv8 = self.conv8(conv7)
        per_pixel_kernels = conv8  # [bs,k*k,h,w]
        img_patches = extract_image_patches(x, ksizes=3, strides=1, rates=1) # x=[bs,c,h,w] -> patches=[bs,c,ksizes*ksizes,h,w]
        per_pixel_kernels = self.chanAttentionTopixel(per_pixel_kernels)
        per_pixel_kernels = per_pixel_kernels.unsqueeze(1)  # [bs,ksizes*ksizes,h,w] -> [bs,1,ksizes*ksizes,h,w]
        tempImage = img_patches.mul(per_pixel_kernels)  # [bs,c,ksizes*ksizes,h,w]
        tempImage = torch.sum(tempImage, dim=2)
        img_pred = w * tempImage + (1 - w) * residual_img
        #return img_prd + x
        return img_pred 

# new added

# used to pad image for extract_image_patches
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def extract_image_patches(x, ksizes, strides, rates):
    bs, c, h, w = x.size()
    x = same_padding(x, (ksizes, ksizes), (strides, strides), (rates, rates))
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                            dilation=rates,
                            padding=0,
                            stride=strides)
    x_patches = unfold(x)  # [N, C*k*k, L] --> [N, C, k*k, h, w]
    x_patches = x_patches.view(bs,c,ksizes*ksizes,h,w)
    return x_patches