import torch
import torch.nn as nn
from config import cfg
from utils.network_utils import *


#
# Deblurring Loss
#
def mseLoss(output, target):
    mse_loss = nn.MSELoss(reduction ='elementwise_mean')
    MSE = mse_loss(output, target)
    return MSE

# L1 loss for new added
def l1Loss(output, target):
    l1_loss = nn.L1Loss(reduction ='elementwise_mean')
    L1 = l1_loss(output, target)
    return L1

def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)


def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss
