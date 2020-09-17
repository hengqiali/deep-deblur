#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import sys
import torch.backends.cudnn
import torch.utils.data

import utils.data_transforms
import utils.network_utils
import utils.data_loaders
import models
from models.DeblurNet import DeblurNet

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from core.train_deblur import train_deblurnet
from core.test_deblur import test_deblurnet
from losses.multiscaleloss import *

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

#from .dataset_GoPro import PairedDataset
#from .dataset_GP import GoProDataset

IMAGE_SIZE = 256



def bulid_net(cfg):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ### HQ add GroPro Dataset Dataloader V2
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
        utils.data_transforms.RandomVerticalFlip(),
        utils.data_transforms.RandomColorChannel(),
        utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        utils.data_transforms.ToTensor(),
    ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    # dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()

    train_data_loader = torch.utils.data.DataLoader(
        dataset = utils.data_loaders.GoProDataset(cfg.DATASET.DATASET_TRAIN_PATH, train_transforms),
        batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset = utils.data_loaders.GoProDataset(cfg.DATASET.DATASET_TEST_PATH, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)
    ### End GroPro

    # Set up networks
    #print('---test models is {}'.format(models.__dict__[cfg.NETWORK.DEBLURNETARCH].__dict__[cfg.NETWORK.DEBLURNETARCH]()))
    # deblurnet = models.__dict__[cfg.NETWORK.DEBLURNETARCH].__dict__[cfg.NETWORK.DEBLURNETARCH]()
    deblurnet = DeblurNet()

    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.DEBLURNETARCH,
                                                utils.network_utils.count_parameters(deblurnet)))

    # Initialize weights of networks
    deblurnet.apply(utils.network_utils.init_weights_xavier)
    # Set up solver
    deblurnet_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, deblurnet.parameters()), lr=cfg.TRAIN.DEBLURNET_LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    if torch.cuda.is_available():
        deblurnet = torch.nn.DataParallel(deblurnet).cuda()

    # Load pretrained model if exists
    init_epoch       = 0
    Best_Epoch       = -1
    Best_Disp_EPE    = float('Inf')
    Best_Img_PSNR    = 0
    if cfg.NETWORK.PHASE in ['test', 'resume']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        if cfg.NETWORK.MODULE == 'deblurnet':
            deblurnet.load_state_dict(checkpoint['deblurnet_state_dict'])
            init_epoch = checkpoint['epoch_idx']+1
            Best_Img_PSNR = checkpoint['Best_Img_PSNR']
            Best_Epoch = checkpoint['Best_Epoch']
            deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
            print('[INFO] {0} Recover complete. Current epoch #{1}, Best_Img_PSNR = {2} at epoch #{3}.' \
                  .format(dt.now(), init_epoch, Best_Img_PSNR, Best_Epoch))
            

    # Set up learning rate scheduler to decay learning rates dynamically
    deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(deblurnet_solver,
                                                                   milestones=cfg.TRAIN.DEBLURNET_LR_MILESTONES,
                                                                   gamma=cfg.TRAIN.LEARNING_RATE_DECAY)

    # Summary writer for TensorBoard
    #if cfg.NETWORK.MODULE == 'deblurnet':
        #output_dir   = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat()+'_'+cfg.NETWORK.DEBLURNETARCH, '%s')

    log_dir      = 'output/' + 'bs2_decay5_leakyrelu_ratio162_conv'
    ckpt_dir     = 'ckpt'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer  = SummaryWriter(os.path.join(log_dir, 'test'))

    if cfg.NETWORK.PHASE in ['train', 'resume']:
        # train and val
        if cfg.NETWORK.MODULE == 'deblurnet':
            train_deblurnet(cfg, init_epoch, train_data_loader, test_data_loader, deblurnet, deblurnet_solver,
                              deblurnet_lr_scheduler, ckpt_dir, train_writer, test_writer, Best_Img_PSNR, Best_Epoch)
            return
    else:
        assert os.path.exists(cfg.CONST.WEIGHTS),'[FATAL] Please specify the file path of checkpoint!'
        if cfg.NETWORK.MODULE == 'deblurnet':
            test_deblurnet(cfg, init_epoch, test_data_loader, deblurnet, test_writer)
            return
      
