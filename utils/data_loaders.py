#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import cv2
import json
import numpy as np
import os
import io
import random
import scipy.io
import sys
import torch.utils.data.dataset

from config import cfg
from datetime import datetime as dt
from enum import Enum, unique
from utils.imgio_gen import readgen
import utils.network_utils

class DatasetType(Enum):
    TRAIN = 0
    TEST  = 1

class GoProDataset(torch.utils.data.dataset.Dataset):
    """GoProDataset class used for PyTorch DataLoader"""

    def __init__(self, list_dir, transforms = None):
        file = open(list_dir)
        self.list = file.readlines()
        self.transforms = transforms

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):

        imgs = self.get_datum(idx)
        imgs = self.transforms(imgs)

        return imgs[0], imgs[1]

    def get_datum(self, idx):

        # grounth_path = cfg.DIR.DATASET_ROOT + cfg.NETWORK.PHASE + '/' + self.list[idx][:-1][0:15] + 'sharp' + self.list[idx][:-1][19:30]
        # sample_path = cfg.DIR.DATASET_ROOT + cfg.NETWORK.PHASE + '/' + self.list[idx][:-1]

        grounth_path = cfg.DIR.DATASET_ROOT + self.list[idx][:-16] + 'sharp' + self.list[idx][-12:-1]
        sample_path = cfg.DIR.DATASET_ROOT  + self.list[idx][:-1]

        # print(sample_path)

        img_blur = readgen(sample_path).astype(np.float32)
        img_clear = readgen(grounth_path).astype(np.float32)
        imgs = [img_blur, img_clear]

        return imgs
