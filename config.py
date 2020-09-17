
from easydict import EasyDict as edict
import socket

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = 'all'                   # '0'
__C.CONST.NUM_WORKER                    = 1                       # number of data workers
__C.CONST.WEIGHTS                       = 'ckpt/best-noleakyrelu_bs2_l1loss_nopcloss_decay0.5___-ckpt.pth.tar'
__C.CONST.TRAIN_BATCH_SIZE              = 2
__C.CONST.TEST_BATCH_SIZE               = 1

#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'GoPro'          # FlyingThings3D, StereoDeblur, GoPro
__C.DATASET.DATASET_TRAIN_PATH          = './train_blur_file.txt'          # training data
__C.DATASET.DATASET_TEST_PATH           = './test_blur_file.txt'          # testing data
__C.DATASET.WITH_MASK                   = True




#
# Directories
#
__C.DIR                                 = edict()
#__C.DIR.OUT_PATH = '/data/code/StereodeblurNet/output'
__C.DIR.OUT_PATH = './output'

if cfg.DATASET.DATASET_NAME == 'GoPro':
    # __C.DIR.DATASET_JSON_FILE_PATH          = './datasets/flyingthings3d.json'
    __C.DIR.DATASET_ROOT                    = '/data/hq/unetattentionDeblur/datasets/GoPro/'

#
# data augmentation
#

__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.DIV_DISP                       = 40.0                    # 40.0 for disparity
__C.DATA.CROP_IMG_SIZE                  = [256, 256]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.DEBLURNETARCH               = 'DeblurNet'       # available options: DeblurNet, StereoDeblurNet
__C.NETWORK.LEAKY_VALUE                 = 0.1
__C.NETWORK.BATCHNORM                   = False
__C.NETWORK.PHASE                       = 'train'                 # available options: 'train', 'test', 'resume'
__C.NETWORK.MODULE                      = 'deblurnet'                   # available options: 'dispnet', 'deblurnet', 'all'

#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PERCET_LOSS               = False
__C.TRAIN.NUM_EPOCHES                   = 2000                     # maximum number of epoches
__C.TRAIN.BRIGHTNESS                    = .25
__C.TRAIN.CONTRAST                      = .25
__C.TRAIN.SATURATION                    = .25
__C.TRAIN.HUE                           = .25
__C.TRAIN.DEBLURNET_LEARNING_RATE       = 1e-4

__C.TRAIN.DEBLURNET_LR_MILESTONES       = [600,800,900,950,1000]  # lr1

__C.TRAIN.LEARNING_RATE_DECAY           = 0.5                     # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                     # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                     # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ                    = 10
__C.TRAIN.SAVE_FREQ                     = 5                       # weights will be overwritten every save_freq epoch

__C.LOSS                                = edict()
__C.LOSS.MULTISCALE_WEIGHTS             = [0.3, 0.3, 0.2, 0.1, 0.1]

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.VISUALIZATION_NUM              = 3
__C.TEST.PRINT_FREQ                     = 5
if __C.NETWORK.PHASE == 'test':
    __C.CONST.TEST_BATCH_SIZE           = 1
