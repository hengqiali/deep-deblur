import os
import torch.backends.cudnn
import torch.utils.data

import utils.data_transforms
import utils.network_utils
import torchvision

from losses.multiscaleloss import *
from time import time

from core.test_deblur import test_deblurnet
from models.VGG19 import VGG19
import torch
import random

random.seed(1234)
torch.manual_seed(1234)

def train_deblurnet(cfg, init_epoch, train_data_loader, val_data_loader, deblurnet, deblurnet_solver,
                    deblurnet_lr_scheduler, ckpt_dir, train_writer, val_writer, Best_Img_PSNR, Best_Epoch):
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        test_time = utils.network_utils.AverageMeter()
        deblur_losses = utils.network_utils.AverageMeter()
        mse_losses = utils.network_utils.AverageMeter()
        if cfg.TRAIN.USE_PERCET_LOSS:
            percept_losses = utils.network_utils.AverageMeter()
        img_PSNRs = utils.network_utils.AverageMeter()

        # Adjust learning rate
        deblurnet_lr_scheduler.step()
        

        batch_end_time = time()
        n_batches = len(train_data_loader)
        if cfg.TRAIN.USE_PERCET_LOSS:
            vggnet = VGG19()
            if torch.cuda.is_available():
                vggnet = torch.nn.DataParallel(vggnet).cuda()
        #print('\n\n\n-----------  train   -------------\n\n\n')
        for batch_idx, (imgs, labels) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)
            # Get data from data loader
            img_blur_left = utils.network_utils.var_or_cuda(imgs)
            img_clear_left = utils.network_utils.var_or_cuda(labels)
            
            # switch models to training mode
            deblurnet.train()

            output_img_clear_left = deblurnet(img_blur_left)

            # mse_left_loss  = mseLoss(output_img_clear_left, img_clear_left)
            mse_left_loss  = l1Loss(output_img_clear_left, img_clear_left)
            if cfg.TRAIN.USE_PERCET_LOSS:
                percept_left_loss  = perceptualLoss(output_img_clear_left, img_clear_left, vggnet)
                deblur_left_loss  = mse_left_loss + 0.01 * percept_left_loss
            else:
                deblur_left_loss = mse_left_loss

            img_PSNR_left = PSNR(output_img_clear_left, img_clear_left)

            # Gradient decent
            deblurnet_solver.zero_grad()
            deblur_left_loss.backward()
            deblurnet_solver.step()

            mse_loss = mse_left_loss
            mse_losses.update(mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            if cfg.TRAIN.USE_PERCET_LOSS:
                percept_loss = 0.01 *percept_left_loss
                percept_losses.update(percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            deblur_loss = deblur_left_loss
            deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            img_PSNR = img_PSNR_left
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('DeblurNet/MSELoss_0_TRAIN', mse_loss.item(), n_itr)
            if cfg.TRAIN.USE_PERCET_LOSS:
                train_writer.add_scalar('DeblurNet/PerceptLoss_0_TRAIN', percept_loss.item(), n_itr)
            train_writer.add_scalar('DeblurNet/DeblurLoss_0_TRAIN', deblur_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                if cfg.TRAIN.USE_PERCET_LOSS:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t  Loss {4} [{5}, {6}]\t PSNR {7}'
                        .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches,
                                deblur_losses, mse_losses, percept_losses, img_PSNRs))
                else:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t  DeblurLoss {4} \t PSNR {5}'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches,
                                  deblur_losses, img_PSNRs))
            """
            if batch_idx < cfg.TEST.VISUALIZATION_NUM:

                img_left_blur = imgs[0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_left_clear = labels[0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                out_left = output_img_clear_left[0][[2,1,0],:,:].cpu().clamp(0.0,1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                result = torch.cat([img_left_blur, img_left_clear, out_left],1)
                result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                train_writer.add_image('DeblurNet/TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)
            """

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('DeblurNet/EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t  DeblurLoss_avg {2}\t ImgPSNR_avg {3}'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, deblur_losses.avg,
                      img_PSNRs.avg))
        

        if epoch_idx == 0:
            img_PSNR = test_deblurnet(cfg, epoch_idx, val_data_loader, deblurnet, val_writer)
        # Validate the training models
        if (epoch_idx >= 750 or epoch_idx == 5):
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            Best_Epoch = epoch_idx + 1
            print('\nLearning rate at this epoch is: %0.9f' % deblurnet_lr_scheduler.get_lr()[0])
            utils.network_utils.save_deblur_checkpoints(os.path.join(ckpt_dir, 'bs2_decay5_leakyrelu_ratio162_conv-ckpt.pth.tar'), \
                                                    epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
                                                    Best_Epoch)
            print('\n\n\n-----------  test   -------------\n\n\n')
            img_PSNR = test_deblurnet(cfg, epoch_idx, val_data_loader, deblurnet, val_writer)

            # Save weights to file
            # if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            #     if not os.path.exists(ckpt_dir):
            #         os.makedirs(ckpt_dir)

            #     utils.network_utils.save_deblur_checkpoints(os.path.join(ckpt_dir, 'ckpt-pkernel_leakyrelu_bs2_l1loss_nopcloss-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
            #                                         epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
            #                                         Best_Epoch)
            if img_PSNR > Best_Img_PSNR:
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                Best_Img_PSNR = img_PSNR
                Best_Epoch = epoch_idx + 1
                utils.network_utils.save_deblur_checkpoints(os.path.join(ckpt_dir, 'bs2_decay5_leakyrelu_ratio162_conv-ckpt.pth.tar'), \
                                                     epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
                                                     Best_Epoch)
        print("Utile Current Best PSNR = {}".format(Best_Img_PSNR))
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


