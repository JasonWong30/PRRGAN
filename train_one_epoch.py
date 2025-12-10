# --------------------------------------------------------
import math
import sys
from typing import Iterable
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import os

#model, netD,data_loader_train,optimizer_1, optimizer_2, device, epoch, loss_scaler,log_writer=log_writer, args=args
def train_one_epoch(model: torch.nn.Module, netD:torch.nn.Module,
                    data_loader: Iterable, optimizer_1: torch.optim.Optimizer,
                    optimizer_2: torch.optim.Optimizer,device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40
    accum_iter = args.accum_iter

    criterion_BCEloss = torch.nn.BCEWithLogitsLoss()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (color, other, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer_1, data_iter_step / len(data_loader) + epoch, args)            

        color = color.to(device, non_blocking=True)
        other = other.to(device, non_blocking=True)
        T2 = T2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_ssim, out = model(color, other, T2)
        optimizer_2.zero_grad()
        D_real = netD(other)
        D_real_loss = criterion_BCEloss(D_real, torch.ones_like(D_real))
        loss_scaler(D_real_loss, optimizer_2, parameters=netD.parameters(), update_grad=False)  # clip_grad

        D_fake = netD(out.detach())
        D_fake_loss = criterion_BCEloss(D_fake, torch.zeros_like(D_fake))
        loss_scaler(D_fake_loss, optimizer_2, parameters=netD.parameters(), update_grad=True)  # clip_grad

        G_GAN_D = netD(out)
        G_loss_GAN_D = criterion_BCEloss(G_GAN_D, torch.ones_like(G_GAN_D))
        optimizer_1.zero_grad()
        loss = loss_ssim + G_loss_GAN_D
        # loss = loss_ssim
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer_1, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer_1.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ 
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
