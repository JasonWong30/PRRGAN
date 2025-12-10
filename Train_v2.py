import argparse
from pathlib import Path
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torchvision.transforms as transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import train_one_epoch
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
# from model.net import MODEL
from model.net import MODEL
import datetime
import json
import time
import timm.optim.optim_factory as optim_factory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util.loader import Med_dataset
from model.Discriminator import Discriminator
import torch.optim as optim


def get_args_parser():
    parser = argparse.ArgumentParser('GCN pre-training', add_help=False)
    parser.add_argument('--batch_size', default=90, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
   
    parser.add_argument('--save_dir', default='./output_dir', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.005, #0.05
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', #1e-3
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--data_dir', default='./Dataset_2', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

    return parser

def valid(val_loader, network):
    torch.cuda.empty_cache()
    network.eval()
    for  color, other, _  in val_loader:
        with torch.no_grad():  # torch.no_grad() may cause warning
            loss, pred = network(color.to('cuda').float(),  other.to('cuda').float())
            fig = write_img(pred)
    return fig

def write_img(img):
    img = img.cpu().numpy()
    fig = plt.figure(figsize=(9, 4),dpi=100)
    for i in range(2):
        # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
        ax = fig.add_subplot(1, 3, i + 1, xticks=[], yticks=[])
        # CHW -> HWC
        npimg = img[i].transpose(1, 2, 0)        
        npimg = npimg * 255 #        
        npimg = np.clip(npimg, 0, 255)
        # plt.imshow(npimg.astype('uint8'))
        plt.imshow(npimg.astype('uint8'), cmap=plt.get_cmap('gray'))
    return fig

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_valid = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_train = Med_dataset(args.data_dir, 'train', 'train', transform=transform_train)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    dataset_valid = Med_dataset(args.data_dir, 'valid', 'valid', transform=transform_valid)   
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=2, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True)
    model = MODEL()
    netD = Discriminator()

    model.to(device)
    netD.to(device)
    # checkpoint = torch.load('./output_dir/checkpoint-1000.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model'], strict=True)

    model_without_ddp = model
    netD_without_ddp = netD

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
        netD_without_ddp = netD.module

    param_groups_1 = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups_2 = optim_factory.add_weight_decay(netD_without_ddp, args.weight_decay)

    optimizer_1 = torch.optim.AdamW(param_groups_1, lr=args.lr, betas=(0.9, 0.999))    
    optimizer_2 = torch.optim.AdamW(param_groups_2, lr=args.lr, betas=(0.9, 0.999))

    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer_1, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch.train_one_epoch(
            model, netD, data_loader_train,
            optimizer_1, optimizer_2, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer_1,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                fig = valid(data_loader_valid, model)
                log_writer.add_figure('prediction', figure=fig, global_step=epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # wandb.login()
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)