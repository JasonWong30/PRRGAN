# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from util import *
import torch
from torch.utils.data import DataLoader
from util.loader import Med_dataset
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.nn.functional
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2

def save_img_single(img, name):
    img = tensor2img(img, is_norm=True)
    img = Image.fromarray(img)
    img = img.convert("L")
    img.save(name)

def tensor2img(img, is_norm=True):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  if is_norm:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)


def main(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model = eval('MODEL')()
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)

    print('fusionmodel load done!')
    test_dataset = Med_dataset(args.data_dir, 'test_Gad_T2_PET', 'test', transform=transform_test)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    time_list = []
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (color_Y, other, T2, name) in enumerate(test_bar):
            color_Y = color_Y.to(device, non_blocking=True)
            other = other.to(device, non_blocking=True)
            T2 = T2.to(device, non_blocking=True)
            start_time = time.time()
            _, fused_img = model(color_Y,  other, T2)
            end_time = time.time()
            print(end_time-start_time)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(args.save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))

    print(time_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./output_dir/checkpoint-650.pth')
    parser.add_argument('--data_dir', type=str, default='./Dataset/')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./GCN/')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--Train', type=bool, default=False)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    main(args)


