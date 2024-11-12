# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
from torch.utils.data import DataLoader
from util.loader import Med_dataset_DMIF
from model.net2 import MODEL
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
def tensor2img(img, is_norm=True):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  if is_norm:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)

def save_img_single(img, name, size, is_norm=True):
  img = tensor2img(img, is_norm=True)
  img = Image.fromarray(img[:,:,0])
  img = img.resize(size)
  img.save(name)

def main(data_dir, save_dir, fusion_model_path):

    fusionmodel = MODEL()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(fusion_model_path)['model'])
    fusionmodel = fusionmodel.to(device)
    fusionmodel.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = Med_dataset_DMIF('./Dataset_2', 'test/test_PET-MRI', 'test', transform_test)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=args.batch_size,shuffle=False,
        num_workers=args.num_workers,pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            _, _, H, W = img_vis.shape
            size = (W, H)

            _, logits = fusionmodel(img_ir, img_ir)

            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(logits[k, ::], save_path, size)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./output_dir/checkpoint-120.pth')
    ## dataset
    parser.add_argument('--data_dir', '-data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./M3-UNet')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    main(data_dir=args.data_dir, save_dir=args.save_dir, fusion_model_path=args.model_path)
