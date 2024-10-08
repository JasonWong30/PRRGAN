# coding:utf-8
import torchvision.transforms.functional as TF
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import glob
# from natsort import natsorted
import numpy as np
import cv2
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

class Med_dataset(Dataset):
    def __init__(self, data_dir, sub_dir, mode, transform):
        super(Med_dataset, self).__init__()
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'Y_color')))
        self.length = len(self.img_names)
        self.transform = transform  #


    def __getitem__(self, index):
        img_name = self.img_names[index]

        color = cv2.imread(os.path.join(self.root_dir, 'Y_color', img_name), 0)
        other = cv2.imread(os.path.join(self.root_dir, 'other', img_name), 0)
        T2 = cv2.imread(os.path.join(self.root_dir, 'T2', img_name), 0)

        color = Image.fromarray(color)
        other = Image.fromarray(other)
        T2 = Image.fromarray(T2)

        color = self.transform(color)
        other = self.transform(other)
        T2 = self.transform(T2)

        return color, other, T2, img_name
    def __len__(self):
        return self.length


class Med_dataset_DMIF(Dataset):
    def __init__(self, data_dir, sub_dir, mode, transform):
        super(Med_dataset_DMIF, self).__init__()
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'Y_color')))
        self.length = len(self.img_names)
        self.transform = transform  #


    def __getitem__(self, index):
        img_name = self.img_names[index]  #11111111

        color = cv2.imread(os.path.join(self.root_dir, 'Y_color', img_name), 0)
        other = cv2.imread(os.path.join(self.root_dir, 'MRI', img_name), 0)

        color = Image.fromarray(color)
        other = Image.fromarray(other)

        color = self.transform(color)
        other = self.transform(other)

        return color, other, img_name

    def __len__(self):
        return self.length


class ivif_dataset(Dataset):
    def __init__(self, data_dir, sub_dir, mode, transform):
        super(ivif_dataset, self).__init__()
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'Y_color')))
        self.length = len(self.img_names)
        self.transform = transform  #

    def __getitem__(self, index):
        img_name = self.img_names[index]

        color = cv2.imread(os.path.join(self.root_dir, 'Y_color', img_name), 0)
        other = cv2.imread(os.path.join(self.root_dir, 'other', img_name), 0)


        color = Image.fromarray(color)
        other = Image.fromarray(other)

        color = self.transform(color)
        other = self.transform(other)

        return color, other, img_name


    def __len__(self):
        return self.length

