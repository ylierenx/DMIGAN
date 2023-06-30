from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from PIL import Image
import sys


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()


    return y


class UCF101(Dataset):
    def __init__(self, gop_size=9, batch_size=16, image_size=160, load_filename='fnames_shuffle_train.npy'):
        self.load_filename = load_filename
        self.fnames = np.load(self.load_filename)
        self.fnames = self.fnames.tolist()
        self.gop_size = gop_size
        self.batch_size = batch_size
        self.frame_height = 240
        self.frame_width = 320
        self.image_size = image_size

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        #video_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        video_x = self.get_single_video_x(self.fnames[idx])
        return video_x

    def get_single_image(self, image_path):
        image = load_img(image_path)
        input_compose = Compose([CenterCrop(self.image_size), Resize(self.image_size), ToTensor()])
        image = input_compose(image)
        return image

    def get_single_video_x(self, file_dir):
        a_file = os.listdir(file_dir)

        a_file.sort(key=lambda x: int(x[:-4]))

        #frames = [os.path.join(file_dir, img) for img in a_file]
        frame_count = len(a_file)

        train_x = torch.Tensor(3, 1, self.image_size, self.image_size)

        image_start = random.randint(0, frame_count - self.gop_size)
        image_id = image_start

        s = "%06d" % image_id
        image_name = s + '.jpg'
        image_path = os.path.join(file_dir, image_name)
        single_image = self.get_single_image(image_path)
        train_x[0, :, :, :] = single_image

        image_index_i = random.randint((image_id+1), (image_id+self.gop_size-2))
        s = "%06d" % image_index_i
        image_name = s + '.jpg'
        image_path = os.path.join(file_dir, image_name)
        single_image = self.get_single_image(image_path)
        train_x[1, :, :, :] = single_image

        image_index = image_id+self.gop_size-1
        s = "%06d" % image_index
        image_name = s + '.jpg'
        image_path = os.path.join(file_dir, image_name)
        single_image = self.get_single_image(image_path)
        train_x[2, :, :, :] = single_image

        return train_x, image_index_i-image_id-1
# if __name__=='__main__':
#     myUCF101 = UCF101()
#     dataloader = DataLoader(myUCF101, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)
#     for i_batch, sample_batched in enumerate(dataloader):
#         if ((i_batch % 100) == 0):
#             print(i_batch, sample_batched.size())
