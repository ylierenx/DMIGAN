import math
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model_dmigan
import dataloder1
from math import log10
import sys
import time
from PIL import Image
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
import scipy.io as scio
import random
from torchvision.transforms import Compose, ToTensor

import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:0")
load_flag = False

sampling_rate_1 = 0.5
iw_train = 160
ih_train = 160
iw_test = 192
ih_test = 160
sample_and_recon = model_dmigan.hsganet(num_filters1=int(int(sampling_rate_1*1024)*0.8), num_filters2=int(sampling_rate_1*1024)-int(int(sampling_rate_1*1024)*0.8), B_size=32)
sample_and_recon.to(device)

params_key = list(sample_and_recon.parameters())

optimizer_key = optim.Adam(params_key, lr=0.0001)

trainset = dataloder1.UCF101(gop_size=1, image_size=160)
train_loader = dataloder1.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True,
                                     drop_last=True)

if load_flag:
    dict1 = torch.load('./check_point/key_0.5_hsganet.ckpt')
    sample_and_recon.load_state_dict(dict1['state_dict_sample_and_recon'])
    optimizer_key.load_state_dict(dict1['state_dict_optimizer_key'])

else:
    dict1 = {'epoch': -1}

checkpoint_counter = 1

# load matrix

p_name1 = './permutation_matrix/' + str(iw_test) + '.mat'
p_name1_t = './permutation_matrix/' + str(iw_test) + '_t' + '.mat'
p_name2 = './permutation_matrix/' + str(ih_test) + '.mat'
p_name2_t = './permutation_matrix/' + str(ih_test) + '_t' + '.mat'

data = scio.loadmat(p_name1)
rand_sw_p1 = data['rand_sw_p1']
rand_sw_p1 = np.array(rand_sw_p1).astype(np.float32)
data_t = scio.loadmat(p_name1_t)
rand_sw_p1_t = data_t['rand_sw_p1_t']
rand_sw_p1_t = np.array(rand_sw_p1_t).astype(np.float32)

data = scio.loadmat(p_name2)
rand_sw_p2 = data['rand_sw_p1']
rand_sw_p2 = np.array(rand_sw_p2).astype(np.float32)
data_t = scio.loadmat(p_name2_t)
rand_sw_p2_t = data_t['rand_sw_p1_t']
rand_sw_p2_t = np.array(rand_sw_p2_t).astype(np.float32)

rand_sw_p1 = torch.from_numpy(rand_sw_p1)
rand_sw_p1_t = torch.from_numpy(rand_sw_p1_t)
test_rand_sw_p1 = rand_sw_p1.to(device)
test_rand_sw_p1_t = rand_sw_p1_t.to(device)

rand_sw_p2 = torch.from_numpy(rand_sw_p2)
rand_sw_p2_t = torch.from_numpy(rand_sw_p2_t)
test_rand_sw_p2 = rand_sw_p2.to(device)
test_rand_sw_p2_t = rand_sw_p2_t.to(device)

p_name1 = './permutation_matrix/' + str(iw_train) + '.mat'
p_name1_t = './permutation_matrix/' + str(iw_train) + '_t' + '.mat'
p_name2 = './permutation_matrix/' + str(ih_train) + '.mat'
p_name2_t = './permutation_matrix/' + str(ih_train) + '_t' + '.mat'

data = scio.loadmat(p_name1)
rand_sw_p1 = data['rand_sw_p1']
rand_sw_p1 = np.array(rand_sw_p1).astype(np.float32)
data_t = scio.loadmat(p_name1_t)
rand_sw_p1_t = data_t['rand_sw_p1_t']
rand_sw_p1_t = np.array(rand_sw_p1_t).astype(np.float32)

data = scio.loadmat(p_name2)
rand_sw_p2 = data['rand_sw_p1']
rand_sw_p2 = np.array(rand_sw_p2).astype(np.float32)
data_t = scio.loadmat(p_name2_t)
rand_sw_p2_t = data_t['rand_sw_p1_t']
rand_sw_p2_t = np.array(rand_sw_p2_t).astype(np.float32)

rand_sw_p1 = torch.from_numpy(rand_sw_p1)
rand_sw_p1_t = torch.from_numpy(rand_sw_p1_t)
train_rand_sw_p1 = rand_sw_p1.to(device)
train_rand_sw_p1_t = rand_sw_p1_t.to(device)

rand_sw_p2 = torch.from_numpy(rand_sw_p2)
rand_sw_p2_t = torch.from_numpy(rand_sw_p2_t)
train_rand_sw_p2 = rand_sw_p2.to(device)
train_rand_sw_p2_t = rand_sw_p2_t.to(device)



def load_img(filepath):
    img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    # img = img[:, :, np.newaxis]
    img = np.pad(img, ((0, 16), (0, 16)), 'constant', constant_values=0)
    return img


def get_single_image(image_path):
    image = load_img(image_path)
    input_compose = Compose([ToTensor()])
    image = input_compose(image)
    return image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "tif"])


def PSNR(img_rec, img_ori):
    img_rec = img_rec.astype(np.float32)
    img_ori = img_ori.astype(np.float32)
    max_gray = 1.
    mse = np.mean(np.power(img_rec - img_ori, 2))
    return 10. * np.log10(max_gray ** 2 / mse)


def adjust_learning_rate1(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test_hsganet(block_size=32):
    for dataset_name in sorted(os.listdir('./dataset1')):
        psnr_value = []
        fnames = []
        dataset_n = os.path.join('./dataset1', dataset_name)
        for file_name in sorted(os.listdir(dataset_n)):
            fnames.append(os.path.join(dataset_n, file_name))
    with torch.no_grad():
        for f_i in range(len(fnames)):

            img = Image.open(fnames[f_i])
            I = np.array(img)
            I = Image.fromarray(I)

            input_compose = Compose([ToTensor()])
            I = input_compose(I)
            I = I.unsqueeze(0)

            inputs = I

            ih = I.shape[2]
            iw = I.shape[3]

            if np.mod(iw, block_size) != 0:
                col_pad = block_size - np.mod(iw, block_size)
                inputs = torch.cat((inputs, torch.zeros([1, 1, ih, col_pad])), axis=3)
            else:
                col_pad = 0
                inputs = inputs
            if np.mod(ih, block_size) != 0:
                row_pad = block_size - np.mod(ih, block_size)
                inputs = torch.cat((inputs, torch.zeros([1, 1, row_pad, iw + col_pad])), axis=2)
            else:
                row_pad = 0
            inputs = inputs.cuda()

            x_output = sample_and_recon(inputs, test_rand_sw_p1, test_rand_sw_p1_t, test_rand_sw_p2, test_rand_sw_p2_t)

            x_output = x_output.cpu().numpy()
            I = I.cpu().numpy()
            recon_img = x_output[0, 0, :ih, :iw]
            ori_img = I[0, 0, :ih, :iw]
            p1 = PSNR(recon_img, ori_img)
            #print(p1)
            psnr_value.append(p1)
    return np.mean(psnr_value)


if __name__ == '__main__':
    start = time.time()
    
    for epoch in range(0, 521):

        if epoch >= 500 and epoch < 510:
            lr = 0.00001
            adjust_learning_rate1(optimizer_key, lr)

        if epoch >= 510:
            lr = 0.000001
            adjust_learning_rate1(optimizer_key, lr)


        for i, inputs in enumerate(train_loader):
            # if epoch>2 and gb_flag<20:
            #    optimizer_key = optim.Adam(params_key, lr=0.0001)

            inputs = inputs[:, 0, :, :, :]
            inputs = inputs.to(device)

            optimizer_key.zero_grad()

            x_output = sample_and_recon(inputs, train_rand_sw_p1, train_rand_sw_p1_t, train_rand_sw_p2, train_rand_sw_p2_t)

            recnLoss_final = torch.mean(
                torch.norm((inputs - x_output), p=2, dim=(2, 3)) * torch.norm((inputs - x_output), p=2, dim=(2, 3)))

            recnLoss_final.backward()
            optimizer_key.step()

            if ((i % 100) == 0):
                print('test')

                p1 = test_hsganet()

                print("0.1:%0.6f" % (p1))

                print("train_loss: %0.6f Iterations: %4d/%4d epoch:%d " % (
                    recnLoss_final.item(), i, len(train_loader), epoch))

                f = open('test_key_dmigan_0.5.txt', 'a')
                f.write("%0.6f %d %0.6f" % (p1, epoch, recnLoss_final.item()))
                f.write('\n')
                f.close()

                end = time.time()
                print(end - start)
                start = time.time()
        if ((epoch % 10) == 0 and epoch > 0):
            dict1 = {
                'Detail': "dmigan",
                'epoch': epoch,

                'state_dict_sample_and_recon': sample_and_recon.state_dict(),

                'state_dict_optimizer_key': optimizer_key.state_dict()

            }
            torch.save(dict1, './check_point' + "/key_0.5_hsganet_" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1
        













