import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model_dmigan
import scipy.io as scio

from math import log10
import sys
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from torchvision.transforms import Compose, ToTensor
import cv2
from os.path import join
from os import listdir
import math
import numpy as np
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import xlwt
from scipy.signal import convolve2d
import xlwt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

rgb = False
flag = True
block_size = 32
device = torch.device("cuda:0")
gop_size = 8
image_width = 192
image_height = 160
img_w = 176
img_h = 144
num_gop = 12

test_video_name = 'test_video'
sr = 0.1
sampling_rate_1 = 0.5
sample_and_recon_key = model_dmigan.hsganet(num_filters1=int(int(sampling_rate_1 * 1024) * 0.8),
                                  num_filters2=int(sampling_rate_1 * 1024) - int(int(sampling_rate_1 * 1024) * 0.8),
                                  B_size=32)
sample_and_recon_key.to(device)
# nonkey reconstruction
sample_and_recon_nonkey = model_dmigan.sample_recon_nonkey_map(num_filters=int(1024*sr), B_size=32, device=device)
sample_and_recon_nonkey.to(device)

if flag:
    dict1 = torch.load('./check_point/dmigan_0.5_0.1.ckpt')
    sample_and_recon_key.load_state_dict(dict1['state_dict_sample_and_recon_key'])
    sample_and_recon_nonkey.load_state_dict(dict1['state_dict_sample_and_recon_nonkey'])
else:
    dict1 = {'epoch': -1}

# load matrix

p_name1 = './permutation_matrix/' + str(image_width) + '.mat'
p_name1_t = './permutation_matrix/' + str(image_width) + '_t' + '.mat'
p_name2 = './permutation_matrix/' + str(image_height) + '.mat'
p_name2_t = './permutation_matrix/' + str(image_height) + '_t' + '.mat'

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

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def load_img(filepath):
    img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    # img = img[:, :, np.newaxis]
    # img = np.pad(img, ((0, 16), (0, 16)), 'constant', constant_values=0)
    return img


def get_single_image(image_path):
    # print(image_path)
    if rgb:
        Img = cv2.imread(image_path, 1)
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv[:, :, 0]
    else:
        # print(image_path)
        Img_yuv = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv

    # image = load_img(image_path)
    input_compose = Compose([ToTensor()])
    # image = input_compose(image)
    image = input_compose(Iorg_y)

    inputs = image[0, :, :]
    ih = inputs.shape[0]
    iw = inputs.shape[1]

    if np.mod(iw, block_size) != 0:
        col_pad = block_size - np.mod(iw, block_size)
        inputs = torch.cat((inputs, torch.zeros([ih, col_pad])), axis=1)
    else:
        col_pad = 0
        inputs = inputs
    if np.mod(ih, block_size) != 0:
        row_pad = block_size - np.mod(ih, block_size)
        inputs = torch.cat((inputs, torch.zeros([row_pad, iw + col_pad])), axis=0)
    else:
        row_pad = 0
    inputs = inputs.cuda()
    # I = I.cuda()

    # print(inputs.shape)
    # print(torch.max(image)*255)
    # sys.exit()
    return inputs, Img_rec_yuv


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "tif"])


def PSNR(data, tar):
    data = torch.squeeze(data)
    tar = torch.squeeze(tar)
    data = data.cpu().numpy()
    tar = tar.cpu().numpy()

    data = data.astype(np.float32)
    tar = tar.astype(np.float32)

    max_gray = 1.
    mse = np.mean(np.power(data - tar, 2))

    return 10. * np.log10(max_gray ** 2 / mse)


if __name__ == '__main__':
    workbook = xlwt.Workbook(encoding='utf-8')
    video_index = 0
    worksheet = workbook.add_sheet('text_video_psnr_ssim')
    worksheet_1 = workbook.add_sheet('text_video_psnr_ssim_ave')
    psnr_ave = []
    ssim_ave = []

    # temp_psnr = []
    # temp_ssim = []
    for video_name in sorted(os.listdir('./' + test_video_name)):
        psnr_ave = []
        ssim_ave = []
        worksheet.write(0, video_index, video_name + '_psnr')
        worksheet.write(0, video_index + 1, video_name + '_ssim')
        print(video_name)
        file_path = './' + test_video_name + '/' + video_name
        if not os.path.exists('./' + test_video_name + '_output'):
            os.mkdir('./' + test_video_name + '_output')
        if not os.path.exists('./' + test_video_name + '_output/' + video_name + str(sr)):
            os.mkdir('./' + test_video_name + '_output/' + video_name + str(sr))
        if not os.path.exists('./' + test_video_name + '_output/psnr_ssim'):
            os.mkdir('./' + test_video_name + '_output/psnr_ssim')

        input_tensor = torch.zeros([1, gop_size + 1, 1, image_height, image_width])
        file = listdir(file_path)
        file.sort()
        image_filenames = [join(file_path, x) for x in file if is_image_file(x)]
        # num_gop = len(image_filenames) / gop_size

        img_index = 1
        with torch.no_grad():
            for i in range(int(num_gop)):
                if img_index == (len(image_filenames) - 8):
                    break
                yuv_recon = []
                start = time.time()
                for j in range(int(gop_size) + 1):
                    # according to the index format
                    # s = "%03d" % (j + i * gop_size + 1)
                    # s = 'im' + s
                    # image_name = s + '.png'
                    s = "%06d" % (j + i * gop_size)
                    image_name = s + '.tif'

                    image_path = os.path.join(file_path, image_name)
                    single_image, single_yuv = get_single_image(image_path)
                    yuv_recon.append(single_yuv)
                    input_tensor[0, j, 0, :, :] = single_image

                frame0 = input_tensor[:, 0, :, :, :]
                frame8 = input_tensor[:, 8, :, :, :]
                F_0 = frame0.to(device)
                F_8 = frame8.to(device)
                # key_initial reconstruction
                out_F_0 = sample_and_recon_key(F_0, test_rand_sw_p1, test_rand_sw_p1_t, test_rand_sw_p2,
                                                test_rand_sw_p2_t)

                out_F_8 = sample_and_recon_key(F_8, test_rand_sw_p1, test_rand_sw_p1_t, test_rand_sw_p2,
                                               test_rand_sw_p2_t)

                psnr_temp = PSNR(out_F_0[:, :, :img_h, :img_w], F_0[:, :, :img_h, :img_w])
                print(psnr_temp)


                out_F_0numpy = out_F_0.cpu().numpy()
                out_F_0numpy = out_F_0numpy[0, 0, :img_h, :img_w]
                F_0numpy = F_0.cpu().numpy()
                F_0numpy = F_0numpy[0, 0, :img_h, :img_w]

                ssim_temp = compute_ssim(np.around(out_F_0numpy * 255), np.around(F_0numpy * 255))

                Prediction_value = out_F_0.cpu().data.numpy().squeeze()
                X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
                if rgb:
                    yuv_recon[0][:, :, 0] = X_rec[:img_h, :img_w] * 255
                    im_rec_rgb = cv2.cvtColor(yuv_recon[0], cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                    cv2.imwrite(test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                        psnr_temp) + str(ssim_temp) + '.png', im_rec_rgb)
                else:
                    yuv_recon[0][:, :] = X_rec[:img_h, :img_w] * 255
                    im_rec_y = np.clip(yuv_recon[0][:, :], 0, 255).astype(np.uint8)
                    cv2.imwrite(test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                        psnr_temp) + str(ssim_temp) + '.png', im_rec_y)

                psnr_ave.append(psnr_temp)
                ssim_ave.append(ssim_temp)
                worksheet.write(img_index + 1, video_index, psnr_temp)
                worksheet.write(img_index + 1, video_index + 1, ssim_temp)
                img_index = img_index + 1


                for k in range(int(gop_size) - 1):
                    frame1 = input_tensor[:, k + 1, :, :, :]
                    F_1 = frame1.to(device)
                    # non_key_initial reconstruction
                    out_F_1_4 = sample_and_recon_nonkey(F_1, out_F_0, out_F_8)

                    psnr_temp = PSNR(out_F_1_4[:, :, :img_h, :img_w], F_1[:, :, :img_h, :img_w])
                    print(psnr_temp)


                    out_F_1_4numpy = out_F_1_4.cpu().numpy()
                    out_F_1_4numpy = out_F_1_4numpy[0, 0, :img_h, :img_w]
                    F_1numpy = F_1.cpu().numpy()
                    F_1numpy = F_1numpy[0, 0, :img_h, :img_w]

                    ssim_temp = compute_ssim(np.around(out_F_1_4numpy * 255), np.around(F_1numpy * 255))

                    Prediction_value = out_F_1_4.cpu().data.numpy().squeeze()
                    X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
                    if rgb:
                        yuv_recon[0][:, :, 0] = X_rec[:img_h, :img_w] * 255
                        im_rec_rgb = cv2.cvtColor(yuv_recon[0], cv2.COLOR_YCrCb2BGR)
                        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                        cv2.imwrite(
                            test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                                psnr_temp) + str(ssim_temp) + '.png', im_rec_rgb)
                    else:
                        yuv_recon[0][:, :] = X_rec[:img_h, :img_w] * 255
                        im_rec_y = np.clip(yuv_recon[0][:, :], 0, 255).astype(np.uint8)
                        cv2.imwrite(
                            test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                                psnr_temp) + str(ssim_temp) + '.png', im_rec_y)

                    # f = open('./test_video/psnr_ssim/' + video_name + '_output/final/' + video_name + '_psnr_final.txt', 'a')
                    # f.write(" %0.6f " % (psnr_temp))
                    # f.write('\n')
                    # f.close()
                    # f = open('./test_video/psnr_ssim/' + video_name + '_output/final/' + video_name + '_ssim_final.txt', 'a')
                    # f.write(" %0.6f " % (ssim_temp))
                    # f.write('\n')
                    # f.close()

                    worksheet.write(img_index + 1, video_index, psnr_temp)
                    worksheet.write(img_index + 1, video_index + 1, ssim_temp)
                    psnr_ave.append(psnr_temp)
                    ssim_ave.append(ssim_temp)
                    img_index = img_index + 1
            psnr_ave = psnr_ave[0:gop_size * num_gop]
            print(psnr_ave)
            print(np.mean(psnr_ave))
            ssim_ave = ssim_ave[0:gop_size * num_gop]
            worksheet_1.write(0, video_index, video_name + '_psnr')
            worksheet_1.write(0, video_index + 1, video_name + '_ssim')
            worksheet_1.write(1, video_index, np.mean(psnr_ave))
            worksheet_1.write(1, video_index + 1, np.mean(ssim_ave))
            video_index = video_index + 2
    workbook.save('./' + test_video_name + '_output/psnr_ssim/psnr_ssim_0.5_' + str(sr) + '.xls')
    sys.exit()