import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys
import torchvision
#import flow_vis_torch

# import flow_vis_torch
num_of = 12
num_of1 = 6
num_of2 = 6
B_size = 32


class backWarp_MH(nn.Module):

    def __init__(self, device):
        super(backWarp_MH, self).__init__()
        self.device = device

    def forward(self, img, flow, num_of=num_of, flag=True):
        B, C, H, W = img.shape
        if flag:
            img = img.repeat(1, num_of, 1, 1)
            img = torch.reshape(img, (B * num_of, C, H, W))
        else:
            img = torch.reshape(img, (B * num_of, C // num_of, H, W))

        Bf, Cf, Hf, Wf = flow.shape

        flow = torch.reshape(flow, (Bf * num_of, Cf // num_of, Hf, Wf))
        self.gridX, self.gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H

        self.gridX = torch.tensor(self.gridX, requires_grad=False, device=self.device)
        self.gridY = torch.tensor(self.gridY, requires_grad=False, device=self.device)

        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        grid = torch.stack((x, y), dim=3)
        imgOut = torch.nn.functional.grid_sample(img, grid)
        if flag:
            imgOut = imgOut.view(B, C * num_of, H, W)
        else:
            imgOut = imgOut.view(B, C, H, W)
        return imgOut


class flownet_first(nn.Module):
    def __init__(self):
        super(flownet_first, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3):
        # x = torch.cat((x1, x2), dim=1)
        x3 = x3.repeat(1, num_of, 1, 1)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x


class gd_fusion(nn.Module):
    def __init__(self):
        super(gd_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64 + 50, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=50 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=num_of + 1, kernel_size=3, stride=1, padding=1)

    def forward(self, gd1, gd2, gd3):
        x = torch.cat((gd1, gd2, gd3), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        return output


class final_fusion(nn.Module):
    def __init__(self):
        super(final_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64 + 50, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=50 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = torch.cat((gd1, gd2, gd3), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        xa = F.relu(x, inplace=True)
        x = F.interpolate(xa, scale_factor=0.5, mode='bilinear')

        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        xb = F.relu(x, inplace=True)
        x = F.interpolate(xb, scale_factor=0.5, mode='bilinear')

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)

        up1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xb, up1), dim=1)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)

        up2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((xa, up2), dim=1)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        output = self.conv10(x)

        return output


class denoising(nn.Module):
    def __init__(self):
        super(denoising, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of + 1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=num_of + 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = x + x_in
        return x


class denoising_ini(nn.Module):
    def __init__(self):
        super(denoising_ini, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = x + x_in
        return x


class flow_refine(nn.Module):
    def __init__(self):
        super(flow_refine, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of * 4 + 3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x[:, 0:num_of, :, :], x[:, num_of:, :, :]


class flownet(nn.Module):
    def __init__(self):
        super(flownet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3 * num_of + 2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=num_of * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x


class spynet_mae_mh(nn.Module):
    def __init__(self):
        super(spynet_mae_mh, self).__init__()
        self.flownet5 = flownet_first()
        self.flownet4 = flownet()
        self.flownet3 = flownet()
        self.flownet2 = flownet()
        self.flownet1 = flownet()

    def forward(self, nonkey, key0, key8, FlowBackWarp):
        frame_all_d1 = torch.cat((nonkey, key0, key8), dim=0)

        frame_all_d2 = F.interpolate(frame_all_d1, scale_factor=0.5, mode='bilinear')
        frame_all_d3 = F.interpolate(frame_all_d2, scale_factor=0.5, mode='bilinear')
        frame_all_d4 = F.interpolate(frame_all_d3, scale_factor=0.5, mode='bilinear')
        frame_all_d5 = F.interpolate(frame_all_d4, scale_factor=0.5, mode='bilinear')

        batch_size, _, _, _ = nonkey.shape

        fl_5 = self.flownet5(frame_all_d5[2 * batch_size:, :, :, :], frame_all_d5[batch_size:2 * batch_size, :, :, :],
                             frame_all_d5[0:batch_size, :, :, :])

        fl_5_up = F.interpolate(fl_5, scale_factor=2, mode='bilinear')
        x2_4_warp = FlowBackWarp(frame_all_d4[0:batch_size, :, :, :], fl_5_up)
        fl_4_re = self.flownet4(frame_all_d4[2 * batch_size:, :, :, :],
                                frame_all_d4[batch_size:2 * batch_size, :, :, :],
                                x2_4_warp, fl_5_up)
        fl_4 = fl_4_re + fl_5_up

        fl_4_up = F.interpolate(fl_4, scale_factor=2, mode='bilinear')
        x2_3_warp = FlowBackWarp(frame_all_d3[0:batch_size, :, :, :], fl_4_up)
        fl_3_re = self.flownet3(frame_all_d3[2 * batch_size:, :, :, :],
                                frame_all_d3[batch_size:2 * batch_size, :, :, :],
                                x2_3_warp, fl_4_up)
        fl_3 = fl_3_re + fl_4_up

        fl_3_up = F.interpolate(fl_3, scale_factor=2, mode='bilinear')
        x2_2_warp = FlowBackWarp(frame_all_d2[0:batch_size, :, :, :], fl_3_up)
        fl_2_re = self.flownet2(frame_all_d2[2 * batch_size:, :, :, :],
                                frame_all_d2[batch_size:2 * batch_size, :, :, :],
                                x2_2_warp, fl_3_up)
        fl_2 = fl_2_re + fl_3_up

        fl_2_up = F.interpolate(fl_2, scale_factor=2, mode='bilinear')
        x2_1_warp = FlowBackWarp(nonkey, fl_2_up)
        fl_1_re = self.flownet1(key8, key0, x2_1_warp, fl_2_up)
        fl_1 = fl_1_re + fl_2_up

        x1_1_pred = FlowBackWarp(nonkey, fl_1)

        return x1_1_pred[:, 0:num_of1, :, :], x1_1_pred[:, num_of1:, :, :], fl_1[:, 0:num_of, :, :], fl_1[:, num_of:, :,
                                                                                                     :]


class basic_block_ini(nn.Module):
    def __init__(self):
        super(basic_block_ini, self).__init__()
        self.t2image = nn.PixelShuffle(B_size)
        self.flownet_mh = spynet_mae_mh()
        self.gd_fusionnet = gd_fusion()

        self.denoisingnet = denoising()

    def forward(self, nonkey, key0, key8, FlowBackWarp, sample_cs, phi_matrix, y_label):
        y_temp = sample_cs(nonkey)
        delta_y = y_label - y_temp
        delta_xty = F.conv2d(delta_y, phi_matrix, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        nonkey1 = nonkey + delta_xty

        k8_w, k0_w, k8_flow, k0_flow = self.flownet_mh(nonkey, key0, key8, FlowBackWarp)
        delta_k8 = key8 - k8_w
        delta_k0 = key0 - k0_w

        delta_k8_inv = FlowBackWarp(img=delta_k8, flow=-k8_flow, num_of=num_of1, flag=False)
        delta_k0_inv = FlowBackWarp(img=delta_k0, flow=-k0_flow, num_of=num_of1, flag=False)
        nonkey8 = nonkey + delta_k8_inv
        nonkey0 = nonkey + delta_k0_inv

        fusion_gd = self.gd_fusionnet(nonkey1, nonkey8, nonkey0)
        fusion_gd_d = self.denoisingnet(fusion_gd)

        return fusion_gd_d, k0_flow, k8_flow


class basic_block(nn.Module):
    def __init__(self):
        super(basic_block, self).__init__()
        self.t2image = nn.PixelShuffle(B_size)
        self.flow_refinenet1 = flow_refine()
        self.gd_fusionnet = gd_fusion()

        self.denoisingnet = denoising()

    def forward(self, nonkey, key0, key8, FlowBackWarp, sample_cs, phi_matrix, y_label, flow_ini0, flow_ini8):
        y_temp = sample_cs(nonkey[:, [0], :, :])
        delta_y = y_label - y_temp
        delta_xty = F.conv2d(delta_y, phi_matrix, stride=1, padding=0)
        delta_xty = self.t2image(delta_xty)
        nonkey1 = nonkey[:, [0], :, :] + delta_xty

        warp_k0_ini = FlowBackWarp(img=nonkey[:, num_of1 + 1:, :, :], flow=flow_ini0, num_of=num_of1, flag=False)
        warp_k8_ini = FlowBackWarp(img=nonkey[:, 1:num_of1 + 1, :, :], flow=flow_ini8, num_of=num_of1, flag=False)
        # k8_w, k0_w, k8_flow, k0_flow = self.flow_refinenet1(nonkey, key0, key8, flow_ini0, flowini8, warp_k0_ini, warp_k8_ini)
        delta_flow0_gd, delta_flow8_gd = self.flow_refinenet1(nonkey, key0, key8, flow_ini0, flow_ini8, warp_k0_ini,
                                                              warp_k8_ini)
        k8_flow = delta_flow8_gd + flow_ini8
        k0_flow = delta_flow0_gd + flow_ini0
        k0_w = FlowBackWarp(img=nonkey[:, num_of1 + 1:, :, :], flow=k0_flow, num_of=num_of1, flag=False)
        k8_w = FlowBackWarp(img=nonkey[:, 1:num_of1 + 1, :, :], flow=k8_flow, num_of=num_of1, flag=False)
        delta_k8 = key8 - k8_w
        delta_k0 = key0 - k0_w
        delta_k8_inv = FlowBackWarp(img=delta_k8, flow=-k8_flow, num_of=num_of1, flag=False)
        delta_k0_inv = FlowBackWarp(img=delta_k0, flow=-k0_flow, num_of=num_of1, flag=False)
        nonkey8 = nonkey[:, 1:num_of1 + 1, :, :] + delta_k8_inv
        nonkey0 = nonkey[:, num_of1 + 1:, :, :] + delta_k0_inv

        fusion_gd = self.gd_fusionnet(nonkey1, nonkey8, nonkey0)
        fusion_gd_d = self.denoisingnet(fusion_gd)

        return fusion_gd_d, k0_flow, k8_flow


class sample_recon_nonkey_map(nn.Module):
    def __init__(self, num_filters, B_size, device):
        super(sample_recon_nonkey_map, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)

        self.denoisingnet = denoising_ini()

        self.map_block0 = basic_block_ini()
        self.map_block1 = basic_block()
        self.map_block2 = basic_block()
        self.map_block3 = basic_block()
        self.map_block4 = basic_block()
        self.map_block5 = basic_block()
        self.map_block6 = basic_block()
        self.map_block7 = basic_block()
        self.map_block8 = basic_block()
        self.map_block9 = basic_block()
        self.map_block10 = basic_block()
        self.map_block11 = basic_block()
        self.map_block12 = basic_block()

        self.final_fusionnet = final_fusion()

        self.num_filters = num_filters
        self.B_size = B_size
        self.t2image = nn.PixelShuffle(B_size)

        self.FlowBackWarp = backWarp_MH(device)

    def forward(self, x_ini, x_key1, x_key2):
        sample_w = self.sample.weight
        sample_w = torch.reshape(sample_w, (self.num_filters, (self.B_size * self.B_size)))
        sample_w_t = sample_w.t()

        sample_w_t = torch.unsqueeze(sample_w_t, 2)
        t_mat = torch.unsqueeze(sample_w_t, 3)

        phi_x = self.sample(x_ini)
        zk = F.conv2d(phi_x, t_mat, stride=1, padding=0)
        zk = self.t2image(zk)
        xoutput = self.denoisingnet(zk)

        xoutput, flow0, flow8 = self.map_block0(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x)
        xoutput, flow0, flow8 = self.map_block1(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block2(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block3(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block4(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block5(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block6(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block7(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block8(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block9(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                flow0, flow8)
        xoutput, flow0, flow8 = self.map_block10(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                 flow0, flow8)
        xoutput, flow0, flow8 = self.map_block11(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                 flow0, flow8)
        xoutput, flow0, flow8 = self.map_block12(xoutput, x_key1, x_key2, self.FlowBackWarp, self.sample, t_mat, phi_x,
                                                 flow0, flow8)

        xoutput = self.final_fusionnet(xoutput)

        return xoutput

# hsganet for key frame reconstruction
class sample_and_inirecon(nn.Module):
    def __init__(self, num_filters1, num_filters2, B_size):
        super(sample_and_inirecon, self).__init__()
        self.sample1 = nn.Conv2d(in_channels=1, out_channels=num_filters1, kernel_size=B_size, stride=B_size, padding=0,
                                 bias=False)
        self.sample2 = nn.Conv2d(in_channels=1, out_channels=num_filters2, kernel_size=B_size, stride=B_size, padding=0,
                                 bias=False)

        self.B_size = B_size
        self.t2image = nn.PixelShuffle(B_size)
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2

    def forward(self, x_ini, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag):
        sample_w1 = self.sample1.weight
        sample_w1 = torch.reshape(sample_w1, (self.num_filters1, (self.B_size * self.B_size)))
        sample_w_t1 = sample_w1.t()
        sample_w_t1 = torch.unsqueeze(sample_w_t1, 2)
        t_mat1 = torch.unsqueeze(sample_w_t1, 3)

        sample_w2 = self.sample2.weight
        sample_w2 = torch.reshape(sample_w2, (self.num_filters2, (self.B_size * self.B_size)))
        sample_w_t2 = sample_w2.t()
        sample_w_t2 = torch.unsqueeze(sample_w_t2, 2)
        t_mat2 = torch.unsqueeze(sample_w_t2, 3)
        if flag:
            x1 = torch.matmul(x_ini, rand_sw_p1)
            x1 = torch.matmul(rand_sw_p2, x1)
            x1 = self.sample1(x1)
            x1 = y1 - x1
            x1 = F.conv2d(x1, t_mat1, stride=1, padding=0)
            x1 = self.t2image(x1)
            x1 = torch.matmul(rand_sw_p2_t, x1)
            x1 = torch.matmul(x1, rand_sw_p1_t)

            x2 = self.sample2(x_ini)
            x2 = y2 - x2
            x2 = F.conv2d(x2, t_mat2, stride=1, padding=0)
            x2 = self.t2image(x2)
            return x1, x2
        else:
            x1 = torch.matmul(x_ini, rand_sw_p1)
            x1 = torch.matmul(rand_sw_p2, x1)
            phi_x1 = self.sample1(x1)
            x1 = F.conv2d(phi_x1, t_mat1, stride=1, padding=0)
            x1 = self.t2image(x1)
            x1 = torch.matmul(rand_sw_p2_t, x1)
            x1 = torch.matmul(x1, rand_sw_p1_t)

            phi_x2 = self.sample2(x_ini)
            x2 = F.conv2d(phi_x2, t_mat2, stride=1, padding=0)
            x2 = self.t2image(x2)
            return x1, phi_x1, x2, phi_x2


class basic_fusion_ini_key(nn.Module):
    def __init__(self, in_filters):
        super(basic_fusion_ini_key, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_filters, out_channels=32, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_ini):
        # x_ini = torch.cat((x1, x2), dim=1)
        x1 = self.conv1_1(x_ini)
        x1 = F.relu(x1, inplace=True)
        x1 = self.conv1_2(x1)
        x2 = self.conv2(x_ini)
        x3 = self.conv3(x_ini)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv4(x)
        # x = F.relu(x, inplace=True)
        return x


class fusion_ini_key(nn.Module):
    def __init__(self):
        super(fusion_ini_key, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.basic_fusion1 = basic_fusion_ini_key(in_filters=32)
        self.basic_fusion2 = basic_fusion_ini_key(in_filters=32)
        self.basic_fusion3 = basic_fusion_ini_key(in_filters=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x_in = F.relu(x, inplace=True)
        x = self.basic_fusion1(x_in)
        x = x + x_in
        x_in = F.relu(x, inplace=True)
        x = self.basic_fusion2(x_in)
        x = x + x_in
        x_in = F.relu(x, inplace=True)
        x = self.basic_fusion3(x_in)
        x = x + x_in
        x = F.relu(x, inplace=True)
        x = self.conv2(x)

        return x


class denoising_small_hsganet(nn.Module):
    def __init__(self):
        super(denoising_small_hsganet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_ini):
        # x_ini = x
        x = self.conv1(x_ini)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = x_ini + x

        return x


class softmax_gate_net_small(nn.Module):
    def __init__(self):
        super(softmax_gate_net_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        # x = F.relu(x, inplace=True)
        # x = self.conv5(x)
        output = F.softmax(x, dim=1)
        return output


class basic_block_hsganet(nn.Module):
    def __init__(self):
        super(basic_block_hsganet, self).__init__()
        self.softmax_gate_net = softmax_gate_net_small()
        self.denoising_key = denoising_small_hsganet()

    def forward(self, sample_and_inirecon, x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
        x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
        prob = self.softmax_gate_net(x1, x2)
        x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
        x = self.denoising_key(x)
        return x


class hsganet(nn.Module):
    def __init__(self, num_filters1, num_filters2, B_size):
        super(hsganet, self).__init__()
        self.sample_and_inirecon = sample_and_inirecon(num_filters1, num_filters2, B_size)
        self.fusion_ini = fusion_ini_key()
        self.denoising = denoising_small_hsganet()
        self.basic_block_key1 = basic_block_hsganet()
        self.basic_block_key2 = basic_block_hsganet()
        self.basic_block_key3 = basic_block_hsganet()
        self.basic_block_key4 = basic_block_hsganet()
        self.basic_block_key5 = basic_block_hsganet()
        self.basic_block_key6 = basic_block_hsganet()
        self.basic_block_key7 = basic_block_hsganet()
        self.basic_block_key8 = basic_block_hsganet()
        self.basic_block_key9 = basic_block_hsganet()
        self.basic_block_key10 = basic_block_hsganet()
        self.basic_block_key11 = basic_block_hsganet()
        self.basic_block_key12 = basic_block_hsganet()

    def forward(self, inputs, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
        ini_x1, phi_x1, ini_x2, phi_x2 = self.sample_and_inirecon(inputs, 0, 0, rand_sw_p1, rand_sw_p1_t, rand_sw_p2,
                                                                  rand_sw_p2_t, False)

        ini_x = self.fusion_ini(ini_x1, ini_x2)
        ini_x = self.denoising(ini_x)

        ini_x = self.basic_block_key1(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key2(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key3(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key4(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key5(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key6(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key7(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key8(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key9(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                      rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key10(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                       rand_sw_p2, rand_sw_p2_t)
        ini_x = self.basic_block_key11(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t,
                                       rand_sw_p2, rand_sw_p2_t)
        x_output = self.basic_block_key12(self.sample_and_inirecon, ini_x, phi_x1, phi_x2, True, rand_sw_p1,
                                          rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)

        return x_output
