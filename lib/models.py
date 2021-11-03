import numpy as np
import torch
import torch.nn as nn

import pdb

class UNet(nn.Module):
    def __init__(self,
                 blurry_channels=1,
                 event_channels=26,
                 keypoint_channels=10,
                 use_blurry=True,
                 use_events=True,
                 out_channels=10):
        super(UNet, self).__init__()
        in_channels = keypoint_channels
        if use_blurry:
            in_channels += blurry_channels
        if use_events:
            in_channels += event_channels
        self.use_blurry = use_blurry
        self.use_events = use_events
        self.en_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.en_conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.en_conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.en_conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.res1 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512)
                )
        self.relu1 = nn.ReLU(inplace=True)
        self.res2 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512)
                )
        self.relu2 = nn.ReLU(inplace=True)
        self.de_conv1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=(0, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.de_conv2 = nn.Sequential(
                nn.ConvTranspose2d(256 + 256, 128, kernel_size=5, stride=2, padding=2, output_padding=(0, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.de_conv3 = nn.Sequential(
                nn.ConvTranspose2d(128 + 128, 64, kernel_size=5, stride=2, padding=2, output_padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.de_conv4 = nn.Sequential(
                nn.ConvTranspose2d(64 + 64, 32, kernel_size=5, stride=2, padding=2, output_padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.pred = nn.Conv2d(32 + in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, blurry_frame, event_map, keypoints):
        # assemble input tensor
        x_in = []
        if self.use_blurry:
            x_in.append(blurry_frame)
        if self.use_events:
            x_in.append(event_map)
        x_in.append(keypoints)
        x_in = torch.cat(x_in, dim=1)
        # pass to the network
        x1 = self.en_conv1(x_in)
        x2 = self.en_conv2(x1)
        x3 = self.en_conv3(x2)
        x4 = self.en_conv4(x3)
        x5 = self.relu1(self.res1(x4) + x4)
        x6 = self.relu2(self.res2(x5) + x5)
        x7 = self.de_conv1(x6)
        x8 = self.de_conv2(torch.cat([x7, x3], dim=1))
        x9 = self.de_conv3(torch.cat([x8, x2], dim=1))
        x10 = self.de_conv4(torch.cat([x9, x1], dim=1))
        x_out = self.pred(torch.cat([x10, x_in], dim=1))
        return { 'derivative': x_out }

class RefineNet(nn.Module):
    def __init__(self,
                 event_channels=26,
                 refine_iters=7,
                 lambda_cons=1.,
                 alpha_multiplier=0.1):
        super(RefineNet, self).__init__()
        self.alpha_net = nn.Conv2d(1, 1, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        # To predict the first residual (R0 = I1 - I0), we assume R0 depends on
        # * I0_init and I1_init
        # * I0_init_x, I0_init_y, I1_init_x, I1_init_y
        # * E[0:2]
        self.first_res_net = nn.Sequential(
                nn.Conv2d(2 + 4 + 2, 32, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, 1, 1, bias=False)
                )
        # To predict other residuals (Ri = I{i+1} - Ii), we assume R0 depends on
        # * Ii_init and I{i+1}_init
        # * Ii_init_x, Ii_init_y, I{i+1}_init_x, I{i+1}_init_y

        # * E[2*i:2*i+2]
        # * R{i-1}
        self.other_res_net = nn.Sequential(
                nn.Conv2d(2 + 4 + 2 + 1, 32, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, 1, 1, bias=False)
                )
        self.final = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, 1, 1, bias=False)
                )
        self.refine_iters = refine_iters
        self.lambda_cons = lambda_cons
        self.alpha_multiplier = alpha_multiplier

    def compute_spatial_derivatives(self, I):
        # I: [bs, h, w]
        I_x = torch.cat([I[:, :, :1], I[:, :, :-1]], dim=2) - I
        I_y = torch.cat([I[:, :1, :], I[:, :-1, :]], dim=1) - I
        return I_x, I_y

    def forward(self, frame_init, event_map):
        frame_prev = frame_init
        n_frms = frame_prev.shape[1] # this should be 14
        residual = []
        for _ in range(self.refine_iters):
            # predict residuals
            I_curr = frame_prev[:, 0]
            I_x_curr, I_y_curr = self.compute_spatial_derivatives(I_curr)
            residual.append([])
            for i in range(n_frms - 1):
                # predict the residual Ri = R{i+1} - R{i}
                I_next = frame_prev[:, i + 1]
                I_x_next, I_y_next = self.compute_spatial_derivatives(I_next)
                if i == 0:
                    res_in = torch.stack([I_curr, I_next,
                                          I_x_curr, I_y_curr, I_x_next, I_y_next,
                                          event_map[:, 2 * i],
                                          event_map[:, 2 * i + 1]], dim=1)
                    residual[-1].append(self.first_res_net(res_in).squeeze(dim=1))
                else:
                    res_in = torch.stack([I_curr, I_next,
                                          I_x_curr, I_y_curr, I_x_next, I_y_next,
                                          event_map[:, 2 * i],
                                          event_map[:, 2 * i + 1],
                                          residual[-1][-1]], dim=1)
                    residual[-1].append(self.other_res_net(res_in).squeeze(dim=1))
                I_curr, I_x_curr, I_y_curr = I_next, I_x_next, I_y_next
            residual[-1] = torch.stack(residual[-1], dim=1)
            # simulate gradient descent step
            frame_refine = []
            for i in range(n_frms):
                # predict the step size
                alpha = self.relu(self.alpha_net(frame_prev[:, i:i + 1])).squeeze(dim=1)
                alpha = self.alpha_multiplier * alpha
                # compute the update direction
                direction = 2 * self.lambda_cons * \
                            (frame_prev[:, i] - frame_init[:, i])
                if i != 0:
                    direction = direction - \
                                2 * (frame_prev[:, i - 1] + residual[-1][:, i - 1] - frame_prev[:, i])
                if i != n_frms - 1:
                    direction = direction + \
                                2 * (frame_prev[:, i] + residual[-1][:, i] - frame_prev[:, i + 1])
                frame_step = frame_prev[:, i] - alpha * direction
                frame_refine.append(frame_step)
            frame_refine = torch.stack(frame_refine, dim=1)
            frame_prev = frame_refine
        frame_refine_final = []
        for i in range(n_frms):
            frame_refine_final.append(self.final(frame_refine[:, i:i+1]))
        frame_refine_final = torch.cat(frame_refine_final, dim=1)
        residual = torch.stack(residual, dim=1)
        return frame_refine_final, residual
