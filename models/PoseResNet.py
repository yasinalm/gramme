# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .resnet_encoder import *


class PoseDecoder(nn.Module):
    def __init__(self, dataset, is_vo, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.is_vo = is_vo

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(
            num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 256, 3, stride, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.25)

        # num_features = 8192 if dataset=='hand' else 16384 #cart-16384 pol-90624 hand-8192
        # num_features = 16384  # cart-16384 pol-90624 hand-8192
        num_features = 3840 if is_vo else 16384  # vo-16384 ro-16384

        self.fc_t1 = nn.Linear(num_features, 128)
        if self.is_vo:
            self.fc_t2 = nn.Linear(
                128, 3 * num_frames_to_predict_for)  # [x,y,z]
        else:
            self.fc_t2 = nn.Linear(128, 2 * num_frames_to_predict_for)  # [x,y]
        self.fc_r1 = nn.Linear(num_features, 128)
        if self.is_vo:
            self.fc_r2 = nn.Linear(
                128, 3 * num_frames_to_predict_for)  # [rx,ry,rz]
        else:
            self.fc_r2 = nn.Linear(128, 1 * num_frames_to_predict_for)  # [rz]

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f))
                        for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        x = cat_features
        for i in range(3):
            x = self.convs[("pose", i)](x)
            if i != 2:
                x = self.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)

        # print(x.shape)  # to calculate linear layer input size

        # Translation: Pass data through fc1
        t = self.fc_t1(x)
        # t = F.relu(t)
        t = self.dropout1(t)
        t = self.fc_t2(t)

        # Translation: Pass data through fc1
        r = self.fc_r1(x)
        # r = F.relu(r)
        r = self.dropout1(r)
        r = self.fc_r2(r)

        # out = out.mean(3).mean(2)

        # pose = 0.01 * out.view(-1, 6)

        # t = t.clamp(-30, 30)  # [-v_max, +v_max] m/s
        # r = r.clamp(-6.28, 6.28) # [-2pi, 2pi] rad
        r = torch.fmod(r, 2*np.pi)

        # # SE(2) pose
        if self.is_vo:
            # pose = torch.zeros(r.shape[0], 6).to(r.device)
            # pose[:, 1:2] = r  # [ry]
            # pose[:, [3, 5]] = t  # [tx,tz]
            pose = torch.cat((r, t), 1)  # [B, 6]
        else:
            pose = torch.zeros(r.shape[0], 6).to(r.device)
            pose[:, 2:3] = r
            pose[:, 3:5] = t
        # pose = torch.cat((r, t), 1)  # [B, 6]

        return pose


class PoseResNet(nn.Module):

    def __init__(self, dataset, num_layers=18, pretrained=False, is_vo=False):
        super(PoseResNet, self).__init__()
        n_img_channels = 3 if is_vo else 1  # n_channels = vo-3 ro-1
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=2, n_img_channels=n_img_channels)
        self.decoder = PoseDecoder(dataset, is_vo, self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], 1)
        features = self.encoder(x)
        pose = self.decoder([features])
        return pose

# if __name__ == "__main__":

#     torch.backends.cudnn.benchmark = True

#     model = PoseResNet().cuda()
#     model.train()

#     tgt_img = torch.randn(4, 1, 256, 64).cuda()
#     ref_imgs = [torch.randn(4, 1, 256, 64).cuda() for i in range(2)]

#     pose = model(tgt_img, ref_imgs[0])

#     print(pose.size())
