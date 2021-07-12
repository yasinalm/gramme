# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from .resnet_encoder import *


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = nn.ModuleDict() # OrderedDict()
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs["pose{}".format(0)] = nn.Conv2d(
            num_input_features * 256, 256, 3, stride, 1)
        self.convs["pose{}".format(1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs["pose{}".format(2)] = nn.Conv2d(
            256, 128, 1)

        self.conv_trans = nn.Conv2d(
            128, 3 * num_frames_to_predict_for, 1)
        self.conv_rot = nn.Conv2d(
            128, 3 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f))
                        for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs["pose{}".format(i)](out)
            # if i != 2:
            out = self.relu(out)
        
        r = self.conv_rot(out)
        t = self.conv_trans(out)
        r = 0.1 * r.mean(3).mean(2).view(-1, 3)
        t = 0.01 * t.mean(3).mean(2).view(-1, 3)
        pose = torch.cat((r, t), 1)  # [B, 6]

        # out = out.mean(3).mean(2)
        # pose = 0.1 * out.view(-1, 6)

        return pose


class PoseResNetStereo(nn.Module):

    def __init__(self, num_layers=18, pretrained=True):
        super(PoseResNetStereo, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=2, n_img_channels=3)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)

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
