"""
Mask R-CNN
The main Mask R-CNN model implemenetation.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


############################################################
#  MaskRCNN Class
############################################################

# class MaskRCNN(nn.Module):
#     """Encapsulates the Mask RCNN model functionality.
#     """

#     def __init__(self):
#         """
#         config: A Sub-class of the Config class
#         model_dir: Directory to save training logs and trained weights
#         """
#         super(MaskRCNN, self).__init__()
#         self.initialize_weights()

#     def build(self):
#         """Build Mask R-CNN architecture.
#         """

#         # # Image size must be divisible by 2 multiple times
#         # h, w = config.IMAGE_SHAPE[:2]
#         # if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
#         #     raise Exception("Image size must be divisible by 64 "
#         #                     "to avoid fractions when downscaling and upscaling."
#         #                     "For example, use 256, 320, 384, 448, 512, ... etc. ")

#         # Mask
#         self.mask = Mask(num_output_channels=1)

#         # # Fix batch norm layers
#         # def set_bn_fix(m):
#         #     classname = m.__class__.__name__
#         #     if classname.find('BatchNorm') != -1:
#         #         for p in m.parameters():
#         #             p.requires_grad = False

#         # # Set batchnorm always in eval mode during training
#         # def set_bn_eval(m):
#         #     classname = m.__class__.__name__
#         #     if classname.find('BatchNorm') != -1:
#         #         m.eval()

#         # self.apply(set_bn_fix)

#     def initialize_weights(self):
#         """Initialize model weights.
#         """

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.mask(x)
#         return x

############################################################
#  Feature Pyramid Network Heads
############################################################


class MaskNet(nn.Module):
    def __init__(self, num_channels):
        super(MaskNet, self).__init__()
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(128, eps=0.001)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32, eps=0.001)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16, eps=0.001)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(8, eps=0.001)
        # self.deconv = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(8, num_channels,
                               kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        # x = self.deconv(x)
        # x = self.relu(x)
        x = self.conv5(self.padding(x))
        x = self.sigmoid(x)

        return x


############################################################
#  Pytorch Utility Functions
############################################################


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
