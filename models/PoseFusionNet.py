from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
# from inverse_warp_vo2 import euler2mat, pose_vec2mat
# from conversions import rotation_matrix_to_angle_axis
import conversions as tgm


class PoseFusionNet(nn.Module):

    def __init__(self):
        super(PoseFusionNet, self).__init__()
        # self.fc_ro = nn.Linear(12, 6)
        # self.fc_vo = nn.Linear(6, 6)
        # Regressor for the 3 * 2 affine matrix
        self.fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # self.R = torch.randn(3, requires_grad=True)
        # self.R = euler2mat(self.R)  # [3, 3]
        # self.s = torch.randn(1, requires_grad=True)

        # self.T = torch.eye(4, dtype=torch.float, requires_grad=True)
        # Initialize the weights/bias with identity transformation
        # self.T.weight.data.zero_()
        # self.T.bias.data.copy_(torch.eye(4, dtype=torch.float).view(-1))

    def forward(self, vo):
        # x = torch.cat([ro, vo], -1)
        # ro_pose = self.fc_ro(x)
        vo_pose = self.fc(vo)

        # pose_mat = pose_vec2mat(vo)  # [B,3,4]
        # pose_mat = self.R @ pose_mat  # [B,3,4]
        # pose_mat *= self.s
        # vo_pose = rotation_matrix_to_angle_axis(pose_mat)

        # print(vo.shape)
        # pose_mat = tgm.rtvec_to_pose(vo)  # [B,6] -> [B,4,4]
        # pose_mat = self.T @ pose_mat
        # t = pose_mat[..., :3]
        # r = tgm.rotation_matrix_to_angle_axis(pose_mat[:, :3, :])
        # pose = torch.cat((r, t), -1)

        return vo_pose
