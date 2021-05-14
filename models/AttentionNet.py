from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()
        # self.fc_ro = nn.Linear(12, 6)
        self.fc_vo = nn.Linear(6, 6)

    def forward(self, vo):
        # x = torch.cat([ro, vo], -1)
        # ro_pose = self.fc_ro(x)
        vo_pose = self.fc_vo(vo)
        return vo_pose
