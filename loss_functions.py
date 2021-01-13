from __future__ import division
import torch
from torch import nn
from inverse_warp import inverse_warp_fft
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#TODO: num_scales, mask eklenebilir buraya.
# decibels loss
def compute_db_loss(tgt_img, ref_imgs, poses, poses_inv, padding_mode):

    db_loss = 0

    for ref_img, pose, pose_inv in zip(ref_imgs, poses, poses_inv):

        db_loss1 = compute_pairwise_loss(tgt_img, ref_img, pose, padding_mode)
        db_loss2 = compute_pairwise_loss(ref_img, tgt_img, pose_inv, padding_mode)

        db_loss += (db_loss1 + db_loss2)

    return db_loss


def compute_pairwise_loss(tgt_img, ref_img, pose, padding_mode):

    ref_img_warped = inverse_warp_fft(ref_img, pose, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    # compute all loss
    reconstruction_loss = diff_img.sum()

    return reconstruction_loss





