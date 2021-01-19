from __future__ import division
import torch
from torch import nn
from inverse_warp import inverse_warp_fft
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#TODO: num_scales, mask eklenebilir buraya.
# decibels loss
def compute_db_loss(tgt_img, ref_imgs, poses, poses_inv, with_auto_mask, padding_mode):

    db_loss = 0

    for ref_img, pose, pose_inv in zip(ref_imgs, poses, poses_inv):

        db_loss1 = compute_pairwise_loss(tgt_img, ref_img, pose, with_auto_mask, padding_mode)
        db_loss2 = compute_pairwise_loss(ref_img, tgt_img, pose_inv, with_auto_mask, padding_mode)

        db_loss += (db_loss1 + db_loss2)

    return db_loss


def compute_pairwise_loss(tgt_img, ref_img, pose, with_auto_mask, padding_mode):

    ref_img_warped, valid_mask = inverse_warp_fft(ref_img, pose, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)

    return reconstruction_loss

# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value



