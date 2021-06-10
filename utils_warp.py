from __future__ import division
import torch
import numpy as np


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth[0], tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth[0], ref_img)

    return loss


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return phi, rho


def pol2cart(phi, rho):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def camera_matrix(pinholes, eps=1e-6):
    k = torch.eye(4, device=pinholes.device, dtype=pinholes.dtype) + eps
    # k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4
    # fill output with pinhole values
    k[..., 0, 0] = pinholes[0]  # fx
    k[..., 0, 2] = pinholes[1]  # cx
    k[..., 1, 1] = pinholes[2]  # fy
    k[..., 1, 2] = pinholes[3]  # cy
    return k


def get_intrinsics_matrix(dataset):
    intrinsics = np.eye(3, dtype=np.float32)+1e-6
    if dataset == 'robotcar':
        fx, fy, cx, cy = 983.044006, 983.044006, 643.646973, 493.378998
    elif dataset == 'radiate':
        fx = 3.379191448899105e+02
        fy = 3.386957068549526e+02
        cx = 3.417366010946575e+02
        cy = 2.007359735313929e+02
    else:
        raise NotImplementedError(
            'The chosen dataset is not implemented! Given: {}'.format(dataset))

    intrinsics[0, 0] = fx
    intrinsics[0, 2] = cx
    intrinsics[1, 1] = fy
    intrinsics[1, 2] = cy

    return intrinsics


def get_rightTleft(dataset):
    rightTleft = np.zeros(6, dtype=np.float32)
    if dataset == 'robotcar':
        tx = -0.239983
    elif dataset == 'radiate':
        tx = 0.0
    else:
        raise NotImplementedError(
            'The chosen dataset is not implemented! Given: {}'.format(dataset))

    rightTleft[0] = tx
    return rightTleft
