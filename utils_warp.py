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
        mean_disp = torch.clamp(mean_disp, min=1e-7)
        norm_disp = disp / mean_disp
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
    """ Returns the intrinsic matrix as a tensor.

    Args:
        pinholes (list): List of fx, cx, fy, cy camera parameters.
        eps (float, optional): A small number for computational stability. Defaults to 1e-6.

    Returns:
        torch.Tensor: Intrinsic matrix as a [4,4] Tensor.
    """
    k = torch.eye(4, device=pinholes.device, dtype=pinholes.dtype) + eps
    # k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4
    # fill output with pinhole values
    k[..., 0, 0] = pinholes[0]  # fx
    k[..., 0, 2] = pinholes[1]  # cx
    k[..., 1, 1] = pinholes[2]  # fy
    k[..., 1, 2] = pinholes[3]  # cy
    return k


def get_intrinsics_matrix(dataset, preprocessed, cam='left'):
    """ Return the intrinsic matrix of a given dataset. If the dataset is preprocessed, returns the adjusted matrix.

    Args:
        dataset (str): Name of the dataset
        preprocessed (bool): Whether the dataset is preprocessed

    Raises:
        NotImplementedError: The dataset should be one of the supported datasets.

    Returns:
        np.ndarray: Intrinsic matrix in the form of [3,3] matrix.
    """
    intrinsics = np.eye(3, dtype=np.float32)+1e-6
    if dataset == 'robotcar':
        # Left and right cameras have the same params
        fx, fy, cx, cy = 983.044006, 983.044006, 643.646973, 493.378998
        if preprocessed:
            scaled_h, scaled_w = 192, 320
            in_h, in_w = 800, 1280
            x_scaling = float(scaled_w)/in_w
            y_scaling = float(scaled_h)/in_h

            fx *= x_scaling
            cx *= x_scaling
            fy *= y_scaling
            cy *= y_scaling
    elif dataset == 'radiate':
        if cam == 'left':
            fx = 337.873451599077
            fy = 338.530902554779
            cx = 329.137695760749
            cy = 186.166590759716
        else:
            fx = 3.379191448899105e+02
            fy = 3.386957068549526e+02
            cx = 3.417366010946575e+02
            cy = 2.007359735313929e+02
        if preprocessed:
            scaled_h, scaled_w = 192, 320
            in_h, in_w = 376, 672
            x_scaling = float(scaled_w)/in_w
            y_scaling = float(scaled_h)/in_h

            fx *= x_scaling
            cx *= x_scaling
            fy *= y_scaling
            cy *= y_scaling
    elif dataset == 'cadcd':
        # fx, cx, fy, cy = 653.956033188809, 653.221172545916, 655.540088617960, 508.732863993917 # cam00
        if cam == 'left':
            fx, cx, fy, cy = 655.400620284058, 630.296420683747, 657.186232327181, 513.826153608948  # cam07
        else:
            fx, cx, fy, cy = 659.538950747990, 640.585315682482, 660.928189799613, 490.646003044410  # cam01
        if preprocessed:
            # Cropping
            offset_x = 140
            cx -= offset_x
            offset_y = 250  # has no effect on intrinsics

            # Resizing
            scaled_h, scaled_w = 192, 320
            in_h, in_w = 774, 1000
            x_scaling = float(scaled_w)/in_w
            y_scaling = float(scaled_h)/in_h

            fx *= x_scaling
            cx *= x_scaling
            fy *= y_scaling
            cy *= y_scaling
    else:
        raise NotImplementedError(
            'The chosen dataset is not implemented! Given: {}'.format(dataset))

    intrinsics[0, 0] = fx
    intrinsics[0, 2] = cx
    intrinsics[1, 1] = fy
    intrinsics[1, 2] = cy

    return intrinsics


def get_rightTleft(dataset):
    """ Return the stereo baseline, e.g., the distance between left and right cameras.

    Args:
        dataset (str): Name of the dataset

    Raises:
        NotImplementedError: The dataset should be one of the supported datasets.

    Returns:
        np.ndarray: Transformation values of length [6]: tx, ty, tz, rx, ry, rz
    """
    rightTleft = np.zeros(6, dtype=np.float32)
    if dataset == 'robotcar':
        rightTleft[0] = -0.239983
    elif dataset == 'radiate':
        rightTleft[0] = -0.12
    elif dataset == 'cadcd':
        rightTleft[0] = -1.027041266425808
    else:
        raise NotImplementedError(
            'The chosen dataset is not implemented! Given: {}'.format(dataset))

    # rightTleft[0] = tx
    return rightTleft
