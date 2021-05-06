from __future__ import division
import PIL
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sn

# import shutil
import numpy as np
import torch
# from pathlib import Path
# import datetime
# from collections import OrderedDict
import matplotlib as mpl
mpl.use('Agg')  # No x-server

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Seaborn settings
# sns.set(style=\"whitegrid\", rc={\"font.size\":8,\"axes.titlesize\":8,\"axes.labelsize\":5})
sn.set(style="whitegrid", font_scale=1.5)
sn.set_palette("bright", n_colors=4, color_codes=True)

# def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
#     # Construct the list colormap, with interpolated values for higer resolution
#     # For a linear segmented colormap, you can just specify the number of point in
#     # cm.get_cmap(name, lutsize) with the parameter lutsize
#     x = np.linspace(0, 1, low_res_cmap.N)
#     low_res = low_res_cmap(x)
#     new_x = np.linspace(0, max_value, resolution)
#     high_res = np.stack([np.interp(new_x, x, low_res[:, i])
#                          for i in range(low_res.shape[1])], axis=1)
#     return ListedColormap(high_res)


# def opencv_rainbow(resolution=1000):
#     # Construct the opencv equivalent of Rainbow
#     opencv_rainbow_data = (
#         (0.000, (1.00, 0.00, 0.00)),
#         (0.400, (1.00, 1.00, 0.00)),
#         (0.600, (0.00, 1.00, 0.00)),
#         (0.800, (0.00, 0.00, 1.00)),
#         (1.000, (0.60, 0.00, 1.00))
#     )

#     return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


# COLORMAPS = {'rainbow': opencv_rainbow(),
#              'magma': high_res_colormap(cm.get_cmap('magma')),
#              'bone': cm.get_cmap('bone', 1000)}

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
imagenet_std = torch.Tensor([0.229, 0.224, 0.225])

# imagenet_mean_grey = torch.Tensor([0.445])
# imagenet_std_grey = torch.Tensor([0.269])


def tensor2array(tensor, max_value=None, colormap='magma'):
    #tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.abs().max()  # .item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        # if is_grey:  # grey image
        #     array = imagenet_mean_grey[:, None, None].to(tensor.device) + \
        #         tensor*imagenet_std_grey[:, None, None].to(tensor.device)
        #     array = array.detach().cpu().numpy()
        # else:  # Depth image
        # norm_array = tensor.squeeze().numpy()/max_value
        norm_array = tensor.squeeze()/max_value
        norm_array = norm_array.detach().cpu().numpy()
        # array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = cm.get_cmap(colormap)(norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        # array = 119.4501 + tensor.numpy()*6.5258
        array = imagenet_mean[:, None, None].to(tensor.device) + \
            tensor*imagenet_std[:, None, None].to(tensor.device)
        array = array.detach().cpu().numpy()
    return array


def traj2Img(pred_xyz):
    """Make an image from the 2D plot of a given trajectory.

    Args:
        pred_xyz (torch.Tensor): Trajectory to plot. Shape: [N,3]

    Returns:
        PIL.Image: Image of the trajectory plot
    """

    pred_xyz = pred_xyz.detach().cpu()
    pred_x = pred_xyz[:, 0].numpy()
    pred_y = pred_xyz[:, 1].numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pred_x, pred_y)
    fig.canvas.draw()

    img_plt = PIL.Image.frombytes(
        'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    img_plt = np.asarray(img_plt)  # CHW
    img_plt = img_plt.transpose(2, 0, 1)  # HWC
    return img_plt


def traj2Fig(pred_xyz):
    """Make `matplotlib.pyplot.figure` from the 2D plot of a given trajectory.

    Args:
        pred_xyz (torch.Tensor): Trajectory to plot. Shape: [N,3]

    Returns:
        matplotlib.pyplot.figure: Figure of the trajectory plot
    """

    pred_xyz = pred_xyz.cpu()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1])
    # fig.canvas.draw()

    return fig


def traj2Fig_withgt(pred_xyz, gt_xyz):
    """Make `matplotlib.pyplot.figure` from the 2D plot of a given trajectory.

    Args:
        pred_xyz (torch.Tensor): Trajectory to plot. Shape: [N,3]
        gt_xyz (torch.Tensor): Trajectory to plot. Shape: [N,3]

    Returns:
        matplotlib.pyplot.figure: Figure of the trajectory plot
    """

    pred_xyz = pred_xyz.cpu()
    gt_xyz = gt_xyz.cpu()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], label='Prediction')
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], label='Ground-truth')
    ax.legend()
    # fig.canvas.draw()

    return fig


def save_traj_plots(results_dir, f_pred_xyz, b_pred_xyz):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    ax = sn.lineplot(x=f_pred_xyz[:, 0].cpu().numpy(
    ), y=f_pred_xyz[:, 1].cpu().numpy(), sort=False)
    ax.set(title='Forward Trajectory', xlabel='X (m)', ylabel='Y (m)')
    plt.subplot(1, 2, 2)
    ax = sn.lineplot(x=b_pred_xyz[:, 0].cpu().numpy(
    ), y=b_pred_xyz[:, 1].cpu().numpy(), sort=False, markers=True)
    ax.set(title='Backward Trajectory', xlabel='X (m)', ylabel='Y (m)')

    # Save fig
    plt.tight_layout()
    plt.savefig(results_dir/'ro_pred_nogt.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.savefig(results_dir/'ro_pred_nogt.png',
                bbox_inches='tight', pad_inches=0)


def save_traj_plots_with_gt(results_dir, pred_xyz, gt):
    gt_xyz = gt[:, :3, 3].cpu().numpy()
    # np_pred = 0.5*gt_xyz + 0.5*pred_xyz[0].cpu().numpy()
    np_pred = pred_xyz[0].cpu()
    fig, ax = plt.subplots(figsize=(8, 8))
    sn.lineplot(x=np_pred[:, 0], y=np_pred[:, 1],
                sort=False, ax=ax, label='Ours')
    sn.lineplot(x=gt_xyz[:, 0], y=gt_xyz[:, 1],
                sort=False, ax=ax, label='Ground-truth')

    ax.set(xlabel='X (m)', ylabel='Y (m)')

    # Save fig
    plt.tight_layout()
    plt.savefig(results_dir/'ro_pred_with_gt.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.savefig(results_dir/'ro_pred_with_gt.png',
                bbox_inches='tight', pad_inches=0)


def save_checkpoint(save_path, masknet_state, posenet_state, epoch='', filename='checkpoint.pth.tar'):
    file_prefixes = ['posenet']
    states = [posenet_state]
    if masknet_state is not None:
        file_prefixes.append('masknet')
        states.append(masknet_state)
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}{}'.format(prefix, epoch, filename))

    # if is_best:
    #     for prefix in file_prefixes:
    #         shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
    #                         save_path/'{}_best.pth.tar'.format(prefix))


def save_checkpoint_mono(save_path, dispnet_state, exp_pose_state, step, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}{}'.format(prefix, step, filename))

    # if is_best:
    #     for prefix in file_prefixes:
    #         shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
    #                         save_path/'{}_model_best.pth.tar'.format(prefix))
