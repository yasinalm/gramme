from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import warnings

import models
import utils

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import custom_transforms
from datasets.sequence_folders_lidar import SequenceFolder
from inverse_warp_radar import Warper

# Supress UserWarning from grid_sample
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Script for testing depth predictions with the corresponding ground truth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N',
                    help='sequence length for training', default=3)
parser.add_argument('--skip-frames', type=int, metavar='N',
                    help='gap between frames', default=1)
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='hand', help='the dataset to train')
parser.add_argument('--with-preprocessed', type=int, default=1,
                    help='use the preprocessed undistorted images')
parser.add_argument('--with-testfile', type=int, default=0,
                    help='use the test.txt file containing test sequences')
parser.add_argument("--nsamples", default=2000, type=int,
                    help="Number of samples to subsample from each scene.")
parser.add_argument('--with-ssim', type=int,
                    default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1,
                    help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=1,
                    help='with the the mask for stationary points')
parser.add_argument('--with-masknet', action='store_true',
                    help='with the masknet for multipath and noise')
parser.add_argument('--masknet', type=str,
                    choices=['convnet', 'resnet'], default='convnet', help='MaskNet type')
parser.add_argument('--with-timing', type=int, default=0,
                    help='use the timing benchmark to evaluate the runtime speed')
parser.add_argument('--with-vo', action='store_true',
                    help='with VO fusion')
parser.add_argument('--cam-mode', type=str, choices=[
                    'mono', 'stereo'], default='stereo', help='the dataset to train')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', required=True,
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('-j', '--workers', default=4, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument('--num-scales', '--number-of-scales',
                    type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument("--img-height", default=192, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping')
# parser.add_argument("--min-depth", default=1e-3)
# parser.add_argument("--max-depth", default=80)
# parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
# parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
# parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--results-dir', default='results', metavar='PATH',
                    help='directory where to save predicted depth maps and stats')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation')
parser.add_argument('--cart-res', type=float,
                    help='Cartesian resolution of LIDAR in meters/pixel', metavar='W', default=0.25)
parser.add_argument('--cart-pixels', type=int,
                    help='Cartesian size in pixels (used for both height and width)', metavar='W', default=512)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()

    results_dir = Path(args.results_dir)  # /args.sequence
    results_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    img_size = (args.img_height, args.img_width)

    # Calculate center of handheld dataset. The rotation center is not the image center (default image center).
    if args.dataset == 'hand':
        center = (0, args.cart_pixels//2)
        train_transform = custom_transforms.Compose(
            [custom_transforms.ArrayToTensor(),
             transforms.RandomRotation(degrees=10, center=center)])
    else:
        if args.with_vo:
            train_transform = custom_transforms.Compose(
                [custom_transforms.ArrayToTensor(),
                 # transforms.RandomRotation(10)
                 ])
        else:
            train_transform = custom_transforms.Compose(
                [custom_transforms.ArrayToTensor(),
                 transforms.RandomRotation(10)
                 ])
    val_transform = custom_transforms.Compose(
        [custom_transforms.ArrayToTensor()])

    if args.with_testfile:
        root = Path(args.data)
        scene_list_path = root/'test.txt'
        scenes = [root/folder.strip()/'stereo_undistorted/left'
                  for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        scene_names = [folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
    else:
        scene_names = 'sequence'
        scenes = [Path(args.data)]

    cam_valid_transform = None
    lo_params = {
        'cart_pixels': args.cart_pixels,
    }

    # create warper
    print("=> creating loss object")
    warper = Warper(args.with_auto_mask, args.cart_res,
                    args.cart_pixels, args.dataset, args.padding_mode)

    # create model
    print("=> creating model")
    lidar_pose_net = models.PoseResNet(
        args.dataset, args.resnet_layers, False).to(device)
    mask_net = None
    if args.with_masknet:
        if args.masknet == 'resnet':
            mask_net = models.MaskResNet(
                args.resnet_layers, False).to(device)
        elif args.masknet == 'convnet':
            mask_net = models.MaskNet(num_channels=1).to(device)
        else:
            raise NotImplementedError(
                'The chosen MaskNet is not implemented! Given: {}'.format(args.masknet))

    # load parameters
    print("=> using pre-trained weights for PoseNet")
    weights = torch.load(args.pretrained_pose)
    lidar_pose_net.load_state_dict(weights['state_dict'], strict=False)

    # load parameters
    if args.with_masknet:
        print("=> using pre-trained weights for MaskNet")
        weights = torch.load(args.pretrained_mask)
        mask_net.load_state_dict(weights['state_dict'], strict=False)

    lidar_pose_net = torch.nn.DataParallel(lidar_pose_net)

    # switch to evaluate mode
    lidar_pose_net.eval()
    # switch to train mode
    if args.with_masknet:
        mask_net = torch.nn.DataParallel(mask_net)
        mask_net.eval()

    for scene_name, scene in zip(scene_names, scenes):
        print("=> Processing:", scene_name)

        results_mask_dir = results_dir/scene_name/'mask'
        results_mask_dir.mkdir(parents=True)

        test_set = SequenceFolder(
            args.data,
            transform=val_transform,
            seed=args.seed,
            # train=False,
            mode='test',
            sequence_length=args.sequence_length,
            sequence=scene_name,
            skip_frames=args.skip_frames,
            dataset=args.dataset,
            lo_params=lo_params,
            load_camera=args.with_vo,
            cam_mode=args.cam_mode,
            cam_transform=cam_valid_transform,
            cam_preprocessed=args.with_preprocessed,
            nsamples=args.nsamples
        )
        nframes = len(test_set)
        print('{} samples found in {} '.format(
            nframes, scene_name))
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

        avg_time = 0
        for i, (tgt_img, ref_imgs, f_names) in tqdm(enumerate(test_loader)):
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            if args.with_timing:
                # compute speed
                torch.cuda.synchronize()
                t_start = time.time()

            # compute output
            tgt_mask, ref_masks = None, [
                None for i in range(args.sequence_length-1)]
            if args.with_masknet:
                tgt_mask, ref_masks = compute_mask(mask_net, tgt_img, ref_imgs)
            ro_poses, ro_poses_inv = compute_pose_with_inv(
                lidar_pose_net, tgt_img, ref_imgs)

            (rec_loss, geometry_consistency_loss, fft_loss, ssim_loss,
             projected_imgs, projected_masks) = warper.compute_db_loss(
                tgt_img, ref_imgs, tgt_mask, ref_masks, ro_poses, ro_poses_inv)

            # print(len(projected_masks))
            # print(projected_masks[0].shape)
            # print(f_names.shape)
            if args.with_timing:
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                avg_time += elapsed_time

            # tv.utils.save_image(
            #     tgt_depth[0], results_depth_dir/'{0:03d}.png'.format(i))

            for j, (mask, f_name) in enumerate(zip(projected_masks[0], f_names)):
                colour_mask = utils.tensor2array(
                    mask, max_value=1.0, colormap='bone')
                colour_mask = colour_mask.transpose(1, 2, 0)*255
                colour_mask = colour_mask.astype(np.uint8)
                im = Image.fromarray(colour_mask)
                # im.save(results_mask_dir /
                #         '{0:03d}.png'.format(i*args.batch_size+j))
                im.save(results_mask_dir / (f_name+'.png'))

        if args.with_timing:
            avg_time /= nframes
            print('Avg Time: ', avg_time, ' seconds.')
            print('Avg Speed: ', 1.0 / avg_time, ' fps')


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


def compute_mask(mask_net, tgt_img, ref_imgs):
    # tgt_mask = [mask for mask in mask_net(tgt_img)] # for multiple scale
    tgt_mask = mask_net(tgt_img)  # for single scale

    ref_masks = []
    for ref_img in ref_imgs:
        # ref_mask = [mask for mask in mask_net(ref_img)] # for multiple scale
        ref_mask = mask_net(ref_img)  # for multiple scale
        ref_masks.append(ref_mask)

    return tgt_mask, ref_masks


if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=False):
        main()
