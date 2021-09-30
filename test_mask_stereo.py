from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time

import models
import utils
# import custom_transforms_mono as T

import torch
import torch.backends.cudnn as cudnn
import torchvision as tv
import custom_transforms_stereo as T
# import torchvision.transforms as T
from datasets.sequence_folders_stereo import SequenceFolder
from inverse_warp_vo2 import MonoWarper

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
parser.add_argument('--with-timing', type=int, default=0,
                    help='use the timing benchmark to evaluate the runtime speed')
parser.add_argument("--pretrained-disp", required=True,
                    type=str, help="pretrained DispNet path")
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None,
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

    if args.dataset == 'robotcar':
        if args.with_preprocessed:
            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            valid_transform = T.Compose([
                T.ToPILImage(),
                T.CropBottom(),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

    elif args.dataset == 'radiate':
        if args.with_preprocessed:
            valid_transform = T.Compose([
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
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

    # create warper
    print("=> creating loss object")
    mono_warper = MonoWarper(
        max_scales=args.num_scales,
        dataset=args.dataset,
        with_auto_mask=args.with_auto_mask,
        with_mask=args.with_mask,
        with_ssim=args.with_ssim,
        padding_mode=args.padding_mode
    )

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(
        args.resnet_layers, False).to(device)
    pose_net = models.PoseResNetStereo(args.resnet_layers, False).to(device)

    # load parameters
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)

    print("=> using pre-trained weights for PoseResNet")
    weights = torch.load(args.pretrained_pose)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    for scene_name, scene in zip(scene_names, scenes):
        print("=> Processing:", scene_name)

        results_mask_dir = results_dir/scene_name/'mask'
        results_mask_dir.mkdir(parents=True)

        test_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            # train=False,
            mode='test',
            sequence_length=args.sequence_length,
            sequence=scene_name,
            skip_frames=args.skip_frames,
            preprocessed=args.with_preprocessed,
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
        for i, (tgt_img, ref_imgs, intrinsics, f_names) in tqdm(enumerate(test_loader)):
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]
            intrinsics = [i.to(device) for i in intrinsics]

            if args.with_timing:
                # compute speed
                torch.cuda.synchronize()
                t_start = time.time()

            tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
            poses, poses_inv = compute_pose_with_inv(
                pose_net, tgt_img, ref_imgs)

            (photo_loss, smooth_loss, geometry_loss, ssim_loss,
             ref_imgs_warped, valid_mask) = mono_warper.compute_photo_and_geometry_loss(
                tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv)

            if args.with_timing:
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                avg_time += elapsed_time

            # tv.utils.save_image(
            #     tgt_depth[0], results_depth_dir/'{0:03d}.png'.format(i))

            for j, (mask, f_name) in enumerate(zip(valid_mask, f_names)):
                colour_mask = utils.tensor2array(
                    mask, max_value=1.0, colormap='bone')
                colour_mask = colour_mask.transpose(1, 2, 0)*255
                colour_mask = colour_mask.astype(np.uint8)
                im = Image.fromarray(colour_mask)
                # im.save(results_mask_dir /
                #         '{0:03d}.png'.format(i*args.batch_size+j))
                im.save(results_mask_dir / f_name)

        if args.with_timing:
            avg_time /= nframes
            print('Avg Time: ', avg_time, ' seconds.')
            print('Avg Speed: ', 1.0 / avg_time, ' fps')


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [disp_to_depth(disp) for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [disp_to_depth(disp) for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths

# def disp_to_depth(disp):
#     """Convert network's sigmoid output into depth prediction
#     The formula for this conversion is given in the 'additional considerations'
#     section of the paper.
#     """
#     # Disp is not scaled in DispResNet.
#     min_depth = 0.1
#     max_depth = 100.0
#     min_disp = 1 / max_depth
#     max_disp = 1 / min_depth
#     # disp = disp.clamp(min=1e-6)
#     scaled_disp = min_disp + (max_disp - min_disp) * disp
#     depth = 1 / scaled_disp
#     # depth = 1./disp
#     return depth


def disp_to_depth(disp):
    # depth_scale = 10.0
    # id_disp = torch.rand(disp.shape).to(device)*1e-12
    # disp = disp + id_disp
    depth = 1./disp
    # depth = depth/depth_scale
    depth = depth.clamp(min=1e-6)
    return depth


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=False):
        main()
