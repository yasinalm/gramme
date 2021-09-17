import argparse
import time
import csv
import datetime
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms

import models

import custom_transforms_mono
import custom_transforms_stereo
import custom_transforms
import utils
from datasets.sequence_folders_lidar import SequenceFolder
from inverse_warp_radar import Warper
from inverse_warp_vo2 import MonoWarper
from radar_eval.eval_utils import getTraj, RadarEvalOdom
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

# Supress UserWarning from grid_sample
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Unsupervised Geometry-Aware Ego-motion Estimation for LIDARs and cameras.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N',
                    help='sequence length for training', default=3)
parser.add_argument('--skip-frames', type=int, metavar='N',
                    help='gap between frames', default=1)
parser.add_argument('-j', '--workers', default=4, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--train-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--val-size', default=0, type=int, metavar='N',
                    help='manual validation size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float,
                    metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0,
                    type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv',
                    metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv',
                    metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', type=int,  default=1,
                    help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales',
                    type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('--img-height', type=int,
                    help='resized mono image height', metavar='W', default=192)
parser.add_argument('--img-width', type=int,
                    help='resized mono image width', metavar='W', default=320)
parser.add_argument('-p', '--photo-loss-weight', type=float,
                    help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-f', '--fft-loss-weight', type=float,
                    help='weight for FFT loss', metavar='W', default=3e-4)
parser.add_argument('-s', '--ssim-loss-weight', type=float,
                    help='weight for SSIM loss', metavar='W', default=1)
# parser.add_argument('-s', '--smooth-loss-weight', type=float,
#                     help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float,
                    help='weight for depth consistency loss', metavar='W', default=1.0)
parser.add_argument('--with-ssim', type=int,
                    default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1,
                    help='with the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int, default=1,
                    help='with the mask for stationary points')
parser.add_argument('--with-masknet', action='store_true',
                    help='with the masknet for multipath and noise')
parser.add_argument('--masknet', type=str,
                    choices=['convnet', 'resnet'], default='convnet', help='MaskNet type')
parser.add_argument('--with-vo', action='store_true',
                    help='with VO fusion')
parser.add_argument('--cam-mode', type=str, choices=[
                    'mono', 'stereo'], default='stereo', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp',
                    default=None, metavar='PATH', help='path to pre-trained DispResNet model')
parser.add_argument('--pretrained-vo-pose', dest='pretrained_vo_pose', default=None,
                    metavar='PATH', help='path to pre-trained VO Pose net model')
parser.add_argument('--with-pretrain', action='store_true',
                    help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='robotcar', help='the dataset to train')
parser.add_argument('--with-preprocessed', type=int, default=1,
                    help='use the preprocessed undistorted images')
parser.add_argument('--pretrained-mask', dest='pretrained_mask',
                    default=None, metavar='PATH', help='path to pre-trained masknet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None,
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--pretrained-fusenet', dest='pretrained_fusenet', default=None,
                    metavar='PATH', help='path to pre-trained PoseFusionNet model')
parser.add_argument('--pretrained-optim', dest='pretrained_optim', default=None,
                    metavar='PATH', help='path to pre-trained optimizer state')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping')
# parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation')
parser.add_argument('--gt-file', metavar='DIR',
                    help='path to ground truth validation file')
parser.add_argument('--cart-res', type=float,
                    help='Cartesian resolution of LIDAR in meters/pixel', metavar='W', default=0.25)
parser.add_argument('--cart-pixels', type=int,
                    help='Cartesian size in pixels (used for both height and width)', metavar='W', default=512)

n_iter = 0
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

depth_scale = 1.0  # 200 robotcar, 1 radiate


def main():
    global best_error, n_iter, device, depth_scale
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    val_writer = None
    if args.log_output:
        val_writer = SummaryWriter(args.save_path/'valid')

    # Data loading code

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

    # mono_transform = None
    cam_train_transform = None
    cam_valid_transform = None
    if args.with_vo:
        T = custom_transforms_mono if args.cam_mode == 'mono' else custom_transforms_stereo

        imagenet_mean = utils.imagenet_mean
        imagenet_std = utils.imagenet_std
        img_size = (args.img_height, args.img_width)
        if args.dataset == 'robotcar':
            if args.cam_mode == 'stereo':
                depth_scale = 200.0

            if args.with_preprocessed:
                cam_train_transform = T.Compose([
                    # T.ToPILImage(),
                    # T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1,
                                  saturation=0.1, hue=0.1),
                    T.RandomScaleCrop(),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])

                cam_valid_transform = T.Compose([
                    # T.ToPILImage(),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])
            else:
                cam_train_transform = T.Compose([
                    T.ToPILImage(),
                    T.CropBottom(),
                    T.Resize(img_size),
                    # T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2,
                                  saturation=0.2, hue=0.2),
                    T.RandomScaleCrop(),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])

                cam_valid_transform = T.Compose([
                    T.ToPILImage(),
                    T.CropBottom(),
                    T.Resize(img_size),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])

        elif args.dataset == 'radiate':
            if args.cam_mode == 'stereo':
                depth_scale = 10.0

            if args.with_preprocessed:
                cam_train_transform = T.Compose([
                    # T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1,
                                  saturation=0.1, hue=0.1),
                    T.RandomScaleCrop(),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])

                cam_valid_transform = T.Compose([
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])
            else:
                cam_train_transform = T.Compose([
                    # T.ToPILImage(),
                    T.Resize(img_size),
                    # T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1,
                                  saturation=0.1, hue=0.1),
                    T.RandomScaleCrop(),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])

                cam_valid_transform = T.Compose([
                    # T.ToPILImage(),
                    T.Resize(img_size),
                    T.ToTensor(),
                    # T.Normalize(imagenet_mean, imagenet_std)
                ])

    print("=> fetching scenes in '{}'".format(args.data))
    lo_params = {
        'cart_pixels': args.cart_pixels,
    }
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset,
        lo_params=lo_params,
        load_camera=args.with_vo,
        cam_mode=args.cam_mode,
        cam_transform=cam_train_transform,
        cam_preprocessed=args.with_preprocessed
    )

    val_set = SequenceFolder(
        args.data,
        transform=val_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset,
        lo_params=lo_params,
        load_camera=args.with_vo,
        cam_mode=args.cam_mode,
        cam_transform=cam_valid_transform,
        cam_preprocessed=args.with_preprocessed
    )

    print('{} samples found in {} train scenes'.format(
        len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(
        len(val_set), len(val_set.scenes)))
    cl_fn = None
    # if args.with_vo:
    #     cl_fn = mono_collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=cl_fn)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=cl_fn)

    if args.train_size == 0:
        args.train_size = len(train_loader)
    print('Epoch size: ', args.train_size)
    if args.val_size == 0:
        args.val_size = len(val_loader)
    print('Validation size: ', args.val_size)

    # create loss
    print("=> creating loss object")
    warper = Warper(args.with_auto_mask, args.cart_res,
                    args.cart_pixels, args.dataset, args.padding_mode)
    mono_warper = None
    if args.with_vo:
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
    mask_net = None
    if args.with_masknet:
        if args.masknet == 'resnet':
            mask_net = models.MaskResNet(
                args.resnet_layers, args.with_pretrain).to(device)
        elif args.masknet == 'convnet':
            mask_net = models.MaskNet(num_channels=1).to(device)
        else:
            raise NotImplementedError(
                'The chosen MaskNet is not implemented! Given: {}'.format(args.masknet))

    lidar_pose_net = models.PoseResNet(
        args.dataset, args.resnet_layers, args.with_pretrain).to(device)

    disp_net = camera_pose_net = None
    fuse_net = attention_net = None
    lidar_pose_features = None
    camera_pose_features = None
    if args.with_vo:
        disp_net = models.DispResNet(
            args.resnet_layers, args.with_pretrain).to(device)
        # if args.cam_mode == 'mono':
        camera_pose_net = models.PoseResNetMono(
            args.resnet_layers, args.with_pretrain).to(device)
        # else:
        #     camera_pose_net = models.PoseResNetStereo(
        #         args.resnet_layers, args.with_pretrain).to(device)

        fuse_net = models.PoseFusionNet().to(device)
        attention_net = models.AttentionNet().to(device)

        camera_pose_net.encoder.register_forward_hook(
            get_activation(camera_pose_features))
        lidar_pose_net.encoder.register_forward_hook(
            get_activation(lidar_pose_features))

    # load parameters
    if args.with_masknet and args.pretrained_mask:
        print("=> using pre-trained weights for MaskNet")
        weights = torch.load(args.pretrained_mask)
        mask_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseNet")
        weights = torch.load(args.pretrained_pose)
        lidar_pose_net.load_state_dict(weights['state_dict'], strict=False)

    if args.with_vo:
        if args.pretrained_disp:
            print("=> using pre-trained weights for VO DispResNet")
            weights = torch.load(args.pretrained_disp)
            disp_net.load_state_dict(weights['state_dict'], strict=False)
        if args.pretrained_vo_pose:
            print("=> using pre-trained weights for VO PoseNet")
            weights = torch.load(args.pretrained_vo_pose)
            camera_pose_net.load_state_dict(
                weights['state_dict'], strict=False)
        if args.pretrained_fusenet:
            print("=> using pre-trained weights for PoseFusionNet")
            weights = torch.load(args.pretrained_vo_pose)
            camera_pose_net.load_state_dict(
                weights['state_dict'], strict=False)

        disp_net = torch.nn.DataParallel(disp_net)
        camera_pose_net = torch.nn.DataParallel(camera_pose_net)
        fuse_net = torch.nn.DataParallel(fuse_net)
        attention_net = torch.nn.DataParallel(attention_net)

    if args.with_masknet:
        mask_net = torch.nn.DataParallel(mask_net)
    lidar_pose_net = torch.nn.DataParallel(lidar_pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': lidar_pose_net.parameters(), 'lr': args.lr}
        # {'params': lidar_pose_net.parameters(), 'lr': 1e-7}
    ]
    if args.with_masknet:
        optim_params.append({'params': mask_net.parameters(), 'lr': args.lr})
    if args.with_vo:
        optim_params.append(
            {'params': disp_net.parameters(), 'lr': args.lr})
        optim_params.append(
            {'params': camera_pose_net.parameters(), 'lr': args.lr})
        optim_params.append(
            {'params': fuse_net.parameters(), 'lr': args.lr})
        optim_params.append(
            {'params': attention_net.parameters(), 'lr': args.lr})

    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    if args.pretrained_optim:
        print("=> using pre-trained weights for the optimizer")
        weights = torch.load(args.pretrained_optim)
        optimizer.load_state_dict(
            weights['state_dict'])

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss',
                         'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs,
                        train_size=args.train_size, valid_size=args.val_size)
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, mask_net,
                           lidar_pose_net, disp_net, camera_pose_net, fuse_net, optimizer,
                           attention_net, camera_pose_features, lidar_pose_features,
                           logger, training_writer, warper, mono_warper)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        val_loss = validate(
            args, val_loader, mask_net, lidar_pose_net, disp_net, camera_pose_net, fuse_net,
            epoch, logger, warper, mono_warper, val_writer)
        logger.valid_writer.write(' * Avg Loss : {:.3f}'.format(val_loss))

        mask_ckpt_dict = None
        if args.with_masknet:
            mask_ckpt_dict = {
                'epoch': epoch,
                'state_dict': mask_net.module.state_dict()
            }
            utils.save_checkpoint_list(args.save_path, [mask_ckpt_dict],
                                       ['lidar_masknet'])
        if args.with_vo:
            vo_pose_ckpt_dict = {
                'epoch': epoch,
                'state_dict': camera_pose_net.module.state_dict()
            }
            disp_ckpt_dict = {
                'epoch': epoch,
                'state_dict': disp_net.module.state_dict()
            }
            fuse_ckpt_dict = {
                'n_iter': epoch,
                'state_dict': fuse_net.module.state_dict()
            }
            # utils.save_checkpoint_mono(
            #     args.save_path, disp_ckpt_dict, vo_pose_ckpt_dict, fuse_ckpt_dict, epoch=epoch)
            utils.save_checkpoint_list(args.save_path, [disp_ckpt_dict, vo_pose_ckpt_dict, fuse_ckpt_dict],
                                       ['mono_dispnet', 'mono_posenet',
                                           'mono_fusenet'],
                                       epoch=epoch)
        ro_pose_ckpt_dict = {
            'epoch': epoch,
            'state_dict': lidar_pose_net.module.state_dict()
        }
        optim_dict = {
            'epoch': epoch,
            'state_dict': optimizer.state_dict()
        }
        utils.save_checkpoint_list(args.save_path, [ro_pose_ckpt_dict, optim_dict],
                                   ['lidar_posenet', 'lidar_optim'],
                                   epoch=epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss])
    logger.epoch_bar.finish()


def train(
        args, train_loader,
        mask_net, lidar_pose_net, disp_net, camera_pose_net, fuse_net, optimizer,
        attention_net, camera_pose_features, lidar_pose_features,
        logger, train_writer, warper, mono_warper):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(i=11 if args.with_vo else 5, precision=4)
    w1, w2, w3, w4 = args.photo_loss_weight, args.geometry_consistency_weight, args.fft_loss_weight, args.ssim_loss_weight

    # best_error = 9.0e6

    # switch to train mode
    if args.with_masknet:
        mask_net.train()
    if args.with_vo:
        disp_net.train()
        camera_pose_net.train()
        fuse_net.train()
    # lidar_pose_net.eval()
    lidar_pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    # TODO: Each batch must be of the same shape. Could there be a way to use the variable number of camera frames inbetween?
    # collate_fn might help.
    for i, input in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        save_checkpoints = i > 0 and n_iter % 1000 == 0
        tgt_img = input[0]  # [B,1,H,W]
        ref_imgs = input[1]  # [2,B,1,H,W]

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = torch.nan_to_num(tgt_img.to(device))
        ref_imgs = [torch.nan_to_num(img.to(device)) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)

        # compute output
        tgt_mask, ref_masks = None, [
            None for i in range(args.sequence_length-1)]
        if args.with_masknet:
            tgt_mask, ref_masks = compute_mask(mask_net, tgt_img, ref_imgs)
        ro_poses, ro_poses_inv = compute_pose_with_inv(
            lidar_pose_net, tgt_img, ref_imgs)

        vo_loss = 0
        vo2lidar_loss = 0
        if args.with_vo:
            # num_scales = 4, num_match=3
            vo_tgt_img = input[2]  # [B,3,H,W]
            vo_ref_imgs = input[3]  # [2,3,B,3,H,W] First two dims are list
            intrinsics = input[4]
            if args.cam_mode == 'stereo':
                rightTleft = input[5]
                rightTleft = rightTleft.to(device)
            vo_tgt_img = torch.nan_to_num(vo_tgt_img.to(device))
            vo_ref_imgs = [torch.nan_to_num(
                ref_img.to(device)) for ref_img in vo_ref_imgs]
            intrinsics = [i.to(device)for i in intrinsics]
            # tgt_depth: [4,B,1,H,W]
            # ref_depths: [2,3,4,B,1,H,W]
            # vo_poses: [2,3,B,6]
            # vo_poses_inv: [2,3,B,6]
            tgt_depth, ref_depths = compute_depth(
                disp_net, vo_tgt_img, vo_ref_imgs)
            if args.cam_mode == 'mono':
                vo_poses, vo_poses_inv = compute_pose_with_inv(
                    camera_pose_net, vo_tgt_img, vo_ref_imgs)
            else:
                vo_poses, vo_poses_inv = compute_pose_with_inv_stereo(
                    camera_pose_net, vo_tgt_img, vo_ref_imgs, rightTleft)

            # r = (ro_poses[..., 2] + vo_poses[..., 3])/2
            # vo_poses[..., 3] = r
            # r = (ro_poses_inv[..., 2] + vo_poses_inv[..., 3])/2
            # vo_poses_inv[..., 3] = r

            # t = (ro_poses[..., [3, 4]] + 20*vo_poses[..., [1, 2]])/2
            # vo_poses[..., [1, 2]] = t  # -poses[..., 4]
            # t = (ro_poses_inv[..., [3, 4]] +
            #      20*vo_poses_inv[..., [1, 2]])/2
            # vo_poses_inv[..., [1, 2]] = t  # -poses_inv[..., 4]

            # Pass all the corresponding monocular frames, pose and depth variables to the reconstruction module.
            # It calculates the triple-wise losses of the sequence.
            (vo_photo_loss, vo_smooth_loss, vo_geometry_loss, vo_ssim_loss,
             mono_ref_imgs_warped, mono_valid_mask) = mono_warper.compute_photo_and_geometry_loss(
                vo_tgt_img, vo_ref_imgs, intrinsics, tgt_depth, ref_depths, vo_poses, vo_poses_inv)

            # vo_loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            vo_photo_loss = 1.0*vo_photo_loss
            vo_smooth_loss = 0.1*vo_smooth_loss
            vo_geometry_loss = 0.5*vo_geometry_loss
            vo_loss = vo_photo_loss + vo_smooth_loss + vo_geometry_loss + vo_ssim_loss

            # TODO: Calculate attention maps
            # attention_map, cam_conf = attention_net(
            #     camera_pose_features, lidar_pose_features)

            # Drop the right2left stereo pose for radar reconstruction.
            if args.cam_mode == 'stereo':
                vo_poses_mono = torch.cat((
                    depth_scale * vo_poses[1:, ..., :3], vo_poses[1:, ..., 3:]), -1)
                vo_poses_inv_mono = torch.cat((
                    depth_scale * vo_poses_inv[1:, ..., :3], vo_poses_inv[1:, ..., 3:]), -1)
                # Recover the absolute pose scale
                # vo_poses_mono[..., :3] = depth_scale * vo_poses[..., :3]
                # vo_poses_inv_mono[..., :3] = depth_scale * \
                #     vo_poses_inv[..., :3]
            else:
                vo_poses_mono = vo_poses
                vo_poses_inv_mono = vo_poses_inv

            # Scale and project camera pose to lidar frame
            vo2lidar_poses = fuse_net(vo_poses_mono)
            vo2lidar_poses_inv = fuse_net(vo_poses_inv_mono)
            # L1 regularization on VO pose
            # vo2lidar_poses = (ro_poses + vo2lidar_poses)/2
            # vo2lidar_poses_inv = (ro_poses_inv + vo2lidar_poses_inv)/2

            (rec_loss2, geometry_consistency_loss2, fft_loss2, ssim_loss2,
             projected_imgs2, _) = warper.compute_db_loss(tgt_img, ref_imgs, tgt_mask, ref_masks, vo2lidar_poses, vo2lidar_poses_inv)

            vo2lidar_loss = w1*rec_loss2 + w2 * \
                geometry_consistency_loss2 + w3*fft_loss2 + w4*ssim_loss2
            # vo_loss += 5*vo2lidar_loss

        # indices = torch.tensor([4, 5, 3, 1, 2, 0], device=device)
        # vo2lidar_poses = torch.index_select(vo_poses, -1, indices)
        # vo2lidar_poses_inv = torch.index_select(vo_poses_inv, -1, indices)
        (rec_loss, geometry_consistency_loss, fft_loss, ssim_loss,
         projected_imgs, projected_masks) = warper.compute_db_loss(tgt_img, ref_imgs, tgt_mask, ref_masks, ro_poses, ro_poses_inv)

        # loss_1, loss_3 = warper.compute_db_loss(tgt_img, ref_imgs, poses, poses_inv)
        # loss_2 = compute_smooth_loss(tgt_mask, tgt_img, ref_masks, ref_imgs)
        rec_loss = w1*rec_loss
        geometry_consistency_loss = w2*geometry_consistency_loss
        fft_loss = w3*fft_loss
        ssim_loss = w4*ssim_loss
        lidar_loss = rec_loss + geometry_consistency_loss + fft_loss + ssim_loss

        vo_loss = 50*vo_loss
        vo2lidar_loss = 0.5*vo2lidar_loss
        loss = lidar_loss + vo_loss + vo2lidar_loss

        # record loss and EPE
        losses_it = [
            loss.item(), rec_loss.item(), geometry_consistency_loss.item(
            ), fft_loss.item(), ssim_loss.item()
        ]
        if args.with_vo:
            losses_it.extend(
                [vo_loss.item(), vo_photo_loss.item(), vo_smooth_loss.item(), vo_geometry_loss.item(), vo_ssim_loss.item()])
            losses_it.extend(
                [vo2lidar_loss.item()])
        losses.update(losses_it, args.batch_size)

        # compute gradient and do Adam step
        # TODO: Can we repeat this in vo part?
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            # writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
            writer.writerow(losses_it)
        logger.train_bar.update(i+1)
        if log_losses:
            errors = losses.avg
            error_names = ['total_loss', 'rec_loss',
                           'geometry_consistency_loss', 'fft_loss', 'ssim_loss']
            if args.with_vo:
                error_names.extend(
                    ['vo_loss', 'vo_photo_loss', 'vo_smooth_loss', 'vo_geometry_loss', 'vo_ssim_loss'])
                error_names.extend(['vo2lidar_loss'])
            error_string = ', '.join('{} : {:.3f}'.format(name, error)
                                     for name, error in zip(error_names, errors))
            logger.train_writer.write(
                'Train: Batch time {} Data time {} '.format(batch_time, data_time, losses) + error_string)

            for error, name in zip(errors, error_names):
                train_writer.add_scalar('train/'+name, error, n_iter)

        # TODO: is there any pretty way of logging?
        if log_losses:
            tags_rot = ['rot_pred-x', 'rot_pred-y', 'rot_pred-z']
            tags_trans = ['trans_pred-x', 'trans_pred-y', 'trans_pred-z']

            for i, tag in enumerate(tags_rot+tags_trans):
                train_writer.add_histogram(
                    'train/lidar/'+tag, ro_poses[..., i], n_iter)

            if args.with_vo:
                for i, tag in enumerate(tags_trans+tags_rot):
                    train_writer.add_histogram(
                        'train/mono/'+tag, vo_poses_mono[..., i], n_iter)

                for i, tag in enumerate(tags_rot+tags_trans):
                    train_writer.add_histogram(
                        'train/mono2lidar/'+tag, vo2lidar_poses[..., i], n_iter)

            train_writer.add_image(
                'train/lidar/tgt_input', utils.tensor2array(tgt_img[0], max_value=1.0, colormap='bone'), n_iter)
            train_writer.add_image(
                'train/lidar/ref_input', utils.tensor2array(ref_imgs[0][0], max_value=1.0, colormap='bone'), n_iter)
            train_writer.add_image(
                'train/lidar/warped_ref', utils.tensor2array(projected_imgs[0][0], max_value=1.0, colormap='bone'), n_iter)
            if args.with_masknet:
                train_writer.add_image(
                    'train/lidar/warped_mask', utils.tensor2array(projected_masks[0][0], max_value=1.0, colormap='bone'), n_iter)
                train_writer.add_image(
                    'train/lidar/tgt_mask', utils.tensor2array(tgt_mask[0], max_value=1.0, colormap='bone'), n_iter)
            if args.with_vo:
                train_writer.add_image(
                    'train/mono/tgt_input_left', utils.tensor2array(vo_tgt_img[0]), n_iter)
                if args.cam_mode == 'mono':
                    train_writer.add_image(
                        'train/mono/ref_input_left', utils.tensor2array(vo_ref_imgs[0][0]), n_iter)
                    train_writer.add_image(
                        'train/mono/ref_warped_left', utils.tensor2array(mono_ref_imgs_warped[0][0]), n_iter)
                elif args.cam_mode == 'stereo':
                    train_writer.add_image(
                        'train/mono/ref_input_left', utils.tensor2array(vo_ref_imgs[1][0]), n_iter)
                    train_writer.add_image(
                        'train/mono/ref_warped_left', utils.tensor2array(mono_ref_imgs_warped[1][0]), n_iter)
                    train_writer.add_image(
                        'train/mono/ref_input_right', utils.tensor2array(vo_ref_imgs[0][0]), n_iter)
                    train_writer.add_image(
                        'train/mono/ref_warped_right', utils.tensor2array(mono_ref_imgs_warped[0][0]), n_iter)
                train_writer.add_image(
                    'train/mono/tgt_disp', utils.tensor2array(1/tgt_depth[0][0], colormap='viridis'), n_iter)
                train_writer.add_image(
                    'train/mono/tgt_depth', utils.tensor2array(tgt_depth[0][0], colormap='inferno'), n_iter)
                train_writer.add_image(
                    'train/mono/warped_mask', utils.tensor2array(mono_valid_mask[0], max_value=1.0, colormap='bone'), n_iter)

                train_writer.add_image(
                    'train/lidar/warped_ref_from_mono', utils.tensor2array(projected_imgs2[0][0], max_value=1.0, colormap='bone'), n_iter)

        if save_checkpoints:
            mask_ckpt_dict = None
            if args.with_masknet:
                mask_ckpt_dict = {
                    'n_iter': n_iter,
                    'state_dict': mask_net.module.state_dict()
                }
                utils.save_checkpoint_list(args.save_path, [mask_ckpt_dict],
                                           ['lidar_masknet'])
            if args.with_vo:
                vo_pose_ckpt_dict = {
                    'n_iter': n_iter,
                    'state_dict': camera_pose_net.module.state_dict()
                }
                disp_ckpt_dict = {
                    'n_iter': n_iter,
                    'state_dict': disp_net.module.state_dict()
                }
                fuse_ckpt_dict = {
                    'n_iter': n_iter,
                    'state_dict': fuse_net.module.state_dict()
                }
                utils.save_checkpoint_list(args.save_path, [disp_ckpt_dict, vo_pose_ckpt_dict, fuse_ckpt_dict],
                                           ['mono_dispnet', 'mono_posenet', 'mono_fusenet'])
            ro_pose_ckpt_dict = {
                'n_iter': n_iter,
                'state_dict': lidar_pose_net.module.state_dict()
            }
            optim_dict = {
                'n_iter': n_iter,
                'state_dict': optimizer.state_dict()
            }
            utils.save_checkpoint_list(args.save_path, [ro_pose_ckpt_dict, optim_dict],
                                       ['lidar_posenet', 'lidar_optim'])

        if i >= args.train_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate(
        args, val_loader, mask_net, lidar_pose_net, disp_net, camera_pose_net, fuse_net,
        epoch, logger, warper, mono_warper, val_writer):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=10 if args.with_vo else 5, precision=4)
    log_outputs = val_writer is not None
    w1, w2, w3, w4 = args.photo_loss_weight, args.geometry_consistency_weight, args.fft_loss_weight, args.ssim_loss_weight

    # switch to eval mode
    if args.with_masknet:
        mask_net.eval()
    if args.with_vo:
        disp_net.eval()
        camera_pose_net.eval()
        fuse_net.eval()
    lidar_pose_net.eval()

    all_poses = []
    all_inv_poses = []
    all_poses_mono = []
    all_inv_poses_mono = []
    all_poses_mono2lidar = []
    all_inv_poses_mono2lidar = []

    # Randomly choose n indices to log images
    rng = np.random.default_rng()
    log_ind = rng.integers(len(val_loader), size=1)

    end = time.time()
    logger.valid_bar.update(0)
    for i, input in enumerate(val_loader):
        tgt_img = input[0]
        ref_imgs = input[1]
        tgt_img = torch.nan_to_num(tgt_img.to(device))
        ref_imgs = [torch.nan_to_num(img.to(device)) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)
        # intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_mask, ref_masks = None, [
            None for i in range(args.sequence_length-1)]
        if args.with_masknet:
            tgt_mask, ref_masks = compute_mask(mask_net, tgt_img, ref_imgs)
        ro_poses, ro_poses_inv = compute_pose_with_inv(
            lidar_pose_net, tgt_img, ref_imgs)
        all_poses.append(ro_poses)
        all_inv_poses.append(ro_poses_inv)

        (rec_loss, geometry_consistency_loss, fft_loss, ssim_loss,
         projected_imgs, projected_masks) = warper.compute_db_loss(
            tgt_img, ref_imgs, tgt_mask, ref_masks, ro_poses, ro_poses_inv)

        rec_loss = w1*rec_loss
        geometry_consistency_loss = w2*geometry_consistency_loss
        fft_loss = w3*fft_loss
        ssim_loss = w4*ssim_loss
        lidar_loss = rec_loss + geometry_consistency_loss + fft_loss + ssim_loss

        vo_loss = 0
        if args.with_vo:
            # num_scales = 4, num_match=3
            vo_tgt_img = input[2]  # [B,3,H,W]
            vo_ref_imgs = input[3]  # [2,3,B,3,H,W] First two dims are list
            intrinsics = input[4]
            vo_tgt_img = torch.nan_to_num(vo_tgt_img.to(device))
            vo_ref_imgs = [torch.nan_to_num(
                ref_img.to(device)) for ref_img in vo_ref_imgs]
            intrinsics = [i.to(device)for i in intrinsics]
            # tgt_depth: [4,B,1,H,W]
            # ref_depths: [2,3,4,B,1,H,W]
            # vo_poses: [2,3,B,6]
            # vo_poses_inv: [2,3,B,6]

            tgt_depth, ref_depths = compute_depth(
                disp_net, vo_tgt_img, vo_ref_imgs)
            vo_poses, vo_poses_inv = compute_pose_with_inv(
                camera_pose_net, vo_tgt_img, vo_ref_imgs)

            # Recover the absolute pose scale
            vo_poses_mono = torch.cat((
                depth_scale * vo_poses[..., :3], vo_poses[..., 3:]), -1)
            vo_poses_inv_mono = torch.cat((
                depth_scale * vo_poses_inv[..., :3], vo_poses_inv[..., 3:]), -1)

            # Collect camera poses in radar format ([rx,ry,rz,tx,ty,tz])
            all_poses_mono.append(
                torch.cat((vo_poses_mono[..., 3:], vo_poses_mono[..., :3]), -1))
            all_inv_poses_mono.append(
                torch.cat((vo_poses_inv_mono[..., 3:], vo_poses_inv_mono[..., :3]), -1))

            # t = (ro_poses[..., [3, 4]] + vo_poses[..., [1, 2]])/2
            # vo_poses[..., [1, 2]] = t
            # t = (ro_poses_inv[..., [3, 4]] +
            #      vo_poses_inv[..., [1, 2]])/2
            # vo_poses_inv[..., [1, 2]] = t

            # Pass all the corresponding monocular frames, pose and depth variables to the reconstruction module.
            # It calculates the triple-wise losses of the sequence.
            (vo_photo_loss, vo_smooth_loss, vo_geometry_loss, vo_ssim_loss,
             mono_ref_imgs_warped, mono_valid_mask) = mono_warper.compute_photo_and_geometry_loss(
                vo_tgt_img, vo_ref_imgs, intrinsics, tgt_depth, ref_depths, vo_poses, vo_poses_inv)

            # vo_loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            vo_photo_loss = 1.0*vo_photo_loss
            vo_smooth_loss = 0.1*vo_smooth_loss
            vo_geometry_loss = 0.5*vo_geometry_loss
            vo_loss = vo_photo_loss + vo_smooth_loss + vo_geometry_loss + vo_ssim_loss

            vo2lidar_poses = fuse_net(vo_poses_mono)
            vo2lidar_poses_inv = fuse_net(vo_poses_inv_mono)

            # Change VO pose order to RO
            # all_poses_mono.append(
            #     torch.cat((vo_poses[..., 3:], vo_poses[..., :3]), -1))
            # all_inv_poses_mono.append(
            #     torch.cat((vo_poses_inv[..., 3:], vo_poses_inv[..., :3]), -1))
            all_poses_mono2lidar.append(vo2lidar_poses)
            all_inv_poses_mono2lidar.append(vo2lidar_poses_inv)

        loss = lidar_loss + vo_loss

        if log_outputs and i in log_ind:
            val_writer.add_image(
                'val/lidar/tgt_input', utils.tensor2array(tgt_img[0], max_value=1.0, colormap='bone'), epoch)
            val_writer.add_image(
                'val/lidar/ref_input', utils.tensor2array(ref_imgs[0][0], max_value=1.0, colormap='bone'), epoch)
            val_writer.add_image(
                'val/lidar/warped_ref', utils.tensor2array(projected_imgs[0][0], max_value=1.0, colormap='bone'), epoch)
            if args.with_masknet:
                val_writer.add_image(
                    'val/lidar/warped_mask', utils.tensor2array(projected_masks[0][0], max_value=1.0, colormap='bone'), epoch)
                val_writer.add_image(
                    'val/lidar/tgt_mask', utils.tensor2array(tgt_mask[0], max_value=1.0, colormap='bone'), epoch)
            if args.with_vo:
                val_writer.add_image(
                    'val/mono/tgt_input', utils.tensor2array(vo_tgt_img[0]), epoch)
                val_writer.add_image(
                    'val/mono/ref_input', utils.tensor2array(vo_ref_imgs[0][0]), epoch)
                val_writer.add_image(
                    'val/mono/warped_ref', utils.tensor2array(mono_ref_imgs_warped[0][0]), epoch)
                val_writer.add_image(
                    'val/mono/tgt_disp', utils.tensor2array(1/tgt_depth[0][0], colormap='viridis'), epoch)
                val_writer.add_image(
                    'val/mono/tgt_depth', utils.tensor2array(tgt_depth[0][0], colormap='inferno'), epoch)
                val_writer.add_image(
                    'val/mono/warped_mask', utils.tensor2array(mono_valid_mask[0], max_value=1.0, colormap='bone'), epoch)

        losses_it = [loss.item(), rec_loss.item(), geometry_consistency_loss.item(
        ), fft_loss.item(), ssim_loss.item()]
        if args.with_vo:
            losses_it.extend(
                [vo_loss.item(), vo_photo_loss.item(), vo_smooth_loss.item(), vo_geometry_loss.item(), vo_ssim_loss.item()])
        losses.update(losses_it, args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {} Loss {}'.format(batch_time, losses))
        if i >= args.val_size - 1:
            break

    if log_outputs:
        # Log predicted relative poses in histograms
        all_poses_t = torch.cat(all_poses, 1)  # [seq_length, N, 6]
        tags = ['rot_pred-x', 'rot_pred-y', 'rot_pred-z',
                'trans_pred-x', 'trans_pred-y', 'trans_pred-z']

        for i, tag in enumerate(tags):
            val_writer.add_histogram(
                'val/lidar/'+tag, all_poses_t[..., i], epoch)

        if args.with_vo:
            all_poses_mono_t = torch.cat(
                all_poses_mono, 1)
            for i, tag in enumerate(tags):
                val_writer.add_histogram(
                    'val/mono/'+tag, all_poses_mono_t[..., i], epoch)
            all_poses_mono2lidar_t = torch.cat(
                all_poses_mono2lidar, 1)
            for i, tag in enumerate(tags):
                val_writer.add_histogram(
                    'val/mono_fused/'+tag, all_poses_mono2lidar_t[..., i], epoch)

    logger.valid_bar.update(args.val_size)

    # Log errors
    errors = losses.avg
    error_names = ['total_loss', 'rec_loss',
                   'geometry_consistency_loss', 'fft_loss', 'ssim_loss']
    if args.with_vo:
        error_names.extend(
            ['vo_loss', 'vo_photo_loss', 'vo_smooth_loss', 'vo_geometry_loss', 'vo_ssim_loss'])
    error_string = ', '.join('{} : {:.3f}'.format(name, error)
                             for name, error in zip(error_names, errors))
    logger.valid_writer.write(' * Avg {}'.format(error_string))

    if log_outputs:
        for error, name in zip(errors, error_names):
            val_writer.add_scalar('val/'+name, error, epoch)

    if args.gt_file is not None:
        # TODO: Plot mono pose results with ground-truth
        ro_eval = RadarEvalOdom(args.gt_file, args.dataset)

        ate_f, f_pred_xyz, f_pred = ro_eval.eval_ref_poses(
            all_poses, all_inv_poses, args.skip_frames)

        if args.with_vo:
            ate_f_mono, f_pred_xyz_mono, f_pred_mono = ro_eval.eval_ref_poses(
                all_poses_mono, all_inv_poses_mono, args.skip_frames, estimate_scale=True)
            ate_f_mono2lidar, f_pred_xyz_mono2lidar, f_pred_mono2lidar = ro_eval.eval_ref_poses(
                all_poses_mono2lidar, all_inv_poses_mono2lidar, args.skip_frames)

        if log_outputs:
            # Plot and log aligned trajectory
            fig = utils.traj2Fig_withgt(
                f_pred_xyz.squeeze(), ro_eval.gt[:, :3, 3].squeeze())
            # fig2= utils.traj2Fig(f_pred[:,:3,3])
            val_writer.add_figure(
                'val/fig/lidar/traj_aligned_pred', fig, epoch)
            # output_writers[0].add_figure('val/fig/traj_pred_full_aligned', fig2, epoch)

            if args.with_vo:
                # Plot and log aligned trajectory
                fig_mono = utils.traj2Fig_withgt(
                    f_pred_xyz_mono.squeeze(), ro_eval.gt[:, :3, 3].squeeze(), axes=[2, 0])
                val_writer.add_figure(
                    'val/fig/mono/traj_aligned_pred', fig_mono, epoch)
                # Plot and log aligned trajectory
                fig_mono2lidar = utils.traj2Fig_withgt(
                    f_pred_xyz_mono2lidar.squeeze(), ro_eval.gt[:, :3, 3].squeeze())
                val_writer.add_figure(
                    'val/fig/mono2lidar/traj_aligned_pred', fig_mono2lidar, epoch)
    else:
        b_pred_xyz, f_pred_xyz = getTraj(
            all_poses, all_inv_poses, args.skip_frames)

        if log_outputs:
            # Plot and log predicted trajectory
            b_fig = utils.traj2Fig(b_pred_xyz)
            f_fig = utils.traj2Fig(f_pred_xyz)
            val_writer.add_figure('val/fig/lidar/b_traj_pred', b_fig, epoch)
            val_writer.add_figure('val/fig/lidar/f_traj_pred', f_fig, epoch)

        if args.with_vo:
            b_pred_xyz_mono, f_pred_xyz_mono = getTraj(
                all_poses_mono, all_inv_poses_mono, args.skip_frames)
            if log_outputs:
                # Plot and log predicted trajectory
                b_fig = utils.traj2Fig(b_pred_xyz_mono, axes=[2, 0])
                f_fig = utils.traj2Fig(f_pred_xyz_mono, axes=[2, 0])
                val_writer.add_figure('val/fig/mono/b_traj_pred', b_fig, epoch)
                val_writer.add_figure('val/fig/mono/f_traj_pred', f_fig, epoch)

            b_pred_xyz_mono2lidar, f_pred_xyz_mono2lidar = getTraj(
                all_poses_mono2lidar, all_inv_poses_mono2lidar, args.skip_frames)
            if log_outputs:
                # Plot and log predicted trajectory
                b_fig = utils.traj2Fig(b_pred_xyz_mono2lidar)
                f_fig = utils.traj2Fig(f_pred_xyz_mono2lidar)
                val_writer.add_figure(
                    'val/fig/mono2lidar/b_traj_pred', b_fig, epoch)
                val_writer.add_figure(
                    'val/fig/mono2lidar/f_traj_pred', f_fig, epoch)

    return losses.avg[0]


def compute_mask(mask_net, tgt_img, ref_imgs):
    # tgt_mask = [mask for mask in mask_net(tgt_img)] # for multiple scale
    tgt_mask = mask_net(tgt_img)  # for single scale

    ref_masks = []
    for ref_img in ref_imgs:
        # ref_mask = [mask for mask in mask_net(ref_img)] # for multiple scale
        ref_mask = mask_net(ref_img)  # for multiple scale
        ref_masks.append(ref_mask)

    return tgt_mask, ref_masks


# def compute_depth(disp_net, tgt_img, ref_imgs):
#     """Compute the depth map of the given monocular RGB frames.

#     Args:
#         disp_net (nn.module): Disparity network
#         tgt_img (torch.Tensor): Target RGB image [B,3,H,W]
#         ref_imgs (List[List[torch.Tensor]]): Reference monocular images [2,3,B,3,H,W]

#     Returns:
#         tgt_depth (List[torch.Tensor]): Return target depth in 4 scales [4,B,1,H,W]
#         ref_depths (List[List[torch.Tensor]]): Return reference depth maps in 4 scales [2,3,4,B,1,H,W]
#     """

#     tgt_depth = [1/(0.01+0.99*disp) for disp in disp_net(tgt_img)]

#     ref_depths = []
#     for ref_matches in ref_imgs:
#         match_depths = []
#         for ref_img in ref_matches:
#             ref_depth = [1/(0.01+0.99*disp) for disp in disp_net(ref_img)]
#             match_depths.append(ref_depth)
#         ref_depths.append(match_depths)

#     return tgt_depth, ref_depths

def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [disp_to_depth(disp) for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [disp_to_depth(disp) for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def disp_to_depth(disp):
    global depth_scale
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    # Disp is not scaled in DispResNet.
    # min_depth = 0.1
    # max_depth = 100.0
    # min_disp = 1 / max_depth
    # max_disp = 1 / min_depth
    # scaled_disp = min_disp + (max_disp - min_disp) * disp
    # depth = 1 / scaled_disp
    # disp = disp.clamp(min=1e-2)

    id_disp = torch.rand(disp.shape).to(device)*1e-12
    disp = disp + id_disp
    disp = disp.clamp(min=1e-3)
    depth = 1./disp
    depth = depth/depth_scale
    depth = depth.clamp(min=1e-3)
    return depth


def compute_pose_with_inv_stereo(pose_net, tgt_img, ref_imgs, rightTleft):
    global depth_scale
    poses = [rightTleft/depth_scale]
    poses_inv = [-(rightTleft.clone())/depth_scale]
    for ref_img in ref_imgs[1:]:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return torch.stack(poses), torch.stack(poses_inv)
    # return poses, poses_inv


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return torch.stack(poses), torch.stack(poses_inv)


def compute_mono_pose_with_inv(pose_net, tgt_img, ref_imgs):
    """Make a sequence of reference and target images. 
    Compute monocular triple-wise relative pose values of reference frames with respect to the target. 

    Args:
        pose_net (nn.module): PoseNet
        tgt_img (torch.Tensor): Target image [B,3,H,W]
        ref_imgs (List[List[torch.Tensor]]): Reference monocular images [2,3,B,3,H,W]

    Returns:
        vo_poses: [2,3,B,6]
        vo_poses_inv: [2,3,B,6]
    """

    # poses = []
    # poses_inv = []
    # for ref_matches in ref_imgs:

    # Assume tgt=9, refs=[6,12], then
    # backward = [p7-6, p8-7, p9-8]
    # forward = [p9-10, p10-11, p11-12]
    backward = []
    backward_inv = []
    ref_matches = ref_imgs[0]
    for ref, tgt in zip(ref_matches, ref_matches[1:]+[tgt_img]):
        backward.append(pose_net(tgt, ref))
        backward_inv.append(pose_net(ref, tgt))
    backward = torch.stack(backward)
    backward_inv = torch.stack(backward_inv)

    forward = []
    forward_inv = []
    ref_matches = ref_imgs[1]
    for ref, tgt in zip([tgt_img] + ref_matches[:-1], ref_matches):
        forward.append(pose_net(tgt, ref))
        forward_inv.append(pose_net(ref, tgt))
    forward = torch.stack(forward)
    forward_inv = torch.stack(forward_inv)

    poses = [backward, forward]
    poses_inv = [backward_inv, forward_inv]

    # [B, 2, 3, 6], [B, 2, 3, 6]
    return torch.stack(poses), torch.stack(poses_inv)


def get_activation(pose_features):
    def pose_hook(module, input, output):
        pose_features = output
    return pose_hook


def mono_collate_fn(batch):
    b_tgt_img = torch.Tensor([input[0] for input in batch])
    b_ref_imgs = torch.Tensor([input[1] for input in batch])
    b_vo_tgt_img = [input[2] for input in batch]
    b_vo_ref_imgs = [input[3] for input in batch]

    return b_tgt_img, b_ref_imgs, b_vo_tgt_img, b_vo_ref_imgs


if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=False):
        main()
