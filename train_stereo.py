import argparse
import time
import csv
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import models

import custom_transforms_stereo as T
import utils
from radar_eval.eval_utils import getTraj
from datasets.sequence_folders_stereo import SequenceFolder
from inverse_warp_vo2 import MonoWarper
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Unsupervised Geometry-Aware Ego-motion Estimation for stereo cameras.',
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
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float,
                    metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0,
                    type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--ckpt-freq', default=1, type=int,
                    metavar='N', help='checkpoint saving frequency in terms of epochs')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv',
                    metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv',
                    metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true',
                    help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales',
                    type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('--img-height', type=int,
                    help='resized image height', metavar='W', default=192)
parser.add_argument('--img-width', type=int,
                    help='resized image width', metavar='W', default=320)
parser.add_argument('-p', '--photo-loss-weight', type=float,
                    help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float,
                    help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float,
                    help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int,
                    default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1,
                    help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=1,
                    help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1,
                    help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate', 'cadcd'], default='robotcar', help='the dataset to train')
parser.add_argument('--with-preprocessed', type=int, default=1,
                    help='use the preprocessed undistorted images')
parser.add_argument('--pretrained-disp', dest='pretrained_disp',
                    default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None,
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--pretrained-optim', dest='pretrained_optim', default=None,
                    metavar='PATH', help='path to pre-trained optimizer state')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--gt-file', metavar='DIR',
                    help='path to ground truth validation file')


n_iter = 0
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

depth_scale = 1.0  # 200 robotcar, 1 radiate


def main():
    global depth_scale, best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.mkdir(parents=True)

    torch.set_default_dtype(torch.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    val_writer = SummaryWriter(args.save_path/'valid')

    # Data loading code
    # normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
    #                                         std=[0.225, 0.225, 0.225])
    # mean_imgnet, std_imgnet = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    imagenet_mean = utils.imagenet_mean
    imagenet_std = utils.imagenet_std
    img_size = (args.img_height, args.img_width)
    if args.dataset == 'robotcar':
        depth_scale = 200.0
        if args.with_preprocessed:
            train_transform = T.Compose([
                # T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1),
                T.RandomScaleCrop(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            train_transform = T.Compose([
                T.ToPILImage(),
                T.CropBottom(),
                T.Resize(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1),
                T.RandomScaleCrop(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

            valid_transform = T.Compose([
                T.ToPILImage(),
                T.CropBottom(),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

    elif args.dataset == 'radiate':
        depth_scale = 10.0
        if args.with_preprocessed:
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1),
                T.RandomScaleCrop(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

            valid_transform = T.Compose([
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            train_transform = T.Compose([
                # T.ToPILImage(),
                T.Resize(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1),
                T.RandomScaleCrop(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
    elif args.dataset == 'cadcd':
        depth_scale = 20.0
        if args.with_preprocessed:
            train_transform = T.Compose([
                # T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1),
                T.RandomScaleCrop(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            train_transform = T.Compose([
                # T.ToPILImage(),
                T.CropBottom(250),
                T.CropSides(),
                T.Resize(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1),
                T.RandomScaleCrop(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.CropBottom(250),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        dataset=args.dataset,
        transform=train_transform,
        seed=args.seed,
        # train=True,
        mode='train',
        sequence_length=args.sequence_length,
        preprocessed=args.with_preprocessed
    )

    val_set = SequenceFolder(
        args.data,
        dataset=args.dataset,
        transform=valid_transform,
        seed=args.seed,
        mode='val',
        # train=False,
        sequence_length=args.sequence_length,
        preprocessed=args.with_preprocessed
    )

    print('{} samples found in {} train scenes'.format(
        len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(
        len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

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
        args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNetStereo(
        args.resnet_layers, args.with_pretrain).to(device)

    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
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

    logger = TermLogger(n_epochs=args.epochs, train_size=min(
        len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net,
                           optimizer, logger, training_writer, mono_warper)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        val_loss = validate(
            args, val_loader, disp_net, pose_net, epoch, logger, mono_warper, val_writer)
        logger.valid_writer.write(' * Avg Loss : {:.3f}'.format(val_loss))

        if epoch % args.ckpt_freq == 0:
            pose_dict = {
                'epoch': epoch,
                'state_dict': pose_net.module.state_dict()
            }
            disp_dict = {
                'epoch': epoch,
                'state_dict': disp_net.module.state_dict()
            }
            optim_dict = {
                'epoch': epoch,
                'state_dict': optimizer.state_dict()
            }
            utils.save_checkpoint_list(args.save_path, [pose_dict, disp_dict, optim_dict],
                                       ['stereo_pose', 'stereo_disp', 'stereo_optim'], epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, logger, train_writer, mono_warper):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    # disp_net.eval()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, rightTleft) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # print("{} _ {}".format(tgt_img.min(), tgt_img.max()))
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = torch.nan_to_num(tgt_img.to(device), posinf=1, neginf=0)
        ref_imgs = [torch.nan_to_num(
            img.to(device), posinf=1, neginf=0) for img in ref_imgs]
        intrinsics = [i.to(device)for i in intrinsics]
        rightTleft = rightTleft.to(device)

        # The code below might be useful for debugging anomalies in the input.
        # assert ~torch.isnan(tgt_img).any()
        # assert all([~torch.isnan(ref_img).any() for ref_img in ref_imgs])

        # if torch.isnan(tgt_img).any():
        #     # print("nan detected in tgt_img")
        #     continue

        # if any([torch.isnan(ref_img).any() for ref_img in ref_imgs]):
        #     # print("nan detected in ref_imgs")
        #     continue

        # print("{} _ {}".format(tgt_img.min(), tgt_img.max()))

        # condition = [tgt_img.min()>=0]
        # condition.append([tgt_img.max()<=1.0])
        # for ref in ref_imgs:
        #     print(ref.min())
        #     print(ref.max())
        #     condition.append([ref.min()>=0])
        #     condition.append([ref.max()<=1.0])

        # assert(all(condition)), "Not all images are in [0,1]"

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv_stereo(
            pose_net, tgt_img, ref_imgs, rightTleft)

        (photo_loss, smooth_loss, geometry_loss, ssim_loss,
         ref_imgs_warped, valid_mask) = mono_warper.compute_photo_and_geometry_loss(
            tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv)

        loss = w1*photo_loss + w2*smooth_loss + \
            w3*geometry_loss  # + ssim_loss

        if log_losses:
            # poses = torch.cat(poses[0:2])  # Choose only previous source frames
            poses = torch.cat(poses)
            train_writer.add_histogram(
                'train/mono/rot_pred-x', poses[..., 3], n_iter)
            train_writer.add_histogram(
                'train/mono/rot_pred-y', poses[..., 4], n_iter)
            train_writer.add_histogram(
                'train/mono/rot_pred-z', poses[..., 5], n_iter)
            train_writer.add_histogram(
                'train/mono/trans_pred-x', depth_scale*poses[..., 0], n_iter)
            train_writer.add_histogram(
                'train/mono/trans_pred-y', depth_scale*poses[..., 1], n_iter)
            train_writer.add_histogram(
                'train/mono/trans_pred-z', depth_scale*poses[..., 2], n_iter)

            train_writer.add_scalar(
                'train/mono/mean_depth', tgt_depth[0][0].mean(), n_iter)

            train_writer.add_scalar(
                'train/mono/photometric_error', photo_loss.item(), n_iter)
            train_writer.add_scalar(
                'train/mono/disparity_smoothness_loss', smooth_loss.item(), n_iter)
            train_writer.add_scalar(
                'train/mono/geometry_consistency_loss', geometry_loss.item(), n_iter)
            train_writer.add_scalar(
                'train/mono/total_loss', loss.item(), n_iter)

            train_writer.add_image(
                'train/mono/input_tgt_left', utils.tensor2array(tgt_img[0]), n_iter)
            train_writer.add_image(
                'train/mono/input_ref_right', utils.tensor2array(ref_imgs[0][0]), n_iter)
            train_writer.add_image(
                'train/mono/warped_ref_right', utils.tensor2array(ref_imgs_warped[0][0]), n_iter)
            train_writer.add_image(
                'train/mono/input_ref', utils.tensor2array(ref_imgs[1][0]), n_iter)
            train_writer.add_image(
                'train/mono/warped_ref', utils.tensor2array(ref_imgs_warped[1][0]), n_iter)

            train_writer.add_image('train/mono/disp', utils.tensor2array(
                1/tgt_depth[0][0], max_value=None, colormap='viridis'), n_iter)
            train_writer.add_image('train/mono/depth', utils.tensor2array(
                tgt_depth[0][0], max_value=None, colormap='inferno'), n_iter)
            train_writer.add_image(
                'train/mono/warped_mask', utils.tensor2array(valid_mask[0], max_value=1.0, colormap='bone'), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % 500 == 0:
            pose_dict = {
                'iter': n_iter,
                'state_dict': pose_net.module.state_dict()
            }
            disp_dict = {
                'iter': n_iter,
                'state_dict': disp_net.module.state_dict()
            }
            optim_dict = {
                'epoch': n_iter,
                'state_dict': optimizer.state_dict()
            }
            utils.save_checkpoint_list(args.save_path, [pose_dict, disp_dict, optim_dict],
                                       ['stereo_pose', 'stereo_disp', 'stereo_optim'])

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), photo_loss.item(),
                             smooth_loss.item(), geometry_loss.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write(
                'Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= args.epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate(args, val_loader, disp_net, pose_net, epoch, logger, mono_warper, val_writer):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = val_writer is not None
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    all_poses = []
    all_inv_poses = []

    # Randomly choose n indices to log images
    rng = np.random.default_rng()
    log_ind = rng.integers(len(val_loader), size=1)

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = [i.to(device)for i in intrinsics]

        tgt_img = torch.nan_to_num(tgt_img.to(device), posinf=1, neginf=0)
        ref_imgs = [torch.nan_to_num(
            img.to(device), posinf=1, neginf=0) for img in ref_imgs]

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        poses = torch.stack(poses)
        poses_inv = torch.stack(poses_inv)

        # Chaneg VO pose order to RO
        all_poses.append(
            torch.cat((poses[..., 3:], depth_scale*poses[..., :3]), -1))
        all_inv_poses.append(
            torch.cat((poses_inv[..., 3:], depth_scale*poses_inv[..., :3]), -1))

        (photo_loss, smooth_loss, geometry_loss, ssim_loss,
         ref_imgs_warped, valid_mask) = mono_warper.compute_photo_and_geometry_loss(
            tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv)

        loss = w1*photo_loss + w2*smooth_loss + \
            w3*geometry_loss  # + ssim_loss
        losses.update([loss.item(), photo_loss.item(),
                       smooth_loss.item(), geometry_loss.item()])

        if log_outputs and i in log_ind:
            val_writer.add_image(
                'val/mono/input_tgt', utils.tensor2array(tgt_img[0]), n_iter)
            val_writer.add_image(
                'val/mono/input_ref', utils.tensor2array(ref_imgs[0][0]), n_iter)
            val_writer.add_image(
                'val/mono/warped_ref', utils.tensor2array(ref_imgs_warped[0][0]), n_iter)

            val_writer.add_image('val/mono/disp', utils.tensor2array(
                1/tgt_depth[0][0], max_value=None, colormap='viridis'), n_iter)
            val_writer.add_image('val/mono/depth', utils.tensor2array(
                tgt_depth[0][0], max_value=None, colormap='inferno'), n_iter)
            val_writer.add_image(
                'val/mono/warped_mask', utils.tensor2array(valid_mask[0], max_value=1.0, colormap='bone'), n_iter)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {} Loss {}'.format(batch_time, losses))

    if log_outputs:
        # Log predicted relative poses in histograms
        all_poses_t = torch.cat(all_poses, 1)  # [seq_length, N, 6]
        val_writer.add_histogram(
            'val/mono/rot_pred-x', all_poses_t[..., 0], n_iter)
        val_writer.add_histogram(
            'val/mono/rot_pred-y', all_poses_t[..., 1], n_iter)
        val_writer.add_histogram(
            'val/mono/rot_pred-z', all_poses_t[..., 2], n_iter)
        val_writer.add_histogram(
            'val/mono/trans_pred-x', all_poses_t[..., 3], n_iter)
        val_writer.add_histogram(
            'val/mono/trans_pred-y', all_poses_t[..., 4], n_iter)
        val_writer.add_histogram(
            'val/mono/trans_pred-z', all_poses_t[..., 5], n_iter)

    logger.valid_bar.update(len(val_loader))

    errors = losses.avg
    error_names = ['Total loss', 'Photo loss',
                   'Smooth loss', 'Consistency loss']

    error_string = ', '.join('{} : {:.3f}'.format(name, error)
                             for name, error in zip(error_names, errors))
    logger.valid_writer.write(' * Avg {}'.format(error_string))

    if log_outputs:
        for error, name in zip(errors, error_names):
            val_writer.add_scalar('val/'+name, error, epoch)

    # TODO: Plot mono pose results with gt file
    if args.gt_file is not None:
        print('Mono evaluation with GT is not supported yet!')
        # ro_eval = RadarEvalOdom(args.gt_file, args.dataset)

        # ate_bs_mean, ate_bs_std, ate_fs_mean, ate_fs_std, f_pred_xyz, f_pred = ro_eval.eval_ref_poses(
        #     all_poses, all_inv_poses, args.skip_frames)

        # if log_outputs:
        #     # Plot and log aligned trajectory
        #     fig = utils.traj2Fig_withgt(
        #         f_pred_xyz.squeeze(), ro_eval.gt[:, :3, 3].squeeze())
        #     # fig2= utils.traj2Fig(f_pred[:,:3,3])
        #     val_writer.add_figure('val/fig/traj_aligned_pred', fig, epoch)
        #     # output_writers[0].add_figure('val/fig/traj_pred_full_aligned', fig2, epoch)

    else:
        b_pred_xyz, f_pred_xyz = getTraj(
            all_poses, all_inv_poses, args.skip_frames)

        if log_outputs:
            # Plot and log predicted trajectory
            b_fig = utils.traj2Fig(b_pred_xyz, axes=[2, 0])
            f_fig = utils.traj2Fig(f_pred_xyz, axes=[2, 0])
            val_writer.add_figure('val/fig/b_traj_pred', b_fig, epoch)
            val_writer.add_figure('val/fig/f_traj_pred', f_fig, epoch)

    return errors[0]


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
    depth = 1./disp
    depth = depth/depth_scale
    depth = depth.clamp(min=1e-6)
    return depth


def compute_pose_with_inv_stereo(pose_net, tgt_img, ref_imgs, rightTleft):
    global depth_scale
    poses = [rightTleft/depth_scale]
    poses_inv = [-(rightTleft.clone())/depth_scale]
    for ref_img in ref_imgs[1:]:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


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
