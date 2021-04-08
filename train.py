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
from torchvision.transforms.transforms import Resize

import models

import custom_transforms
import utils
from datasets.sequence_folders import SequenceFolder
# from datasets.pair_folders import PairFolder
from inverse_warp import Warper
from inverse_warp_vo import MonoWarper
from radar_eval.eval_utils import getTraj, RadarEvalOdom
# from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

# Supress UserWarning from grid_sample
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
# parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
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
# parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
# parser.add_argument('--with-mask', type=int, default=1, help='with the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', action='store_true',
                    help='with the mask for stationary points')
parser.add_argument('--with-masknet', action='store_true',
                    help='with the masknet for multipath and noise')
parser.add_argument('--masknet', type=str,
                    choices=['convnet', 'resnet'], default='convnet', help='MaskNet type')
parser.add_argument('--with-vo', action='store_true',
                    help='with VO fusion')
parser.add_argument('--pretrained-depth', dest='pretrained_depth',
                    default=None, metavar='PATH', help='path to pre-trained DispResNet model')
parser.add_argument('--pretrained-vo-pose', dest='pretrained_vo_pose', default=None,
                    metavar='PATH', help='path to pre-trained VO Pose net model')
parser.add_argument('--with-pretrain', action='store_true',
                    help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='hand', help='the dataset to train')
parser.add_argument('--pretrained-mask', dest='pretrained_mask',
                    default=None, metavar='PATH', help='path to pre-trained masknet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None,
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
# parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation')
parser.add_argument('--gt-file', metavar='DIR',
                    help='path to ground truth validation file')
parser.add_argument('--gt-type', type=str,
                    choices=['kitti', 'xyz'], default='xyz', help='GT format')
parser.add_argument('--radar-format', type=str,
                    choices=['cartesian', 'polar'], default='polar', help='Range-angle format')
parser.add_argument('--range-res', type=float,
                    help='Range resolution of FMCW radar in meters', metavar='W', default=0.0977)
parser.add_argument('--angle-res', type=float,
                    help='Angular azimuth resolution of FMCW radar in radians', metavar='W', default=1.0)
parser.add_argument('--cart-res', type=float,
                    help='Cartesian resolution of FMCW radar in meters/pixel', metavar='W', default=0.25)
parser.add_argument('--cart-pixels', type=int,
                    help='Cartesian size in pixels (used for both height and width)', metavar='W', default=512)
# parser.add_argument('--num-range-bins', type=int, help='Number of ADC samples (range bins)', metavar='W', default=256)
# parser.add_argument('--num-angle-bins', type=int, help='Number of angle bins', metavar='W', default=64)

best_error = -1
n_iter = 0
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    global best_error, n_iter, device
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
    # if args.dataset == 'hand':
    #     mean, std = 119.4501, 6.5258 # Calculated over all dataset
    # else:
    #     mean, std = 11.49, 16.46 # Calculated over all robotcar dataset
    # normalize = custom_transforms.Normalize(mean=mean, std=std)

    # train_transform = custom_transforms.Compose([
    #     custom_transforms.RandomHorizontalFlip(),
    #     custom_transforms.RandomScaleCrop(),
    #     custom_transforms.ArrayToTensor(),
    #     normalize
    # ])

    # ds_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    # Calculate center of handheld dataset. The rotation center is not the image center (default image center).
    if args.dataset == 'hand':
        center = (0, args.cart_pixels//2)
        train_transform = custom_transforms.Compose(
            [custom_transforms.ArrayToTensor(),
             transforms.RandomRotation(degrees=10, center=center)])
    else:
        train_transform = custom_transforms.Compose(
            [custom_transforms.ArrayToTensor(),
             transforms.RandomRotation(10)])

    mono_transform = None
    if args.with_vo:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=imagenet_mean,
                                         std=imagenet_std)
        mono_transform = [transforms.ToTensor()]
        # Resize RADIATE dataset to make it divisible by 64, which is needed in resnet_encoder.
        if args.dataset == 'radiate':
            mono_transform.append(transforms.Resize((384, 640)))
        mono_transform.append(normalize)
        mono_transform = transforms.Compose(mono_transform)

    val_transform = custom_transforms.Compose(
        [custom_transforms.ArrayToTensor()])

    print("=> fetching scenes in '{}'".format(args.data))
    ro_params = {
        'cart_resolution': args.cart_res,
        'cart_pixels': args.cart_pixels,
        'rangeResolutionsInMeter': args.range_res,
        'angleResolutionInRad': args.angle_res,
        'radar_format': args.radar_format
    }
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset,
        ro_params=ro_params,
        load_mono=args.with_vo,
        mono_transform=mono_transform
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    val_set = SequenceFolder(
        args.data,
        transform=val_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset,
        ro_params=ro_params,
        load_mono=args.with_vo,
        mono_transform=mono_transform
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

    # create model
    print("=> creating loss object")
    warper = Warper(args.with_auto_mask, args.cart_res,
                    args.cart_pixels, args.dataset, args.padding_mode)
    mono_warper = None
    if args.with_vo:
        mono_warper = MonoWarper(
            args.num_scales, args.with_auto_mask,
            args.dataset, args.padding_mode)
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
    disp_net = vo_pose_net = None
    if args.with_vo:
        disp_net = models.DispResNet(
            args.resnet_layers, args.with_pretrain).to(device)
        vo_pose_net = models.PoseResNet(
            args.dataset, args.resnet_layers, args.with_pretrain, is_vo=True).to(device)
    pose_net = models.PoseResNet(
        args.dataset, args.resnet_layers, args.with_pretrain).to(device)

    # load parameters
    if args.with_masknet and args.pretrained_mask:
        print("=> using pre-trained weights for MaskNet")
        weights = torch.load(args.pretrained_mask)
        mask_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    if args.with_vo:
        if args.pretrained_depth:
            print("=> using pre-trained weights for VO DepthResNet")
            weights = torch.load(args.pretrained_depth)
            disp_net.load_state_dict(weights['state_dict'], strict=False)
        if args.pretrained_vo_pose:
            print("=> using pre-trained weights for VO PoseNet")
            weights = torch.load(args.pretrained_vo_pose)
            vo_pose_net.load_state_dict(weights['state_dict'], strict=False)

        disp_net = torch.nn.DataParallel(disp_net)
        vo_pose_net = torch.nn.DataParallel(vo_pose_net)

    if args.with_masknet:
        mask_net = torch.nn.DataParallel(mask_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    if args.with_masknet:
        optim_params.append({'params': mask_net.parameters(), 'lr': args.lr})
    if args.with_vo:
        optim_params.append({'params': disp_net.parameters(), 'lr': args.lr})
        optim_params.append(
            {'params': vo_pose_net.parameters(), 'lr': args.lr})

    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

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
                           pose_net, disp_net, vo_pose_net, optimizer, logger, training_writer, warper, mono_warper)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate(
            args, val_loader, mask_net, pose_net, epoch, logger, warper, val_writer)
        error_string = ', '.join('{} : {:.3f}'.format(name, error)
                                 for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            val_writer.add_scalar('val/'+name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        # errors[0] is ATE error for `validate_with_gt`, and average loss for `validate_without_gt`
        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        mask_ckpt_dict = None
        if args.with_masknet:
            mask_ckpt_dict = {
                'epoch': n_iter + 1,
                'state_dict': mask_net.module.state_dict()
            }
        utils.save_checkpoint(
            args.save_path, mask_ckpt_dict, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(
        args, train_loader,
        mask_net, pose_net, disp_net, vo_pose_net, optimizer,
        logger, train_writer, warper, mono_warper):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(i=9 if args.with_vo else 5, precision=4)
    w1, w2, w3, w4 = args.photo_loss_weight, args.geometry_consistency_weight, args.fft_loss_weight, args.ssim_loss_weight

    # best_error = 9.0e6

    # switch to train mode
    if args.with_masknet:
        mask_net.train()
    if args.with_vo:
        disp_net.train()
        vo_pose_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    # TODO: Haydaaa! Her batch ayni boyutta olacak diye hata veriyor mono frame lerden dolayi
    # Simdilik sadece sabit olarak radar frame ler arasinda 3 mono frame oalcak sekilde aliyoruz.
    # Diger sequence leri atiyoruz. Data efficient degil. Daha akilli yol bul. collate_fn ile
    for i, input in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        tgt_img = input[0]
        ref_imgs = input[1]

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)

        # compute output
        tgt_mask, ref_masks = None, [
            None for i in range(args.sequence_length-1)]
        if args.with_masknet:
            tgt_mask, ref_masks = compute_mask(mask_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        (rec_loss, geometry_consistency_loss, fft_loss, ssim_loss,
         projected_imgs, projected_masks) = warper.compute_db_loss(tgt_img, ref_imgs, tgt_mask, ref_masks, poses, poses_inv)

        # loss_1, loss_3 = warper.compute_db_loss(tgt_img, ref_imgs, poses, poses_inv)
        # loss_2 = compute_smooth_loss(tgt_mask, tgt_img, ref_masks, ref_imgs)
        rec_loss = w1*rec_loss
        geometry_consistency_loss = w2*geometry_consistency_loss
        fft_loss = w3*fft_loss
        ssim_loss = w4*ssim_loss
        loss = rec_loss + geometry_consistency_loss + fft_loss + ssim_loss

        if args.with_vo:
            vo_tgt_img = input[2]
            vo_ref_imgs = input[3]
            vo_tgt_img = vo_tgt_img.to(device)
            vo_ref_imgs = [[ref_img.to(device) for ref_img in refs]
                           for refs in vo_ref_imgs]
            tgt_depth, ref_depths = compute_depth(
                disp_net, vo_tgt_img, vo_ref_imgs)
            vo_poses, vo_poses_inv = compute_mono_pose_with_inv(
                vo_pose_net, vo_tgt_img, vo_ref_imgs)

            # Pass all the corresponding monocular frames, pose and depth variables to the reconstruction module.
            # It calculates the triple-wise losses of the sequence.
            vo_photo_loss, vo_smooth_loss, vo_geometry_loss = mono_warper.compute_photo_and_geometry_loss(
                vo_tgt_img, vo_ref_imgs, tgt_depth, ref_depths, vo_poses, vo_poses_inv)

            # vo_loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            vo_photo_loss = 1.0*vo_photo_loss
            vo_smooth_loss = 0.1*vo_smooth_loss
            vo_geometry_loss = 0.5*vo_geometry_loss
            vo_loss = vo_photo_loss + vo_smooth_loss + vo_geometry_loss

            # TODO: radar ve mono loss lar ayri gayri takiliyorlar. henuz fusion yok ortada.
            loss += vo_loss

        if log_losses:
            # train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            # train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar(
                'train/photometric_error', rec_loss.item(), n_iter)
            train_writer.add_scalar(
                'train/geometry_consistency_loss', geometry_consistency_loss.item(), n_iter)
            train_writer.add_scalar('train/fft_loss', fft_loss.item(), n_iter)
            train_writer.add_scalar(
                'train/ssim_loss', ssim_loss.item(), n_iter)
            train_writer.add_scalar('train/total_loss', loss.item(), n_iter)

            # train_writer.add_histogram('train/rot_pred-x', poses[...,0], n_iter)
            # train_writer.add_histogram('train/rot_pred-y', poses[...,1], n_iter)
            train_writer.add_histogram(
                'train/rot_pred-z', poses[..., 2], n_iter)
            train_writer.add_histogram(
                'train/trans_pred-x', poses[..., 3], n_iter)
            train_writer.add_histogram(
                'train/trans_pred-y', poses[..., 4], n_iter)
            # train_writer.add_histogram('train/trans_pred-z', poses[...,5], n_iter)

            # train_writer.add_image(
            #     'train/img/input', utils.tensor2array(tgt_img[0], max_value=1.0, colormap='bone'), n_iter)
            # train_writer.add_image(
            #     'train/img/ref_input', utils.tensor2array(ref_imgs[0][0], max_value=1.0, colormap='bone'), n_iter)
            # train_writer.add_image(
            #     'train/img/warped_input', utils.tensor2array(projected_imgs[0][0], max_value=1.0, colormap='bone'), n_iter)
            # if args.with_masknet:
            #     train_writer.add_image(
            #         'train/img/warped_mask', utils.tensor2array(projected_masks[0][0], max_value=1.0, colormap='bone'), n_iter)
            #     train_writer.add_image(
            #         'train/img/tgt_mask', utils.tensor2array(tgt_mask[0], max_value=1.0, colormap='bone'), n_iter)

        # record loss and EPE
        losses_it = [
            loss.item(), rec_loss.item(), geometry_consistency_loss.item(
            ), fft_loss.item(), ssim_loss.item()
        ]
        if args.with_vo:
            losses_it.extend(
                [vo_loss.item(), vo_photo_loss.item(), vo_smooth_loss.item(), vo_geometry_loss.item()])
        losses.update(losses_it, args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % 1000 == 0:
            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            # decisive_error = loss.item()

            # # remember lowest error and save checkpoint
            # is_best = decisive_error < best_error
            # best_error = min(best_error, decisive_error)
            is_best = False  # Do not choose the best in the training but in the validation wrt. ATE error
            mask_ckpt_dict = None
            if args.with_masknet:
                mask_ckpt_dict = {
                    'epoch': n_iter + 1,
                    'state_dict': mask_net.module.state_dict()
                }
            utils.save_checkpoint(
                args.save_path, mask_ckpt_dict, {
                    'n_iter': n_iter + 1,
                    'state_dict': pose_net.module.state_dict()
                },
                is_best)

            train_writer.add_image(
                'train/radar/input', utils.tensor2array(tgt_img[0], max_value=1.0, colormap='bone'), n_iter)
            train_writer.add_image(
                'train/radar/ref_input', utils.tensor2array(ref_imgs[0][0], max_value=1.0, colormap='bone'), n_iter)
            train_writer.add_image(
                'train/radar/warped_input', utils.tensor2array(projected_imgs[0][0], max_value=1.0, colormap='bone'), n_iter)
            if args.with_masknet:
                train_writer.add_image(
                    'train/radar_mask/warped_mask', utils.tensor2array(projected_masks[0][0], max_value=1.0, colormap='bone'), n_iter)
                train_writer.add_image(
                    'train/radar_mask/tgt_mask', utils.tensor2array(tgt_mask[0], max_value=1.0, colormap='bone'), n_iter)
            if args.with_vo:
                train_writer.add_image(
                    'train/img/input', utils.tensor2array(tgt_img[0]), n_iter)
                train_writer.add_image(
                    'train/img/ref_input', utils.tensor2array(ref_imgs[0][0]), n_iter)
                # train_writer.add_image(
                #     'train/img/warped_input', utils.tensor2array(projected_imgs[0][0]), n_iter)
                train_writer.add_image(
                    'train/depth/tgt_disp', utils.tensor2array(1/tgt_depth[0][0], colormap='magma'), n_iter)
                train_writer.add_image(
                    'train/depth/tgt_depth', utils.tensor2array(tgt_depth[0][0], colormap='bone'), n_iter)

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            # writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
            writer.writerow(losses_it)
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            errors = losses.avg
            error_names = ['total_loss', 'rec_loss',
                           'geometry_consistency_loss', 'fft_loss', 'ssim_loss']
            if args.with_vo:
                error_names.extend(
                    ['vo_loss', 'vo_photo_loss', 'vo_smooth_loss', 'vo_geometry_loss'])
            error_string = ', '.join('{} : {:.3f}'.format(name, error)
                                     for name, error in zip(error_names, errors))
            logger.train_writer.write(
                'Train: Batch time {} Data time {} '.format(batch_time, data_time, losses) + error_string)
        if i >= args.train_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate(args, val_loader, mask_net, pose_net, epoch, logger, warper, val_writer):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=5, precision=4)
    log_outputs = val_writer is not None
    w1, w2, w3, w4 = args.photo_loss_weight, args.geometry_consistency_weight, args.fft_loss_weight, args.ssim_loss_weight

    # switch to eval mode
    if args.with_masknet:
        mask_net.eval()
    pose_net.eval()

    all_poses = []
    all_inv_poses = []

    # Randomly choose n indices to log images
    rng = np.random.default_rng()
    log_ind = rng.integers(len(val_loader), size=1)

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)
        # intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_mask, ref_masks = None, [
            None for i in range(args.sequence_length-1)]
        if args.with_masknet:
            tgt_mask, ref_masks = compute_mask(mask_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        all_poses.append(poses)
        all_inv_poses.append(poses_inv)

        rec_loss, geometry_consistency_loss, fft_loss, ssim_loss, projected_imgs, projected_masks = warper.compute_db_loss(
            tgt_img, ref_imgs, tgt_mask, ref_masks, poses, poses_inv)

        rec_loss = w1*rec_loss
        geometry_consistency_loss = w2*geometry_consistency_loss
        fft_loss = w3*fft_loss
        ssim_loss = w4*ssim_loss
        loss = rec_loss + geometry_consistency_loss + fft_loss + ssim_loss

        if log_outputs and i in log_ind:
            # if epoch == 0:
            val_writer.add_image(
                'val/img/input', utils.tensor2array(tgt_img[0], colormap='bone'), epoch)
            val_writer.add_image(
                'val/img/warped_input', utils.tensor2array(projected_imgs[0][0], colormap='bone'), epoch)
            if args.with_masknet:
                val_writer.add_image(
                    'val/img/warped_mask', utils.tensor2array(projected_masks[0][0], colormap='bone'), epoch)
                val_writer.add_image(
                    'val/img/tgt_mask', utils.tensor2array(tgt_mask[0], colormap='bone'), epoch)

        losses.update([loss.item(), rec_loss.item(), geometry_consistency_loss.item(
        ), fft_loss.item(), ssim_loss.item()], args.batch_size)

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
        val_writer.add_histogram(
            'val/rot_pred-z', all_poses_t[..., 2], epoch)
        val_writer.add_histogram(
            'val/trans_pred-x', all_poses_t[..., 3], epoch)
        val_writer.add_histogram(
            'val/trans_pred-y', all_poses_t[..., 4], epoch)

    logger.valid_bar.update(args.val_size)

    errors = losses.avg
    error_names = ['total_loss', 'rec_loss',
                   'geometry_consistency_loss', 'fft_loss', 'ssim_loss']

    if args.gt_file is not None:
        ro_eval = RadarEvalOdom(args.gt_file, args.dataset)

        ate_bs_mean, ate_bs_std, ate_fs_mean, ate_fs_std, f_pred_xyz, f_pred = ro_eval.eval_ref_poses(
            all_poses, all_inv_poses, args.skip_frames)

        if log_outputs:
            # Plot and log aligned trajectory
            fig = utils.traj2Fig_withgt(
                f_pred_xyz.squeeze(), ro_eval.gt[:, :3, 3].squeeze())
            # fig2= utils.traj2Fig(f_pred[:,:3,3])
            val_writer.add_figure('val/fig/traj_aligned_pred', fig, epoch)
            # output_writers[0].add_figure('val/fig/traj_pred_full_aligned', fig2, epoch)

    else:
        b_pred_xyz, f_pred_xyz = getTraj(
            all_poses, all_inv_poses, args.skip_frames)

        if log_outputs:
            # Plot and log predicted trajectory
            b_fig = utils.traj2Fig(b_pred_xyz)
            f_fig = utils.traj2Fig(f_pred_xyz)
            val_writer.add_figure('val/fig/b_traj_pred', b_fig, epoch)
            val_writer.add_figure('val/fig/f_traj_pred', f_fig, epoch)

    return errors, error_names


def compute_mask(mask_net, tgt_img, ref_imgs):
    # tgt_mask = [mask for mask in mask_net(tgt_img)] # for multiple scale
    tgt_mask = mask_net(tgt_img)  # for single scale

    ref_masks = []
    for ref_img in ref_imgs:
        # ref_mask = [mask for mask in mask_net(ref_img)] # for multiple scale
        ref_mask = mask_net(ref_img)  # for multiple scale
        ref_masks.append(ref_mask)

    return tgt_mask, ref_masks


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_matches in ref_imgs:
        match_depths = []
        for ref_img in ref_matches:
            ref_depth = [1/disp for disp in disp_net(ref_img)]
            match_depths.append(ref_depth)
        ref_depths.append(match_depths)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return torch.stack(poses), torch.stack(poses_inv)


def compute_mono_pose_with_inv(pose_net, tgt_img, ref_imgs):
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


def mono_collate_fn(batch):
    b_tgt_img = torch.Tensor([input[0] for input in batch])
    b_ref_imgs = torch.Tensor([input[1] for input in batch])
    b_vo_tgt_img = [input[2] for input in batch]
    b_vo_ref_imgs = [input[3] for input in batch]

    return b_tgt_img, b_ref_imgs, b_vo_tgt_img, b_vo_ref_imgs


if __name__ == '__main__':
    main()
