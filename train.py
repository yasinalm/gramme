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

import models

import custom_transforms
from utils import tensor2array, save_checkpoint, traj2Fig, traj2Img
from datasets.sequence_folders import SequenceFolder
# from datasets.pair_folders import PairFolder
from inverse_warp import Warper
from radar_eval.eval_utils import getTraj, RadarEvalOdom
# from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning) # Supress UserWarning from grid_sample

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
# parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--skip-frames', type=int, metavar='N', help='gap between frames', default=5)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--train-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--val-size', default=0, type=int, metavar='N', help='manual validation size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
# parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-f', '--fft-loss-weight', type=float, help='weight for FFT loss', metavar='W', default=3e-4)
parser.add_argument('-s', '--ssim-loss-weight', type=float, help='weight for SSIM loss', metavar='W', default=1)
# parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
# parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
# parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
# parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=0, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['hand', 'robotcar', 'radiate'], default='hand', help='the dataset to train')
# parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--gt-file', metavar='DIR', help='path to ground truth validation file')
parser.add_argument('--gt-type', type=str, choices=['kitti', 'xyz'], default='xyz', help='GT format')
parser.add_argument('--range-res', type=float, help='Range resolution of FMCW radar in meters', metavar='W', default=0.0977)
parser.add_argument('--angle-res', type=float, help='Angular azimuth resolution of FMCW radar in radians', metavar='W', default=1.0)
parser.add_argument('--cart-res', type=float, help='Cartesian resolution of FMCW radar in meters/pixel', metavar='W', default=0.25)
parser.add_argument('--cart-pixels', type=int, help='Cartesian size in pixels (used for both height and width)', metavar='W', default=501)
# parser.add_argument('--num-range-bins', type=int, help='Number of ADC samples (range bins)', metavar='W', default=256)
# parser.add_argument('--num-angle-bins', type=int, help='Number of angle bins', metavar='W', default=64)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    output_writers = []
    if args.log_output:
        # Keep n different writers to save n images.
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

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
    ds_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])

    

    print("=> fetching scenes in '{}'".format(args.data))
    # if args.folder_type == 'sequence':
    train_set = SequenceFolder(
        args.data,
        transform=ds_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset,
        cart_resolution=args.cart_res,
        cart_pixels = args.cart_pixels,
        rangeResolutionsInMeter=args.range_res,
        angleResolutionInRad = args.angle_res
        )
    # else:
    #     train_set = PairFolder(
    #         args.data,
    #         seed=args.seed,
    #         train=True,
    #         transform=train_transform
    #     )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    val_set = SequenceFolder(
        args.data,
        transform=ds_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset,
        cart_resolution=args.cart_res,
        cart_pixels = args.cart_pixels,
        rangeResolutionsInMeter=args.range_res,
        angleResolutionInRad = args.angle_res
    )
    if args.with_gt:
        if args.gt_file:
            vo_eval = RadarEvalOdom(args.gt_file, args.dataset)
        else:
            warnings.warn('with-gt is set but no ground truth validation file is provided with val-gt arg! with-gt will be ignored.')
            args.with_gt = False
        
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.train_size == 0:
        args.train_size = len(train_loader)
    print('Epoch size: ', args.train_size)
    if args.val_size == 0:
        args.val_size = len(val_loader)
    print('Validation size: ', args.val_size)

    # create model
    print("=> creating loss object")
    warper = Warper(args.range_res, args.angle_res, args.with_auto_mask, args.cart_res, args.cart_pixels, args.dataset, args.padding_mode)

    # create model
    print("=> creating model")
    # disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(args.resnet_layers, args.with_pretrain).to(device)

    # load parameters
    # if args.pretrained_disp:
    #     print("=> using pre-trained weights for DispResNet")
    #     weights = torch.load(args.pretrained_disp)
    #     disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    # disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        # {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=args.train_size, valid_size=args.val_size)
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        # train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.train_size, logger, training_writer, warper)
        train_loss = train(args, train_loader, pose_net, optimizer, args.train_size, logger, training_writer, warper)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, pose_net, vo_eval, epoch, logger, warper, output_writers)            
        else:
            errors, error_names = validate_without_gt(args, val_loader, pose_net, epoch, logger, warper, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar('val/'+name, error, epoch)    

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
            #     'epoch': epoch + 1,
            #     'state_dict': disp_net.module.state_dict()
            # }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, pose_net, optimizer, train_size, logger, train_writer, warper):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.fft_loss_weight, args.ssim_loss_weight

    best_error = 9.0e6

    # switch to train mode
    # disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)

        # compute output
        # tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        rec_loss, fft_loss, ssim_loss, _ = warper.compute_db_loss(tgt_img, ref_imgs, poses, poses_inv)


        # loss_1, loss_3 = warper.compute_db_loss(tgt_img, ref_imgs, poses, poses_inv)
        # loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        rec_loss = w1*rec_loss
        fft_loss = w2*fft_loss
        ssim_loss = w3*ssim_loss
        loss = rec_loss + fft_loss + ssim_loss

        if log_losses:
            # train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            # train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            # train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('train/photometric_error', rec_loss.item(), n_iter)
            train_writer.add_scalar('train/fft_loss', fft_loss.item(), n_iter)
            train_writer.add_scalar('train/ssim_loss', ssim_loss.item(), n_iter)
            train_writer.add_scalar('train/total_loss', loss.item(), n_iter)

            # train_writer.add_histogram('train/rot_pred-x', poses[...,0], n_iter)
            # train_writer.add_histogram('train/rot_pred-y', poses[...,1], n_iter)
            train_writer.add_histogram('train/rot_pred-z', poses[...,2], n_iter)
            train_writer.add_histogram('train/trans_pred-x', poses[...,3], n_iter)
            train_writer.add_histogram('train/trans_pred-y', poses[...,4], n_iter)
            # train_writer.add_histogram('train/trans_pred-z', poses[...,5], n_iter)

        # record loss and EPE
        # TODO: Log losses separately
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i>0 and i%1000 == 0:
                # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            decisive_error = loss.item()

            # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            save_checkpoint(
                args.save_path, {
                #     'epoch': epoch + 1,
                #     'state_dict': disp_net.module.state_dict()
                # }, {
                    'n_iter': n_iter + 1,
                    'state_dict': pose_net.module.state_dict()
                },
                is_best)

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            # writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
            writer.writerow([loss.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= train_size - 1:
            break

        n_iter += 1

    return losses.avg[0]
    

@torch.no_grad()
def validate_without_gt(args, val_loader, pose_net, epoch, logger, warper, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0
    w1, w2, w3 = args.photo_loss_weight, args.fft_loss_weight, args.ssim_loss_weight

    # switch to evaluate mode
    # disp_net.eval()
    pose_net.eval()

    all_poses = []
    all_inv_poses = []

    # Randomly choose 3 indices to log images
    rng = np.random.default_rng()
    log_ind = rng.integers(len(val_loader), size=3)
    k=0 # writer counter
    
    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)
        # intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        all_poses.append(poses)
        all_inv_poses.append(poses_inv)

        rec_loss, fft_loss, ssim_loss, projected_imgs = warper.compute_db_loss(tgt_img, ref_imgs, poses, poses_inv)

        rec_loss = w1*rec_loss
        fft_loss = w2*fft_loss
        ssim_loss = w3*ssim_loss
        loss = rec_loss + fft_loss + ssim_loss

        if log_outputs and i in log_ind:
            # if epoch == 0:
            output_writers[k].add_image('val/img/input', 
                                        tensor2array(tgt_img[0], colormap='bone'), 
                                        epoch)

            output_writers[k].add_image('val/img/warped_input',
                                        tensor2array(projected_imgs[0][0], colormap='bone'),
                                        epoch)
            k = k+1

        loss = loss.item()
        losses.update([loss, rec_loss.item(), fft_loss.item(), ssim_loss.item()])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
        if i >= args.val_size - 1:
            break

    b_pred_xyz, f_pred_xyz = getTraj(all_poses, all_inv_poses, args.skip_frames)

    if log_outputs:
        # Plot and log predicted trajectory
        b_fig = traj2Fig(b_pred_xyz)
        f_fig = traj2Fig(f_pred_xyz)
        output_writers[0].add_figure('val/fig/b_traj_pred', b_fig, epoch)
        output_writers[0].add_figure('val/fig/f_traj_pred', f_fig, epoch)
        # Log predicted relative poses in histograms
        all_poses_t = torch.cat(all_poses, 1) # [seq_length, N, 6]
        output_writers[0].add_histogram('val/rot_pred-z', all_poses_t[...,2], epoch)
        output_writers[0].add_histogram('val/trans_pred-x', all_poses_t[...,3], epoch)
        output_writers[0].add_histogram('val/trans_pred-y', all_poses_t[...,4], epoch)

    logger.valid_bar.update(args.val_size)

    errors = losses.avg
    error_names = ['total_loss', 'rec_loss', 'fft_loss', 'ssim_loss']
    return errors, error_names


@torch.no_grad()
def validate_with_gt(args, val_loader, pose_net, vo_eval, epoch, logger, warper, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = len(output_writers) > 0
    w1, w2, w3 = args.photo_loss_weight, args.fft_loss_weight, args.ssim_loss_weight

    # switch to evaluate mode
    # disp_net.eval()
    pose_net.eval()

    all_poses = []
    all_inv_poses = []

    # Randomly choose 3 indices to log images
    rng = np.random.default_rng()
    log_ind = rng.integers(len(val_loader), size=3)
    k=0 # writer counter

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # intrinsics = intrinsics.to(device)
        # intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        all_poses.append(poses)
        all_inv_poses.append(poses_inv)

        rec_loss, fft_loss, ssim_loss, projected_imgs = warper.compute_db_loss(tgt_img, ref_imgs, poses, poses_inv)

        rec_loss = w1*rec_loss
        fft_loss = w2*fft_loss
        ssim_loss = w3*ssim_loss
        loss = rec_loss + fft_loss + ssim_loss

        if log_outputs and i in log_ind:
            # if epoch == 0:
            output_writers[k].add_image('val/img/input', 
                                        tensor2array(tgt_img[0], colormap='bone'), 
                                        epoch)

            output_writers[k].add_image('val/img/warped_input',
                                        tensor2array(projected_imgs[0][0], colormap='bone'),
                                        epoch)
            k = k+1

        loss = loss.item()
        losses.update([loss, rec_loss.item(), fft_loss.item(), ssim_loss.loss()])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
        if i >= args.val_size - 1:
            break
    ate_bs_mean, ate_bs_std, ate_fs_mean, ate_fs_std, f_pred_xyz = vo_eval.eval_ref_poses(all_poses, all_inv_poses, 
    args.skip_frames)

    if log_outputs:
        # Plot and log aligned trajectory
        fig = traj2Fig(f_pred_xyz)
        output_writers[0].add_figure('val/fig/traj_pred', fig, epoch)
        # Log predicted relative poses in histograms
        all_poses_t = torch.cat(all_poses, 1) # [seq_length, N, 6]
        output_writers[0].add_histogram('val/rot_pred-z', all_poses_t[...,2], epoch)
        output_writers[0].add_histogram('val/trans_pred-x', all_poses_t[...,3], epoch)
        output_writers[0].add_histogram('val/trans_pred-y', all_poses_t[...,4], epoch)


    logger.valid_bar.update(args.val_size)

    errors = losses.avg+[ate_bs_mean.item(), ate_bs_std.item(), ate_fs_mean.item(), ate_fs_std.item()]
    error_names = ['total_loss', 'rec_loss', 'fft_loss', 'ssim_loss']+['ate_bs_mean', 'ate_bs_std', 'ate_fs_mean', 'ate_fs_std']
    return errors, error_names


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return torch.stack(poses), torch.stack(poses_inv)


if __name__ == '__main__':
    main()
