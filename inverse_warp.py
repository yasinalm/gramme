from __future__ import division
import torch
import torch.nn.functional as F
import torch.fft

import conversions as tgm
import loss_ssim

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class Warper(object):
    """Inverse warper class
    """

    def __init__(self, with_auto_mask, cart_resolution, cart_pixels, dataset, padding_mode='zeros'):
        # RF params
        # rangeResolutionsInMeter = 0.0977
        # # dopplerResolutionMps = 0.0951
        # numRangeBins = 256
        # # numDopplerBins = 128
        # num_angle_bins = 64 # our choice
        self.cart_resolution = cart_resolution
        self.cart_pixels = cart_pixels
        # self.rangeResolutionsInMeter=rangeResolutionsInMeter
        # self.angleResolutionInRad = angleResolutionInRad
        # self.numRangeBins=numRangeBins
        # self.num_angle_bins=num_angle_bins
        self.with_auto_mask = with_auto_mask
        self.padding_mode = padding_mode
        self.dataset = dataset

        if self.dataset == 'hand':
            ranges_x = torch.arange(self.cart_pixels//2)
            ranges_y = torch.arange(self.cart_pixels)-self.cart_pixels//2
        else:
            ranges_x = torch.arange(self.cart_pixels)-self.cart_pixels//2
            ranges_y = torch.arange(self.cart_pixels)-self.cart_pixels//2

        ranges_x = ranges_x*self.cart_resolution
        ranges_y = ranges_y*self.cart_resolution

        x, y = torch.meshgrid(ranges_x, ranges_y)
        x = torch.flatten(x)
        y = torch.flatten(y)

        # [3,N] Augment with zero z column
        xy = torch.vstack((x, y, torch.zeros_like(x)))
        xy = torch.transpose(xy, 0, 1)  # [N,3]
        self.xy_hom = tgm.convert_points_to_homogeneous(xy).to(device)  # [N,4]

    def radar2pixel_cart(self, pose_mat):
        """Transform coordinates in the source frame to the target frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        # Transform points
        tformed_xy_hom = torch.matmul(
            pose_mat, torch.transpose(self.xy_hom, 0, 1))  # [B,4,N]
        tformed_xy_hom = torch.transpose(tformed_xy_hom, 1, 2)  # [B,N,4]
        # Convert from homogenous coordinates
        tformed_xy = tgm.convert_points_from_homogeneous(
            tformed_xy_hom)  # [B,N,3]

        X = tformed_xy[:, :, 0]  # theta_tformed # [B,N]
        Y = tformed_xy[:, :, 1]  # [B,N]

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        if self.dataset == 'hand':
            # X and Y are in [0, self.cart_pixels*self.cart_resolution] meters.
            w = (self.cart_pixels//2)*self.cart_resolution
            h = (self.cart_pixels//2)*self.cart_resolution
            X_norm = 2*X/w - 1
            Y_norm = Y/h  # [B, H*W]
        else:
            # X and Y are in [-(self.cart_pixels//2)*self.cart_resolution, (self.cart_pixels//2)*self.cart_resolution] meters.
            w = (self.cart_pixels//2)*self.cart_resolution
            h = (self.cart_pixels//2)*self.cart_resolution
            X_norm = X/w
            Y_norm = Y/h  # [B, H*W]

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        pixel_coords = pixel_coords.reshape(
            self.b, self.w, self.h, 2)  # [B, W, H, 2]
        pixel_coords = pixel_coords.transpose(1, 2)  # [B, H, W, 2]

        return pixel_coords

    def inverse_warp_fft_cart(self, img, mask, pose):
        """
        Inverse warp a source radar frame to the target radar plane.
        H: Number of ADC samples (or Doppler bins)
        W: Number of angle bins
        Args:
            img: the source radar frame image (where to sample pixels) -- [B, H, W]
            pose: 6DoF pose parameters from target to source -- [B, 6]
        Returns:
            projected_img: Source image warped to the target image plane
            valid_points: Boolean array indicating point validity
        """
        check_sizes(img, 'img', 'B1HW')
        check_sizes(pose, 'pose', 'B6')

        self.b, self.c, self.h, self.w = img.size()

        assert self.h == self.cart_pixels  # self.numRangeBins
        if self.dataset == 'hand':
            assert self.w == self.cart_pixels//2  # self.num_angle_bins
        else:
            assert self.w == self.cart_pixels

        # if (xy_hom is None) or xy_hom.size(1) < 4:
        #     set_radar_grid()

        # Convert 6 DoF pose to 4x4 transformation matrix
        # T*R in homogenous coordinates [B,4,4]
        pose_mat = tgm.rtvec_to_pose(pose)

        # src_pixel_coords = pixel_coords.reshape(b, h, w, 2)
        src_pixel_coords = self.radar2pixel_cart(pose_mat)  # [B,H,W,2]

        projected_img = F.grid_sample(
            img, src_pixel_coords, padding_mode=self.padding_mode)
        projected_mask = None
        if mask is not None:
            projected_mask = F.grid_sample(
                mask, src_pixel_coords, padding_mode=self.padding_mode)

        # calculate mask values for each tformed_xy coordinates to match the target xy
        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1  # [B,H,W]
        valid_points = torch.unsqueeze(valid_points, 1)  # [B,1,H,W]

        return projected_img, projected_mask, valid_points

    # TODO: num_scales eklenebilir buraya.
    # TODO: ref_masks kullanmiyoruz, kullanmak icin mantikli bir yol koy.
    # decibels loss
    def compute_db_loss(self, tgt_img, ref_imgs, tgt_mask, ref_masks, poses, poses_inv):

        rec_loss = 0
        geometry_consistency_loss = 0
        fft_loss = 0
        ssim_loss = 0
        projected_imgs = ref_imgs
        projected_masks = ref_masks

        for i, (ref_img, ref_mask, pose, pose_inv) in enumerate(zip(ref_imgs, ref_masks, poses, poses_inv)):

            rec_loss1, geometry_consistency_loss1, fft_loss1, ssim_loss1, projected_img, projected_mask = self.compute_pairwise_loss(
                tgt_img, ref_img, tgt_mask, ref_mask, pose)
            rec_loss2, geometry_consistency_loss2, fft_loss2, ssim_loss2, _, _ = self.compute_pairwise_loss(
                ref_img, tgt_img, ref_mask, tgt_mask, pose_inv)

            rec_loss += (rec_loss1 + rec_loss2)
            geometry_consistency_loss += (geometry_consistency_loss1 +
                                          geometry_consistency_loss2)
            fft_loss += (fft_loss1 + fft_loss2)
            ssim_loss += (ssim_loss1 + ssim_loss2)
            projected_imgs[i] = projected_img
            projected_masks[i] = projected_mask

        # fft_loss = compute_smooth_loss(tgt_mask, tgt_img, ref_masks, ref_imgs)

        return rec_loss, geometry_consistency_loss, fft_loss, ssim_loss, projected_imgs, projected_masks

    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_mask, ref_mask, pose):

        ref_img_warped, projected_mask, valid_mask = self.inverse_warp_fft_cart(
            ref_img, ref_mask, pose)

        if tgt_mask is None:
            diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
        else:
            diff_img = (tgt_img - ref_img_warped *
                        projected_mask).abs().clamp(0, 1)
            # / (computed_depth + projected_depth)).clamp(0, 1)
            # diff_mask = ((tgt_mask - projected_mask).abs()).clamp(0, 1)
            diff_mask = ((tgt_img*tgt_mask - ref_img_warped *
                          projected_mask).abs()).clamp(0, 1)

        if self.with_auto_mask == True:
            auto_mask = (diff_img < (tgt_img - ref_img).abs()
                         ).float()  # [B,1,H,W]
            valid_mask = auto_mask * valid_mask  # element-wise # [B,1,H,W]

        # ssim_loss = loss_ssim.ssim(tgt_img, ref_img_warped, valid_mask)
        ssim_map = loss_ssim.ssim(tgt_img, ref_img_warped)
        # diff_img = (0.90 * diff_img + 0.10 * ssim_map)
        ssim_loss = mean_on_mask(ssim_map, valid_mask)  # ssim_map.mean()

        # # if with_mask == True:
        # weight_mask = (1 - diff_mask)
        # # if tgt_mask is not None:
        # #     weight_mask = weight_mask*tgt_mask
        # diff_img = diff_img * weight_mask

        # compute all loss
        # geometry_consistency_loss = diff_mask.sum() / diff_mask.numel()

        # ssim_map = loss_ssim.ssim(tgt_mask, projected_mask)
        # ssim_loss += mean_on_mask(ssim_map, valid_mask)  # ssim_map.mean()

        reconstruction_loss = mean_on_mask(diff_img, valid_mask)
        if tgt_mask is None:
            geometry_consistency_loss = torch.Tensor([0]).to(device)
        else:
            geometry_consistency_loss = mean_on_mask(diff_mask, valid_mask)
            geometry_consistency_loss += mean_on_mask(diff_img, tgt_mask)

        # fft_loss = fft_rec_loss2(tgt_img, ref_img_warped, valid_mask)
        fft_loss = torch.Tensor([0]).to(device)  # şimdilik disable

        return reconstruction_loss, geometry_consistency_loss, fft_loss, ssim_loss, ref_img_warped, projected_mask


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

    loss = get_smooth_loss(tgt_depth, tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth, ref_img)

    return loss

# compute mean value given a binary mask


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    thr_mask = diff.numel()//3  # at least a third of the input must be valid
    # thr_mask = 6e4*diff.shape[0]
    if mask.sum() > thr_mask:
        mask_diff = diff * mask
        l1 = mask_diff.sum() / mask.sum()
        # l2 = mask_diff.square().sum() / mask.sum()
    else:
        # l1 = torch.tensor(1e3).float().to(device)
        # l2 = torch.tensor(0).float().to(device)
        mask_diff = diff
        l1 = mask_diff.sum() / mask.sum()

    l = l1  # +l2
    return l


def fft_frame(img):
    """Calculate FFT of a given image.

    Args:
        img (torch.Tensor): Image tensor. Shape [B,C,H,W]

    Returns:
        torch.Tensor: Amplitude and phase of FFT result. Shape [B,C,H,W]
    """

    fft_im = torch.fft.rfftn(
        img, s=img.shape[-2:], dim=[-2, -1], norm="forward")  # [B,C,H,W,2]
    fft_amp = fft_im[..., 0]**2 + fft_im[..., 1]**2
    fft_amp = torch.sqrt(fft_amp)  # this is the amplitude [B,C,H,W]
    # this is the phase [B,C,H,W]
    fft_pha = torch.atan2(fft_im[..., 1], fft_im[..., 0])

    return fft_amp, fft_pha


def fft_rec_loss(tgt, src, mask):
    """Calculate FFT loss between pair of images. The loss is masked by `mask`. FFT loss is calculated on GPU and, thus, very fast. We calculate phase and amplitude L1 losses.

    Args:
        tgt (torch.Tensor): Target image. Shape [B,C,H,W]
        src (torch.Tensor): Source image. Shape [B,C,H,W]
        mask (torch.Tensor): Boolean mask for valid points

    Returns:
        torch.Tensor: Scalar sum of phase and amplitude losses.
    """

    tgt_amp, tgt_pha = fft_frame(tgt)
    src_amp, src_pha = fft_frame(src)

    diff_amp = tgt_amp - src_amp
    # diff_amp = diff_amp*mask
    l_amp = diff_amp.abs().sum() / mask.sum()
    diff_pha = tgt_pha - src_pha
    # diff_pha = diff_pha*mask
    l_pha = diff_pha.abs().sum() / mask.sum()

    l = l_amp+l_pha
    return l


def fft_rec_loss2(tgt, src, mask):
    """Calculate FFT loss between pair of images. The loss is masked by `mask`. FFT loss is calculated on GPU and, thus, very fast. We calculate phase and amplitude L1 losses.

    Args:
        tgt (torch.Tensor): Target image. Shape [B,C,H,W]
        src (torch.Tensor): Source image. Shape [B,C,H,W]
        mask (torch.Tensor): Boolean mask for valid points

    Returns:
        torch.Tensor: Scalar sum of phase and amplitude losses.
    """

    # Apply 2D FFT on the last two dimensions, e.g., [H,W] channels.
    fft_tgt = torch.fft.rfftn(
        tgt, s=tgt.shape[-2:], dim=[-2, -1], norm="forward")  # [B,C,H,W,2]
    fft_src = torch.fft.rfftn(
        src, s=tgt.shape[-2:], dim=[-2, -1], norm="forward")  # [B,C,H,W,2]
    # fft_diff = torch.fft.rfftn(tgt-src, s=tgt.shape[-2:], dim=[-2,-1], norm="ortho") # [B,C,H,W,2]
    # fft_diff = fft_tgt - fft_src

    # fft_diff = torch.view_as_real(fft_diff)
    # mag_diff = fft_diff[...,0].abs().sum() #20*torch.log10(fft_diff[...,0]) # mag2db
    # pha_diff = fft_diff[...,1].abs().sum()

    # Convolution over pixels is FFT on frequencies.
    # We may find a more clever way.
    # fft_conv = fft_tgt*fft_src
    # inv_fft_conv = torch.fft.irfftn(fft_conv, s=tgt.shape[-2:], dim=[-2,-1], norm="forward") # [B,C,H,W,2]
    # mask_diff = fft_diff #fft_tgt-fft_src
    # print(mask_diff.shape)
    # print(mask.shape)
    # mask_diff = mask_diff * mask

    # l = 20*torch.log10(fft_diff.abs()) # mag2db 20*log10(abs(complex))

    # Derivative for angle is not implemented yet.
    # pha_diff = torch.abs(fft_tgt.angle() - fft_src.angle())
    mag_diff = torch.abs(fft_tgt.abs() - fft_src.abs())

    fft_tgt = torch.view_as_real(fft_tgt)
    fft_src = torch.view_as_real(fft_src)
    pha_tgt = torch.atan2(fft_tgt[..., 1], fft_tgt[..., 0])
    pha_src = torch.atan2(fft_src[..., 1], fft_src[..., 0])
    # mag_tgt = torch.sqrt(fft_tgt[...,1]**2 + fft_tgt[...,0]**2)
    # mag_src = torch.sqrt(fft_src[...,1]**2 + fft_src[...,0]**2)

    pha_diff = torch.abs(pha_tgt-pha_src)
    # mag_diff = torch.abs(mag_tgt - mag_src)

    # print(pha_diff.sum())
    # print(mag_diff.sum())

    l = 1e-4*mag_diff.sum() + pha_diff.sum()

    return l


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
