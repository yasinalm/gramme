from __future__ import division
import torch
import torch.nn.functional as F

import conversions as tgm
import loss_ssim
import utils_warp as utils


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class MonoWarper(object):
    """Inverse warper class
    """

    def __init__(self, max_scales, with_auto_mask, dataset, with_ssim=True, with_mask=True, padding_mode='zeros'):
        global device
        self.dataset = dataset
        self.padding_mode = padding_mode
        self.with_ssim = with_ssim
        self.with_mask = with_mask
        self.with_auto_mask = with_auto_mask
        self.max_scales = max_scales
        if self.dataset == 'radiate':
            fx = 3.379191448899105e+02
            fy = 3.386957068549526e+02
            cx = 3.417366010946575e+02
            cy = 2.007359735313929e+02
            h, w = 376, 672
            scale_x = 640.0/w
            scale_y = 384.0/h
        elif self.dataset == 'robotcar':
            fx, fy, cx, cy = 964.828979, 964.828979, 643.788025, 484.407990
            h, w = 960, 1280
            scale_x = scale_y = 1.0
        self.pixel_coords_hom = set_id_grid(h, w)
        self.intrinsics = utils.camera_matrix(
            torch.Tensor([fx, fy, cx, cy]).to(device))  # [4,4]
        # Resize the intrinsics
        self.intrinsics[0] *= scale_x
        self.intrinsics[1] *= scale_y
        self.intrinsics_inv = self.intrinsics.inverse()  # [4,4]

    def cam2pixel(self, cam_coords, proj_c2p, dims):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H*W]
            proj_c2p: transformation matrix of cameras -- [B, 4, 4]
            dims: [img_height, img_width]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        h, w = dims

        pcoords = torch.matmul(proj_c2p, cam_coords)  # [B,4,N]

        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2*(X / Z)/(w-1) - 1
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
        if self.padding_mode == 'zeros':
            X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
            # make sure that no point in warped image is a combinaison of im and gray
            X_norm[X_mask] = 2
            Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
            Y_norm[Y_mask] = 2

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.reshape(-1, h, w, 2), Z.reshape(-1, 1, h, w)

    def pixel2cam(self, depth):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, 1, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u*v,1) cam coordinates in homogenous coordinates -- [B, 4, H*W]
        """
        # b, c, h, w = depth.size()
        # pixel_coords = set_id_grid(depth)
        cam_coords = torch.matmul(
            self.intrinsics_inv, torch.transpose(self.pixel_coords, 0, 1))  # [B,4,N]
        # cam_coords = torch.transpose(cam_coords, 1, 2)  # [B,N,4]
        # Convert from homogenous coordinates
        # cam_coords = tgm.convert_points_from_homogeneous(
        #     cam_coords)  # [B,N,3]
        # cam_coords = torch.transpose(cam_coords, 1, 2)  # [B,4,N]
        # cam_coords = cam_coords.reshape(b, 3, h, w)
        cam_coords[:, :3, :] = cam_coords[:, :3, :] * \
            depth.flatten(2)  # [B,3,N] * [B,1,N] = [B,3,N]
        return cam_coords  # [B,4,N]

    def inverse_warp(self, img, depth, ref_depth, pose):
        """
        Inverse warp a source image to the target image plane.
        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            depth: depth map of the target image -- [B, 1, H, W]
            ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
            pose: 6DoF pose parameters from target to source -- [B, 6]
            intrinsics: camera intrinsic matrix -- [B, 3, 3]
        Returns:
            projected_img: Source image warped to the target image plane
            valid_mask: Float array indicating point validity
            projected_depth: sampled depth from source image  
            computed_depth: computed depth of source image using the target depth
        """
        utils.check_sizes(img, 'img', 'B3HW')
        utils.check_sizes(depth, 'depth', 'B1HW')
        utils.check_sizes(ref_depth, 'ref_depth', 'B1HW')
        utils.check_sizes(pose, 'pose', 'B6')
        utils.check_sizes(self.intrinsics, 'intrinsics', 'B44')

        _, _, img_height, img_width = img.size()

        cam_coords = self.pixel2cam(depth)  # [B,3,H,W]

        # Convert 6 DoF pose to 4x4 transformation matrix
        # T*R in homogenous coordinates [B,4,4]
        pose_mat = tgm.rtvec_to_pose(pose)

        # Get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel = self.intrinsics @ pose_mat  # [B, 4, 4]

        # rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
        src_pixel_coords, computed_depth = self.cam2pixel(
            cam_coords, proj_cam_to_src_pixel, [img_height, img_width])  # [B,H,W,2]
        projected_img = F.grid_sample(
            img, src_pixel_coords, padding_mode=self.padding_mode, align_corners=False)

        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
        valid_mask = valid_points.unsqueeze(1).float()

        projected_depth = F.grid_sample(
            ref_depth, src_pixel_coords, padding_mode=self.padding_mode, align_corners=False)

        return projected_img, valid_mask, projected_depth, computed_depth

    def compute_photo_and_geometry_loss(
            self, tgt_img, ref_imgs, tgt_depth, ref_depths, poses, poses_inv):
        """Given a sequence of monocular frames, calculates triple-wise self-supervised photometric, smoothness and geometric losses.
        Converts a given sequence into mini target and source frames. For example, given a target and reference images 9 and [[7,8], [10,11]],
        we form a sequence [7,8,9,10,11]. We take triple mini-sequences and send them to the regular reconstruction, 
        e.g., [7,8,9], [8,9,10] and [9,10,11], where the middle element if the target and the others are source frames.

        Args:
            tgt_img (Tensor): Target image corresponding to the target radar frame.
            ref_imgs (List[List[Tensor]]): Reference images. First element is the list of previous frames and the second element list of next frames.
            intrinsics (Tensor): Intrinsic matrix of the monocular camera
            tgt_depth (Tensor): Target depth map of the `tgt_img`
            ref_depths (Lst[List[Tensor]]): Reference depths of `ref_imgs`
            poses (Tensor): Target to source pose values. poses[0] and poses[1] are the backward (tgt2previous) and forward (tgt2next) poses.
            poses_inv (Tensor): Source to target pose values.
            max_scales (int): Number of intermediate scales
            with_ssim (boolean): Whether to use SSIM
            with_mask (boolean): With mask or not
            with_auto_mask (boolean): With auto calculated mask or not
            padding_mode (str): Padding mode used in the image resampling

        Returns:
            (float, float, float): Photometric, smoothness and geoemtric losses.
        """

        ref_imgs_sq = ref_imgs[0] + [tgt_img] + \
            ref_imgs[1]  # [6,7,8,9,10,11,12]
        ref_depths_sq = ref_depths[0] + [tgt_depth] + \
            ref_depths[1]  # [6,7,8,9,10,11,12]
        # [p6-7, p7-8, p8-9, p9-10, p10-11, p11-12]
        poses_sq = poses_inv[0] + poses[1]
        # [p7-6, p8-7, p9-8, p10-9, p11-10, p12-11]
        poses_inv_sq = poses[0] + poses_inv[1]

        photo_loss = 0
        smooth_loss = 0
        geometry_loss = 0

        for i in range(1, len(ref_imgs_sq)-1):
            mini_tgt_img = ref_imgs_sq[i]  # [7]
            mini_ref_imgs = [ref_imgs_sq[i-1], ref_imgs_sq[i+1]]  # [6,8]
            mini_tgt_depth = ref_depths_sq[i]  # [7]
            mini_ref_depths = [ref_depths_sq[i-1], ref_depths_sq[i+1]]  # [6,8]
            mini_poses = [poses_inv_sq[i-1], poses_sq[i]]  # [p7-6, p7-8]
            mini_poses_inv = [poses_sq[i-1], poses_inv_sq[i]]  # [p6-7, p8-7]

            pl, sl, gl = self.compute_photo_and_geometry_loss_mini(
                mini_tgt_img, mini_ref_imgs, mini_tgt_depth, mini_ref_depths,
                mini_poses, mini_poses_inv)

            photo_loss += pl
            smooth_loss += sl
            geometry_loss += gl

        return photo_loss, smooth_loss, geometry_loss

    def compute_photo_and_geometry_loss_mini(
            self, tgt_img, ref_imgs, tgt_depth, ref_depths, poses, poses_inv):

        photo_loss = 0
        geometry_loss = 0

        num_scales = min(len(tgt_depth), self.max_scales)
        for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
            for s in range(num_scales):

                # # downsample img
                # b, _, h, w = tgt_depth[s].size()
                # downscale = tgt_img.size(2)/h
                # if s == 0:
                #     tgt_img_scaled = tgt_img
                #     ref_img_scaled = ref_img
                # else:
                #     tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
                #     ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
                # intrinsic_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
                # tgt_depth_scaled = tgt_depth[s]
                # ref_depth_scaled = ref_depth[s]

                # upsample depth
                b, _, h, w = tgt_img.size()
                tgt_img_scaled = tgt_img
                ref_img_scaled = ref_img
                # intrinsic_scaled = intrinsics
                if s == 0:
                    tgt_depth_scaled = tgt_depth[s]
                    ref_depth_scaled = ref_depth[s]
                else:
                    tgt_depth_scaled = F.interpolate(
                        tgt_depth[s], (h, w), mode='nearest')
                    ref_depth_scaled = F.interpolate(
                        ref_depth[s], (h, w), mode='nearest')

                photo_loss1, geometry_loss1 = self.compute_pairwise_loss(
                    tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose)
                photo_loss2, geometry_loss2 = self.compute_pairwise_loss(
                    ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv)

                photo_loss += (photo_loss1 + photo_loss2)
                geometry_loss += (geometry_loss1 + geometry_loss2)

        smooth_loss = utils.compute_smooth_loss(
            tgt_depth, tgt_img, ref_depths, ref_imgs)

        return photo_loss, smooth_loss, geometry_loss

    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose):

        ref_img_warped, valid_mask, projected_depth, computed_depth = self.inverse_warp(
            ref_img, tgt_depth, ref_depth, pose)

        diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

        diff_depth = ((computed_depth - projected_depth).abs() /
                      (computed_depth + projected_depth)).clamp(0, 1)

        if self.with_auto_mask == True:
            auto_mask = (diff_img.mean(dim=1, keepdim=True) < (
                tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
            valid_mask = auto_mask

        if self.with_ssim == True:
            ssim_map = loss_ssim.ssim(tgt_img, ref_img_warped)
            # diff_img = (0.90 * diff_img + 0.10 * ssim_map)
            # ssim_loss = mean_on_mask(ssim_map, valid_mask)  # ssim_map.mean()
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        if self.with_mask == True:
            weight_mask = (1 - diff_depth)
            diff_img = diff_img * weight_mask

        # compute all loss
        reconstruction_loss = mean_on_mask(diff_img, valid_mask)
        geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

        return reconstruction_loss, geometry_consistency_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


def set_id_grid(h, w):
    global device
    # b, h, w = depth.size()
    ranges_x = torch.arange(0, h)  # .type(depth.dtype)
    ranges_y = torch.arange(0, w)  # .type(depth.dtype)

    x, y = torch.meshgrid(ranges_x, ranges_y)
    x = torch.flatten(x)
    y = torch.flatten(y)

    # [3,N] Augment with zero z column
    xy = torch.vstack((x, y, torch.zeros_like(x)))
    xy = torch.transpose(xy, 0, 1)  # [N,3]
    pixel_coords_hom = tgm.convert_points_to_homogeneous(
        xy).to(device)  # [N,4]

    # pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    # return pixel_coords_hom.expand(b, -1, -1)  # [B, N, 4]
    return pixel_coords_hom  # [N, 4]
