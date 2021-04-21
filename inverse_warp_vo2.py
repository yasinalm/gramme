from __future__ import division
import torch
import torch.nn.functional as F
from torch import nn

pixel_coords = None

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


class MonoWarper(object):
    """Inverse warper class
    """

    def __init__(self, max_scales, dataset, batch_size, with_auto_mask=True, with_ssim=True, with_mask=True, padding_mode='zeros'):
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
            h, w = 384, 640
        elif self.dataset == 'robotcar':
            # fx, fy, cx, cy = 964.828979, 964.828979, 643.788025, 484.407990
            fx, fy, cx, cy = 983.044006, 983.044006, 643.646973, 493.378998
            h, w = 960, 1280
            scale_x = 640.0/w
            scale_y = 384.0/h
            h, w = 384, 640
        #self.pixel_coords_hom = set_id_grid(h, w)
        self.intrinsics = torch.eye(3, device=device, dtype=torch.float) + 1e-6
        # k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4
        # fill output with pinhole values
        self.intrinsics[..., 0, 0] = fx
        self.intrinsics[..., 0, 2] = cx
        self.intrinsics[..., 1, 1] = fy
        self.intrinsics[..., 1, 2] = cy
        # Resize the intrinsics
        self.intrinsics[0] *= scale_x
        self.intrinsics[1] *= scale_y
        self.intrinsics_inv = self.intrinsics.inverse()  # [4,4]
        self.intrinsics = self.intrinsics.repeat(batch_size, 1, 1)
        self.intrinsics_inv = self.intrinsics_inv.repeat(batch_size, 1, 1)

    def pixel2cam(self, depth):
        global pixel_coords
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        b, h, w = depth.size()
        if (pixel_coords is None) or pixel_coords.size(2) < h:
            set_id_grid(depth)
        current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
            b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = (self.intrinsics_inv @
                      current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)

    def cam2pixel(self, cam_coords, proj_c2p_rot, proj_c2p_tr):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2*(X / Z)/(w-1) - 1
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.reshape(b, h, w, 2)

    def inverse_warp(self, img, depth, pose, rotation_mode='euler'):
        """
        Inverse warp a source image to the target image plane.
        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            depth: depth map of the target image -- [B, H, W]
            pose: 6DoF pose parameters from target to source -- [B, 6]
            intrinsics: camera intrinsic matrix -- [B, 3, 3]
        Returns:
            projected_img: Source image warped to the target image plane
            valid_points: Boolean array indicating point validity
        """
        check_sizes(img, 'img', 'B3HW')
        check_sizes(depth, 'depth', 'BHW')
        check_sizes(pose, 'pose', 'B6')
        check_sizes(self.intrinsics, 'intrinsics', 'B33')

        batch_size, _, img_height, img_width = img.size()

        cam_coords = self.pixel2cam(depth)  # [B,3,H,W]

        pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

        # Get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel = self.intrinsics @ pose_mat  # [B, 3, 4]

        rot, tr = proj_cam_to_src_pixel[:, :,
                                        :3], proj_cam_to_src_pixel[:, :, -1:]
        src_pixel_coords = self.cam2pixel(
            cam_coords, rot, tr)  # [B,H,W,2]
        projected_img = F.grid_sample(
            img, src_pixel_coords, padding_mode=self.padding_mode)

        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

        return projected_img, valid_points

    def cam2pixel2(self, cam_coords, proj_c2p_rot, proj_c2p_tr):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
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
        return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)

    def inverse_warp2(self, img, depth, ref_depth, pose):
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
        check_sizes(img, 'img', 'B3HW')
        check_sizes(depth, 'depth', 'B1HW')
        check_sizes(ref_depth, 'ref_depth', 'B1HW')
        check_sizes(pose, 'pose', 'B6')
        check_sizes(self.intrinsics, 'intrinsics', 'B33')

        batch_size, _, img_height, img_width = img.size()

        cam_coords = self.pixel2cam(depth.squeeze(
            1))  # [B,3,H,W]

        pose_mat = pose_vec2mat(pose)  # [B,3,4]

        # Get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel = self.intrinsics @ pose_mat  # [B, 3, 4]

        rot, tr = proj_cam_to_src_pixel[:, :,
                                        :3], proj_cam_to_src_pixel[:, :, -1:]
        src_pixel_coords, computed_depth = self.cam2pixel2(
            cam_coords, rot, tr)  # [B,H,W,2]
        projected_img = F.grid_sample(
            img, src_pixel_coords, padding_mode=self.padding_mode, align_corners=False)

        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
        valid_mask = valid_points.unsqueeze(1).float()

        projected_depth = F.grid_sample(
            ref_depth, src_pixel_coords, padding_mode=self.padding_mode, align_corners=False)

        return projected_img, valid_mask, projected_depth, computed_depth

    # photometric loss
    # geometry consistency loss

    def compute_photo_and_geometry_loss(self, tgt_img, ref_imgs, tgt_depth, ref_depths, poses, poses_inv):

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
                #intrinsic_scaled = self.intrinsics
                if s == 0:
                    tgt_depth_scaled = tgt_depth[s]
                    ref_depth_scaled = ref_depth[s]
                else:
                    tgt_depth_scaled = F.interpolate(
                        tgt_depth[s], (h, w), mode='nearest')
                    ref_depth_scaled = F.interpolate(
                        ref_depth[s], (h, w), mode='nearest')

                photo_loss1, geometry_loss1, ref_img_warped, valid_mask = self.compute_pairwise_loss(
                    tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose)
                photo_loss2, geometry_loss2, _, _ = self.compute_pairwise_loss(
                    ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv)

                photo_loss += (photo_loss1 + photo_loss2)
                geometry_loss += (geometry_loss1 + geometry_loss2)

        smooth_loss = self.compute_smooth_loss(
            tgt_depth, tgt_img, ref_depths, ref_imgs)

        ssim_loss = torch.Tensor([0]).to(device)

        return photo_loss, smooth_loss, geometry_loss, ssim_loss, ref_img_warped, valid_mask

    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose):
        global device

        ref_img_warped, valid_mask, projected_depth, computed_depth = self.inverse_warp2(
            ref_img, tgt_depth, ref_depth, pose)

        diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

        diff_depth = ((computed_depth - projected_depth).abs() /
                      (computed_depth + projected_depth)).clamp(0, 1)

        if self.with_auto_mask == True:
            auto_mask = (diff_img.mean(dim=1, keepdim=True) < (
                tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
            valid_mask = auto_mask

        if self.with_ssim == True:
            ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        if self.with_mask == True:
            weight_mask = (1 - diff_depth)
            diff_img = diff_img * weight_mask

        # compute all loss
        reconstruction_loss = self.mean_on_mask(diff_img, valid_mask)
        geometry_consistency_loss = self.mean_on_mask(diff_depth, valid_mask)

        return reconstruction_loss, geometry_consistency_loss, ref_img_warped, valid_mask

    # compute mean value given a binary mask

    def mean_on_mask(self, diff, valid_mask):
        global device
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 50000:
            mean_value = (diff * mask).sum() / mask.sum()
        else:
            mean_value = torch.tensor(0).float().to(device)
            #mean_value = diff.sum() / (mask.sum()+1e-12)
        return mean_value

    def compute_smooth_loss(self, tgt_depth, tgt_img, ref_depths, ref_imgs):
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

        loss = get_smooth_loss(tgt_depth[0], tgt_img)

        for ref_depth, ref_img in zip(ref_depths, ref_imgs):
            loss += get_smooth_loss(ref_depth[0], ref_img)

        return loss
