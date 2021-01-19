from __future__ import division
import torch
import torch.nn.functional as F

import conversions as tgm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Warper(object):
    """Inverse warper class
    """

    # xy_hom = None

    def __init__(self, rangeResolutionsInMeter, numRangeBins, num_angle_bins,
    with_auto_mask, padding_mode='zeros'):
        # RF params
        # rangeResolutionsInMeter = 0.0977
        # # dopplerResolutionMps = 0.0951
        # numRangeBins = 256
        # # numDopplerBins = 128
        # num_angle_bins = 64 # our choice
        self.rangeResolutionsInMeter=rangeResolutionsInMeter
        self.numRangeBins=numRangeBins
        self.num_angle_bins=num_angle_bins
        self.with_auto_mask=with_auto_mask
        self.padding_mode=padding_mode

        azimuths = torch.arange(num_angle_bins)
        azimuths = (azimuths - (num_angle_bins / 2))
        ranges = torch.arange(numRangeBins)
        ranges *= rangeResolutionsInMeter

        az_grid, range_grid = torch.meshgrid(azimuths, ranges)
        x, y = pol2cart(torch.deg2rad(az_grid), range_grid)
        x=torch.flatten(x)
        y=torch.flatten(y)

        xy = torch.vstack((x, y, torch.zeros_like(x)))  # Nx3 Augment with zero z column
        self.xy_hom = tgm.convert_points_to_homogeneous(xy)  # Nx4

    
    def radar2pixel(self, pose_mat):
        """Transform coordinates in the source frame to the target frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        # Transform points
        tformed_xy_hom = torch.matmul(pose_mat, torch.transpose(self.xy_hom)) # [B,4,N]
        # Convert from homogenous coordinates
        tformed_xy = tgm.convert_points_from_homogeneous(tformed_xy_hom) # [B,3,N]
        # Convert back from cartesian to polar
        theta_tformed_rad, rho_tformed = cart2pol(tformed_xy[:,0,:], tformed_xy[:,1,:])
        theta_tformed = torch.rad2deg(theta_tformed_rad) # [B,N]

        # tformed_xy = tformed_xy[:,0:2,:] # Drop augmented z column [B,2,N]
        # Replace 0-valued augmented z column with dB values of source img 
        # tformed_xy[:,2,:] = torch.flatten(img, start_dim=1) # [B,3,N] N points with x,y and db values

        X = theta_tformed # [B,N]
        Y = rho_tformed # [B,N]
        # w = self.num_angle_bins
        # h = self.numRangeBins

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2*X/(self.w-1) - 1
        Y_norm = 2*Y/(self.h-1) - 1  # Idem [B, H*W]

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]

        return pixel_coords.reshape(self.b, self.h, self.w, 2)

    
    def inverse_warp_fft(self, img, pose, rotation_mode='euler'):
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
        check_sizes(img, 'img', 'BHW')
        check_sizes(pose, 'pose', 'B6')

        self.b, self.h, self.w = img.size()

        assert self.w == self.num_angle_bins
        assert self.h == self.numRangeBins

        # if (xy_hom is None) or xy_hom.size(1) < 4:
        #     set_radar_grid()

        # Convert 6 DoF pose to 4x4 transformation matrix
        pose_mat = tgm.rtvec_to_pose(pose)  # T*R in homogenous coordinates [B,4,4]
        
        # src_pixel_coords = pixel_coords.reshape(b, h, w, 2)
        src_pixel_coords = self.radar2pixel(pose_mat)  # [B,H,W,2]

        projected_img = F.grid_sample(
            img, src_pixel_coords, padding_mode=self.padding_mode)

        # calculate mask values for each tformed_xy coordinates to match the target xy
        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1 

        return projected_img, valid_points

    #TODO: num_scales, mask eklenebilir buraya.
    # decibels loss
    def compute_db_loss(self, tgt_img, ref_imgs, poses, poses_inv):

        db_loss = 0

        for ref_img, pose, pose_inv in zip(ref_imgs, poses, poses_inv):

            db_loss1 = self.compute_pairwise_loss(tgt_img, ref_img, pose)
            db_loss2 = self.compute_pairwise_loss(ref_img, tgt_img, pose_inv)

            db_loss += (db_loss1 + db_loss2)

        return db_loss


    def compute_pairwise_loss(self, tgt_img, ref_img, pose):

        ref_img_warped, valid_mask = self.inverse_warp_fft(ref_img, pose)

        diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

        if self.with_auto_mask == True:
            auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
            valid_mask = auto_mask

        # compute all loss
        reconstruction_loss = mean_on_mask(diff_img, valid_mask)

        return reconstruction_loss

# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value




# # TODO: function could be parametrized
# def set_radar_grid():
#     global xy_hom
#     # RF params
#     rangeResolutionsInMeter = 0.0977
#     # dopplerResolutionMps = 0.0951
#     numRangeBins = 256
#     # numDopplerBins = 128
#     num_angle_bins = 64 # our choice

#     azimuths = np.arange(num_angle_bins)
#     azimuths = (azimuths - (num_angle_bins / 2))
#     ranges = np.arange(numRangeBins)
#     ranges *= rangeResolutionsInMeter

#     az_grid, range_grid = torch.meshgrid(azimuths, ranges)
#     x, y = pol2cart(torch.deg2rad(az_grid), range_grid)
#     x=torch.flatten(x)
#     y=torch.flatten(y)

#     xy = torch.vstack((x, y, torch.zeros_like(x)))  # Nx3 Augment with zero z column
#     xy_hom = tgm.convert_points_to_homogeneous(input)  # Nx4



def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))



def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x)
    return phi, rho

def pol2cart(phi, rho):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y





