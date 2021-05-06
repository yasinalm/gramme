
from typing import AnyStr, Tuple
import numpy as np
from PIL import Image
#import cv2

import torch.nn.functional as F
import torch


def load_radar(example_path: AnyStr, dataset: AnyStr) -> np.ndarray:
    """Read radar frame
    Args:
        example_path (AnyStr): Path to range-azimuth png
    Returns:
        fft_data (np.ndarray): Radar power readings along each azimuth
    """

    skip_header = 11 if dataset == 'robotcar' else 0

    raw_example_data = Image.open(example_path)
    raw_example_data = np.array(raw_example_data)
    # raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    if dataset == 'radiate' or dataset == 'hand':
        raw_example_data = np.transpose(raw_example_data)
    fft_data = raw_example_data[:, skip_header:].astype(
        np.float32)[:, :, np.newaxis] / 255.

    return fft_data


################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################
def radar_polar_to_cartesian(angleResolutionInRad: float, fft_data: np.ndarray, radar_resolution: float,
                             cart_resolution: float, cart_pixel_width: int, dataset: AnyStr, interpolate_crossover=True) -> np.ndarray:
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_size (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.
    Returns:
        np.ndarray: Cartesian radar power readings
    Notes:
        After using the warping grid the output radar cartesian is defined as as follows where
        X and Y are the `real` world locations of the pixels in metres:
         If 'cart_pixel_width' is odd:
                        +------ Y = -1 * cart_resolution (m)
                        |+----- Y =  0 (m) at centre pixel
                        ||+---- Y =  1 * cart_resolution (m)
                        |||+--- Y =  2 * cart_resolution (m)
                        |||| +- Y =  cart_pixel_width // 2 * cart_resolution (m) (at last pixel)
                        |||| +-----------+
                        vvvv             v
         +---------------+---------------+
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+ <-- X = 0 (m) at centre pixel
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+
         <------------------------------->
             cart_pixel_width (pixels)
         If 'cart_pixel_width' is even:
                        +------ Y = -0.5 * cart_resolution (m)
                        |+----- Y =  0.5 * cart_resolution (m)
                        ||+---- Y =  1.5 * cart_resolution (m)
                        |||+--- Y =  2.5 * cart_resolution (m)
                        |||| +- Y =  (cart_pixel_width / 2 - 0.5) * cart_resolution (m) (at last pixel)
                        |||| +----------+
                        vvvv            v
         +------------------------------+
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         +------------------------------+
         <------------------------------>
             cart_pixel_width (pixels)
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    if dataset == 'hand':  # 180deg FoV
        x_coords = np.linspace(
            0, cart_min_range, cart_pixel_width, dtype=np.float32)
        y_coords = np.linspace(-cart_min_range/2, cart_min_range/2,
                               cart_pixel_width, dtype=np.float32)
        Y, X = np.meshgrid(x_coords, y_coords)
    else:  # Full FoV
        coords = np.linspace(-cart_min_range, cart_min_range,
                             cart_pixel_width, dtype=np.float32)
        Y, X = np.meshgrid(coords, -coords)

    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = angleResolutionInRad  # azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    #sample_v = (sample_angle - azimuths[0]) / azimuth_step
    sample_v = (sample_angle - angleResolutionInRad) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    # polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    # cart_img = np.expand_dims(
    #     cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), 0)

    # Alternative to cv2.remap
    h, w = fft_data.shape[:2]
    polar_to_cart_warp = np.stack((2*sample_u/w-1, 2*sample_v/h-1), -1)
    # print(polar_to_cart_warp)
    # [H,W,C] -> [C,H,W] for F.grid_map
    fft_data = np.squeeze(fft_data)[None, ...]
    # print(polar_to_cart_warp.shape)

    # grid_sample need [N,C,H,W] format
    cart_img = F.grid_sample(torch.Tensor(fft_data[None, ...]), torch.Tensor(
        polar_to_cart_warp[None, ...])).numpy()
    cart_img = np.squeeze(cart_img, axis=0)  # Remove N

    return cart_img
