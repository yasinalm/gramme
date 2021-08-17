import sys  # nopep8
sys.path.append('..')  # nopep8

from datasets.radiate_camera.utils.calibration import Calibration
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import yaml


parser = argparse.ArgumentParser(description='Unrectify RADIATE ZED stereo images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')


def preprocess(img):
    # Crop bottom
    offset_y = 160
    in_h = 960
    in_w = 1280
    (left, upper, right, lower) = (0, 0, in_w, in_h-offset_y)
    img = img.crop((left, upper, right, lower))

    # Resize
    (width, height) = (320, 192)
    img = img.resize((width, height))

    return img


def get_rectified_resized(left_im, right_im, dim, leftMapX, leftMapY, rightMapX, rightMapY):
    """get the left and right image rectfied
    :param left_im: raw left image
    :type left_im: np.array
    :param right_im: raw right image
    :type right_im: np.array
    :return: tuple (left_rect, right_rect, disp_to_depth)
        WHERE
        np.array left_rect is the rectfied left image
        np.array right_rect is the rectfied right image
        np.array disp_to_depth is a matrix that converts the disparity values to distance in meters
    :rtype: tuple
    """

    fixedLeft = cv2.remap(left_im, leftMapX,
                          leftMapY, cv2.INTER_LINEAR)
    fixedRight = cv2.remap(right_im, rightMapX,
                           rightMapY, cv2.INTER_LINEAR)

    # Resize
    fixedLeft = cv2.resize(fixedLeft, dim)
    fixedRight = cv2.resize(fixedRight, dim)

    return fixedLeft, fixedRight


if __name__ == '__main__':
    args = parser.parse_args()

    root = Path(args.data)
    save_dir = 'stereo_undistorted'
    scenes = [f for f in root.iterdir() if f.is_dir()]
    # scenes = [Path('/media/storage/robotcar/2019-01-17-12-48-25-radar-oxford-10k')]
    # scenes = scenes[24:]
    print(scenes)

    stereo_left_folder = 'zed_left'
    stereo_right_folder = 'zed_right'

    config_file = '../datasets/radiate_camera/config/config.yaml'
    calib_file = '../datasets/radiate_camera/config/default-calib.yaml'
    # load parameters and calibration file
    with open(config_file, 'r') as file:
        config = yaml.full_load(file)
    with open(calib_file, 'r') as file:
        calib = yaml.full_load(file)
    config.update(calib)

    # generate calibration matrices from calib file
    calib = Calibration(config)

    # Resize dimensions
    (width, height) = (320, 192)
    dim = (width, height)

    # Calculate the coordinate transform
    (leftRectification, rightRectification, leftProjection,
     rightProjection, dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
        cameraMatrix1=calib.left_cam_mat,
        distCoeffs1=calib.left_cam_dist,
        cameraMatrix2=calib.right_cam_mat,
        distCoeffs2=calib.right_cam_dist,
        imageSize=tuple(calib.left_cam_res),
        R=calib.stereoR,
        T=calib.stereoT,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        calib.left_cam_mat,
        calib.left_cam_dist,
        leftRectification,
        leftProjection, tuple(calib.left_cam_res), cv2.CV_32FC1)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        calib.right_cam_mat,
        calib.left_cam_dist,
        rightRectification,
        rightProjection, tuple(calib.left_cam_res), cv2.CV_32FC1)

    for scene in tqdm(scenes):
        left_imgs = sorted(
            list((scene/stereo_left_folder).glob('*.png')))
        right_imgs = sorted(
            list((scene/stereo_right_folder).glob('*.png')))

        save_stereo_dir = scene/save_dir
        save_stereo_dir.mkdir(exist_ok=True)

        save_stereo_dir_left = save_stereo_dir/'left'
        save_stereo_dir_left.mkdir(exist_ok=True)
        save_stereo_dir_right = save_stereo_dir/'right'
        save_stereo_dir_right.mkdir(exist_ok=True)

        # RADIATE dataset occasionally has missing images. Discard them.
        len_imgs = min(len(left_imgs), len(right_imgs))

        for i in tqdm(range(len_imgs), leave=False):
            lpath = left_imgs[i]
            rpath = right_imgs[i]
            lsave_name = save_stereo_dir_left/lpath.name
            rsave_name = save_stereo_dir_right/rpath.name
            if lsave_name.is_file() and rsave_name.is_file():
                continue

            im_left = cv2.imread(str(lpath))
            im_right = cv2.imread(str(rpath))
            im_left_rect, im_right_rect = get_rectified_resized(
                im_left, im_right, dim, leftMapX, leftMapY, rightMapX, rightMapY)

            cv2.imwrite(str(lsave_name), im_left_rect)
            cv2.imwrite(str(rsave_name), im_right_rect)
