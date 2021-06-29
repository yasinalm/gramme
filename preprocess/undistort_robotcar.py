import sys
sys.path.append('..')

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from datasets.robotcar_camera.camera_model import CameraModel



parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')


def load_as_float(path, cam_model):
    # img = imread(path).astype(np.float32)
    img = Image.open(path)
    # img = img.convert("RGB")
    # img = np.array(img)
    img = demosaic(img, 'gbrg')
    img = cam_model.undistort(img)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img

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

if __name__ == '__main__':
    args = parser.parse_args()

    root = Path(args.data)
    save_dir = 'stereo_undistorted'
    scenes = [f for f in root.iterdir() if f.is_dir()]

    stereo_left_folder = 'stereo/left'
    stereo_right_folder = 'stereo/right'
    cam_model_left = CameraModel()
    cam_model_right = CameraModel('stereo_wide_right')

    for scene in tqdm(scenes):        
        left_imgs = sorted(
            list((scene/stereo_left_folder).glob('*.png')))
        right_imgs = sorted(
            list((scene/stereo_right_folder).glob('*.png')))

        save_stereo_dir = scene/save_dir
        save_stereo_dir.mkdir()

        save_stereo_dir_left = save_stereo_dir/'left'
        save_stereo_dir_left.mkdir()
        save_stereo_dir_right = save_stereo_dir/'right'
        save_stereo_dir_right.mkdir()

        for path in left_imgs:
            img = load_as_float(path, cam_model_left)
            img = preprocess(img)

            save_name = save_stereo_dir_left/path.name
            img.save(save_name)
        
        for path in right_imgs:
            img = load_as_float(path, cam_model_right)
            img = preprocess(img)

            save_name = save_stereo_dir_right/path.name
            img.save(save_name)
