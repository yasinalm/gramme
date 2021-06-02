import torch.utils.data as data
import numpy as np
#from imageio import imread
from pathlib import Path
import random

from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from .robotcar_camera.camera_model import CameraModel

import utils_warp as utils


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, dataset='robotcar',
                 seed=None, train=True, sequence_length=3, transform=None, skip_frames=1):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        self.cart_pixels = 512
        self.max_range = 100.0 if dataset == 'radiate' else 50.0
        self.cart_resolution = self.max_range/self.cart_pixels
        self.transform = transform
        self.train = train
        self.k = skip_frames
        self.lidar_folder = 'velo_lidar' if dataset == 'radiate' else 'lms_front'
        self.lidar_ext = '*.csv' if dataset == 'radiate' else '*.bin'
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k,
                            demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            imgs = sorted(list((scene/self.mono_folder).glob(self.lidar_ext)))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        if self.train:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def load_bin(self, path):
        data = np.fromfile(path, dtype=np.float32)
        ptcld = data.reshape((4, -1))  # [4,N] x,y,z,I
        ptcld = ptcld[[0, 1, 3], :]
        return ptcld

    def load_csv(self, path):
        # x, y, z, intensity, ring
        data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
        data = data[[0, 1, 3], :]
        return data

    def ptc2img(self, data):
        # Calculate the sum of power returns that fall into the same 2D image pixel
        power_sum = np.histogram2d(
            x=data[:, 0], y=data[:, 1],
            bins=[self.cart_pixels, self.cart_pixels],
            weights=data[:2], normed=False
        )
        # Calculate the number of points in each pixel
        power_count = np.histogram2d(
            x=data[:, 0], y=data[:, 1],
            bins=[self.cart_pixels, self.cart_pixels]
        )
        # Calculate the mean of power return in each pixel.
        # histogram2d does either sums or finds the number of poitns, no average.
        img = np.divide(
            power_sum, power_count,
            out=np.zeros_like(power_sum), where=power_count != 0
        )
        return img

    def load_lidar(self, path):
        data = self.load_csv(
            path) if self.dataset == 'radiate' else self.load_bin(path)
        data = self.ptc2img(data)
        return data

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = self.load_lidar(sample['tgt'])
        ref_imgs = [self.load_lidar(ref_img)
                    for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs = self.transform([tgt_img] + ref_imgs)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        return tgt_img, ref_imgs

    def __len__(self):
        return len(self.samples)
