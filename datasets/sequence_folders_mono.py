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

    def __init__(self, root, dataset='robotcar', seed=None, train=True, sequence_length=3, transform=None, skip_frames=1):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        self.transform = transform
        self.train = train
        self.k = skip_frames
        self.mono_folder = 'zed_left' if dataset == 'radiate' else 'stereo/left'
        if dataset == 'robotcar':
            self.cam_model = CameraModel()
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k,
                            demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = utils.get_intrinsics_matrix(self.dataset)
            #imgs = sorted(scene.files('*.png'))
            imgs = sorted(list((scene/self.mono_folder).glob('*.png')))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics,
                          'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        if self.train:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def load_as_float(self, path):
        # img = imread(path).astype(np.float32)
        img = Image.open(path)
        # img = img.convert("RGB")
        # img = np.array(img)
        if self.dataset == 'robotcar':
            img = demosaic(img, 'gbrg')
            img = self.cam_model.undistort(img)
        img = np.array(img).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = self.load_as_float(sample['tgt'])
        ref_imgs = [self.load_as_float(ref_img)
                    for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform(
                [tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics

    def __len__(self):
        return len(self.samples)
