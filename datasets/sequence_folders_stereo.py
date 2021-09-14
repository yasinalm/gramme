from preprocess.undistort_robotcar import preprocess
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

    def __init__(self, root, dataset='robotcar', seed=None, train=True,
                 sequence_length=3, transform=None, skip_frames=1, preprocessed=False,
                 sequence=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        if sequence is not None:
            self.scenes = [self.root/sequence]
        else:
            scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
            self.scenes = [self.root/folder.strip()
                           for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        # self.scenes = self.scenes[:len(self.scenes)//10]
        self.transform = transform
        self.train = train
        self.k = skip_frames
        self.preprocessed = preprocessed
        if dataset == 'radiate':
            if self.preprocessed:
                self.stereo_left_folder = 'stereo_undistorted/left'
                self.stereo_right_folder = 'stereo_undistorted/right'
            else:
                self.stereo_left_folder = 'zed_left'
                self.stereo_right_folder = 'zed_right'
        elif dataset == 'robotcar':
            if self.preprocessed:
                self.stereo_left_folder = 'stereo_undistorted/left'
                self.stereo_right_folder = 'stereo_undistorted/right'
            else:
                self.stereo_left_folder = 'stereo/left'
                self.stereo_right_folder = 'stereo/right'
                self.cam_model_left = CameraModel()
                self.cam_model_right = CameraModel('stereo_wide_right')
        elif dataset == 'cadcd':
            if self.preprocessed:
                self.stereo_left_folder = 'preprocessed/image_07/data'
                self.stereo_right_folder = 'preprocessed/image_01/data'
            else:
                self.stereo_left_folder = 'raw/image_07/data'
                self.stereo_right_folder = 'raw/image_01/data'
        else:
            raise NotImplementedError(
                'The chosen dataset is not implemented yet! Given: {}'.format(dataset))
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
            intrinsics = utils.get_intrinsics_matrix(
                self.dataset, preprocessed=self.preprocessed)
            intrinsics_right = utils.get_intrinsics_matrix(
                self.dataset, preprocessed=self.preprocessed, cam='right')
            rightTleft = utils.get_rightTleft(self.dataset)
            #imgs = sorted(scene.files('*.png'))
            left_imgs = sorted(
                list((scene/self.stereo_left_folder).glob('*.png')))
            right_imgs = sorted(
                list((scene/self.stereo_right_folder).glob('*.png')))
            # RADIATE dataset occasionally has missing images. Discard them.
            len_imgs = min(len(left_imgs), len(right_imgs))

            for i in range(demi_length * self.k, len_imgs-demi_length * self.k):
                sample = {'intrinsics': [],
                          'rightTleft': rightTleft,
                          'tgt': left_imgs[i], 'ref_imgs': []}
                if self.train:
                    sample['ref_imgs'].append(right_imgs[i])
                    sample['intrinsics'].append(intrinsics_right)
                for j in shifts:
                    sample['ref_imgs'].append(left_imgs[i+j])
                    sample['intrinsics'].append(intrinsics)
                sequence_set.append(sample)
        if self.train:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def load_as_float(self, path, cam_model):
        # img = imread(path).astype(np.float32)
        img = Image.open(path)
        # img = img.convert("RGB")
        # img = np.array(img)
        if self.dataset == 'robotcar':
            img = demosaic(img, 'gbrg')
            img = cam_model.undistort(img)
        img = np.array(img).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    def load_undistorted_as_float(self, path):
        # img = imread(path).astype(np.float32)
        img = Image.open(path)
        # img = img.convert("RGB")
        # img = np.array(img)
        # img = np.array(img).astype(np.uint8)
        # img = img.astype(np.uint8)
        # print("{} _ {}".format(img.min(), img.max()))
        # if np.isnan(img).any():
        #     print("nan detected in input")
        return img

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.dataset == 'radiate' or self.dataset == 'cadcd' or self.preprocessed:
            tgt_img = self.load_undistorted_as_float(sample['tgt'])
            ref_imgs = [self.load_undistorted_as_float(
                ref_img) for ref_img in sample['ref_imgs']]
        else:
            tgt_img = self.load_as_float(sample['tgt'], self.cam_model_left)
            ref_imgs = [self.load_as_float(
                sample['ref_imgs'][0], self.cam_model_right)]
            for ref_img in sample['ref_imgs'][1:]:
                ref_imgs.append(self.load_as_float(
                    ref_img, self.cam_model_left))

        if self.transform is not None:
            imgs, intrinsics, extrinsics = self.transform(
                [tgt_img] + ref_imgs, [np.copy(i) for i in sample['intrinsics']],  np.copy(sample['rightTleft']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            # if any([np.isnan(np.array(img)).any() for img in imgs]):
            #     print("getitem: nan detected in input")
        else:
            intrinsics = [np.copy(i) for i in sample['intrinsics']]
            extrinsics = np.copy(sample['rightTleft'])

        if self.train:
            return tgt_img, ref_imgs, intrinsics, extrinsics
        else:
            return tgt_img, ref_imgs, intrinsics

    def __len__(self):
        return len(self.samples)
