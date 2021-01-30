import torch.utils.data as data
import numpy as np
# from skimage import io
from pathlib import Path
import random
import os


def load_as_float(path):
    # return io.imread(path).astype(np.float32)
    img = np.genfromtxt(path, delimiter=',', dtype=np.float32) # [H, W]
    img = img[np.newaxis,:,:] # [1, H, W] single channel image
    return img


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.csv
        root/scene_1/0000001.csv
        ..
        root/scene_2/0000000.csv
        .
        transform functions must take in an image
    """

    def __init__(self, root, skip_frames, sequence_length, train, seed=None, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.train = train
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder.strip() for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        self.transform = transform
        # self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(list(scene.glob('*.csv')))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                # sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                sample = {'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        
        # Shuffle training dataset
        if self.train:    
            random.shuffle(sequence_set)
        self.samples = sequence_set

    
    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.transform:
            tgt_img = self.transform(tgt_img)
            ref_imgs = [self.transform(ref_img) for ref_img in ref_imgs]

        return tgt_img, ref_imgs
    

    def __len__(self):
        return len(self.samples)
