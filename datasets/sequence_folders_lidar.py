import torch.utils.data as data
import numpy as np
import pandas as pd
from pathlib import Path
import random
import lidar


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

    def __init__(self, root, dataset, lo_params,
                 seed=None, train=True, sequence_length=3, transform=None, skip_frames=1
                 ):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        self.cart_pixels = lo_params['cart_pixels']
        # self.cart_pixels = 512
        self.max_range = 80.0  # if dataset == 'radiate' else 50.0
        # self.cart_resolution = self.max_range/self.cart_pixels
        self.transform = transform
        self.train = train
        self.k = skip_frames
        self.lidar_folder = 'velo_lidar' if dataset == 'radiate' else 'velodyne_left'
        self.lidar_ext = '*.csv' if dataset == 'radiate' else '*.png'
        self.ground_thr = -1.8 if dataset == 'radiate' else 1.0
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k,
                            demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            imgs = sorted(list((scene/self.lidar_folder).glob(self.lidar_ext)))

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

    # def load_bin(self, path):
    #     data = np.fromfile(path, dtype=np.float32)
    #     # .transpose()  # [3,N] x,y,I
    #     ptcld = data.reshape((len(data) // 3, 3))  # [N,4] x,y,z,I
    #     # ptcld = ptcld[[0, 1, 3], :]
    #     return ptcld

    def load_velodyne_csv(self, path):
        # x, y, z, intensity, ring
        # data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
        data = pd.read_csv(path, usecols=[0,1,2,3], dtype=np.float32).to_numpy()
        #data = data[[0, 1, 3], :]
        # return data[:,:4].transpose()
        return data.transpose()

    def load_radiate_velo(self, path):
        ptcld = self.load_velodyne_csv(path)
        ptcld[3, :] = self.reflectance2colour(ptcld)
        # Remove ground reflections
        ptcld = ptcld[:,ptcld[2]>self.ground_thr]

        img = self.ptc2img(ptcld)
        return img

    def load_robotcar_velo(self, path):
        ranges, intensities, angles, approximate_timestamps = lidar.load_velodyne_raw(
            path)
        # [4,N] x,y,z,I
        ptcld = lidar.velodyne_raw_to_pointcloud(ranges, intensities, angles)
        ptcld[3, :] = self.reflectance2colour(ptcld)        

        # Filter points at close range
        # ptcld = ptcld[:, np.logical_and(
        #     np.abs(ptcld[0]) > 4.0, np.abs(ptcld[1]) > 4.0)]

        # Remove ground reflections
        ptcld = ptcld[:,ptcld[2]<self.ground_thr]

        img = self.ptc2img(ptcld)
        return img

    def reflectance2colour(self, ptcld):
        # Convert reflectance to colour values in [0,1]
        reflectance = ptcld[3, :]
        if reflectance.size != 0:
            colours = (reflectance - reflectance.min()) / \
                (reflectance.max() - reflectance.min())
            colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
            return colours
        else:
            return reflectance

    def ptc2img(self, data):
        if data.shape[0] != 4:
            raise ValueError("Input must be [4,N]. Got {}".format(
                data.shape))

        # Calculate the sum of power returns that fall into the same 2D image pixel
        power_sum, _, _ = np.histogram2d(
            x=data[0], y=data[1],
            bins=[self.cart_pixels, self.cart_pixels],
            weights=data[3], normed=False,
            range=[[-self.max_range, self.max_range],
                   [-self.max_range, self.max_range]]
        )
        # Calculate the number of points in each pixel
        power_count, _, _ = np.histogram2d(
            x=data[0], y=data[1],
            bins=[self.cart_pixels, self.cart_pixels],
            range=[[-self.max_range, self.max_range],
                   [-self.max_range, self.max_range]]
        )
        # Calculate the mean of power return in each pixel.
        # histogram2d does either sums or finds the number of poitns, no average.
        img = np.divide(
            power_sum, power_count,
            out=np.zeros_like(power_sum), where=power_count != 0
        )
        img = img.astype(np.float32)[np.newaxis, :, :]  # / 255.
        # img[img < 0.2] = 0
        
        #img = np.nan_to_num(img, nan=1e-6)
        # if np.isnan(np.min(img)):
        #     print('NaN detected in input!')
        return img

    def load_lidar(self, path):
        data = self.load_radiate_velo(
            path) if self.dataset == 'radiate' else self.load_robotcar_velo(path)
        # data = self.ptc2img(data)
        return data

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = self.load_lidar(sample['tgt'])
        ref_imgs = [self.load_lidar(ref_img)
                    for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            # imgs = self.transform([tgt_img] + ref_imgs)
            tgt_img = self.transform(tgt_img)  # imgs[0]
            ref_imgs = [self.transform(img) for img in ref_imgs]  # imgs[1:]
        return tgt_img, ref_imgs

    def __len__(self):
        return len(self.samples)
