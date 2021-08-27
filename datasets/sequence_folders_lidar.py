import torch.utils.data as data
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from PIL import Image
import random
import lidar
import utils_warp as utils
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from .robotcar_camera.camera_model import CameraModel


class SequenceFolder(data.Dataset):
    """LIDAR dataset loader.
    """

    def __init__(self, root, dataset, lo_params,
                 seed=None, train=True, sequence_length=3,
                 transform=None, skip_frames=1,
                 load_mono=False, mono_preprocessed=False, mono_transform=None
                 ):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        self.load_mono = load_mono
        self.preprocessed = mono_preprocessed

        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        self.cart_pixels = lo_params['cart_pixels']
        # self.cart_pixels = 512
        self.max_range = 80.0  # if dataset == 'radiate' else 50.0
        # self.cart_resolution = self.max_range/self.cart_pixels

        if self.load_mono:
            # self.mono_folder = 'zed_left' if dataset == 'radiate' else 'stereo/left'
            # if dataset == 'robotcar':
            #     self.cam_model = CameraModel()

            if dataset == 'radiate':
                if self.preprocessed:
                    self.mono_folder = 'stereo_undistorted/left'
                else:
                    self.mono_folder = 'zed_left'
            elif dataset == 'robotcar':
                if self.preprocessed:
                    self.mono_folder = 'stereo_undistorted/left'
                else:
                    self.mono_folder = 'stereo/left'
                    self.cam_model = CameraModel()
            else:
                raise NotImplementedError(
                    'The chosen dataset is not implemented yet! Given: {}'.format(dataset))

        self.transform = transform
        self.mono_transform = mono_transform
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
        self.shifts = list(range(-demi_length * self.k,
                                 demi_length * self.k + 1, self.k))
        self.shifts.pop(demi_length)
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = utils.get_intrinsics_matrix(
                self.dataset, preprocessed=self.preprocessed)

            imgs = sorted(list((scene/self.lidar_folder).glob(self.lidar_ext)))

            if len(imgs) < sequence_length:
                continue

            if self.load_mono:
                # We need to collect the corresponding monocular frames within the same dataset class.
                # If we use separate classes for radar and mono, the order gets messy due to shuffling.

                # Temporary fix to read stereo folder from non-dataroot directory.
                # Delete the following lines once done. uncomment f_mt below.
                # if self.dataset == 'robotcar':
                #     temp_root = Path('/home/yasin/mmwave-raw/robotcar')
                #     f_mt = temp_root/scene.name/'stereo.timestamps'
                #     if not f_mt.is_file():
                #         continue
                #     imgs_mono = sorted(
                #         list((temp_root/scene.name/self.mono_folder).glob(f_type)))
                # elif self.dataset == 'radiate':
                imgs_mono = sorted(
                    list((scene/self.mono_folder).glob('*.png')))
                if len(imgs) < sequence_length:
                    continue

                if self.dataset == 'radiate':
                    f_rt = scene/'velo_lidar.txt'
                    f_mt = scene/'zed_left.txt'

                    rts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_rt)]
                    mts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_mt)]
                    # Some scenes contain timestamps more than images. Drop the extra timestamps.
                    mts = mts[:len(imgs_mono)]

                elif self.dataset == 'robotcar':
                    f_rt = scene/'velodyne_left.timestamps'
                    f_mt = scene/'stereo.timestamps'

                    # Robotcar timestamps are in microsecs.
                    # Read them in secs.
                    rts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_rt)]
                    mts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_mt)]
                else:
                    raise NotImplementedError(
                        'Currently, RADIATE and RobotCar datasets supported for VO')

                lidar_idxs = list(
                    range(demi_length * self.k, len(imgs)-demi_length * self.k))
                mono_matches_all = self.find_mono_samples(
                    lidar_idxs, rts, mts)

            for cnt, i in enumerate(range(demi_length * self.k, len(imgs)-demi_length * self.k)):
                sample = {'tgt': imgs[i], 'ref_imgs': []}
                for j in self.shifts:
                    sample['ref_imgs'].append(imgs[i+j])

                if self.load_mono:
                    # self.find_mono_samples(i, rts, mts)
                    # try:
                    mono_matches = mono_matches_all[cnt]
                    # except IndexError:
                    #     print(len(mono_matches_all))
                    #     print(i)
                    #     raise IndexError('Patladi!')
                    if mono_matches:
                        # Add all the monocular frames between the matched source and target frames.
                        sample['vo_tgt_img'] = imgs_mono[mono_matches[0]]

                        # vo_ref_imgs = [
                        # [imgs_mono[src-1],...,imgs_mono[tgt]],
                        # [imgs_mono[tgt],...,imgs_mono[src+1]
                        # ]
                        sample['vo_ref_imgs'] = [
                            # [imgs_mono[ref] for ref in refs] for refs in mono_matches[1:]]
                            imgs_mono[ref] for ref in mono_matches[1:]]
                        sample['intrinsics'] = intrinsics
                    else:
                        continue

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
        data = pd.read_csv(
            path, usecols=[0, 1, 2, 3], dtype=np.float32).to_numpy()
        #data = data[[0, 1, 3], :]
        # return data[:,:4].transpose()
        return data.transpose()

    def load_radiate_velo(self, path):
        ptcld = self.load_velodyne_csv(path)
        ptcld[3, :] = self.reflectance2colour(ptcld)
        # Remove ground reflections
        ptcld = ptcld[:, ptcld[2] > self.ground_thr]

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
        ptcld = ptcld[:, ptcld[2] < self.ground_thr]

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

    def load_mono_img_as_float(self, path):
        img = Image.open(path)
        # img = img.resize((640, 384))
        if self.dataset == 'robotcar':
            img = demosaic(img, 'gbrg')
            img = self.cam_model.undistort(img)
        img = np.array(img).astype(np.uint8)
        # img = img.astype(np.float32) / 255.
        return img

    def load_undistorted_mono_img_as_float(self, path):
        img = Image.open(path)
        return img

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = self.load_lidar(sample['tgt'])
        ref_imgs = [self.load_lidar(ref_img)
                    for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            # imgs = self.transform([tgt_img] + ref_imgs)
            tgt_img = self.transform(tgt_img)  # imgs[0]
            ref_imgs = [self.transform(img) for img in ref_imgs]  # imgs[1:]

        if self.load_mono:
            if 'vo_tgt_img' in sample:
                if self.preprocessed or self.dataset == 'radiate':
                    # TODO: On-the-fly rectification support for RADIATE dataset
                    self.load_img = self.load_undistorted_mono_img_as_float
                else:
                    self.load_img = self.load_mono_img_as_float
                vo_tgt_img = self.load_img(sample['vo_tgt_img'])
                vo_ref_imgs = [self.load_img(
                    ref_img) for ref_img in sample['vo_ref_imgs']]
                if self.mono_transform:
                    imgs, intrinsics = self.mono_transform(
                        [vo_tgt_img] + vo_ref_imgs, np.copy(sample['intrinsics']))
                    vo_tgt_img = imgs[0]
                    vo_ref_imgs = imgs[1:]
                    # vo_tgt_img = self.mono_transform(vo_tgt_img)
                    # vo_ref_imgs = [
                    #     self.mono_transform(ref_img) for ref_img in vo_ref_imgs]
            else:
                vo_tgt_img = []
                vo_ref_imgs = []
            return tgt_img, ref_imgs, vo_tgt_img, vo_ref_imgs, intrinsics
        else:
            return tgt_img, ref_imgs

    def __len__(self):
        return len(self.samples)

    def find_mono_samples(self, t_idxs: List[int], rts: List[List[float]], mts: List[List[float]]) -> List[List[int]]:
        """Returns indexes of monocular frames in the form of 
        [[tgt, [src-1,...,tgt], [tgt,...,src+1]], [tgt, [src-1,...,tgt], [tgt,...,src+1]],...]

        Args:
            t_idx (List[int]): Indices of the target lidar frames
            rts (List[float]): List of lidar timestamps
            mts (List[float]): List of monocular timestamps

        Returns:
            List[List[int]]: Indexes of the matched monocular frames
        """
        t_matches = []
        last_search_idx = 0
        for t_idx in t_idxs:
            idxs = [find_nearest_mono_idx(rts[t_idx], mts, last_search_idx)]
            for s in self.shifts:
                idxs.append(find_nearest_mono_idx(
                    rts[t_idx+s], mts, last_search_idx))

            # Check if any of the source or target images are not found,
            # also check if the matched indices are unique.
            if any([i < 0 for i in idxs]) or len(set(idxs)) < len(self.shifts)+1:
                t_matches.append([])
                continue
            last_search_idx = idxs[1]
            # # Return all the monocular frame between target and source frames
            # # Convert from [[tgt, src-1, src+1], [tgt, src-1, src+1],...] to
            # # [[tgt, [src-1,...,tgt], [tgt,...,src+1]], [tgt, [src-1,...,tgt], [tgt,...,src+1]],...]
            # idxs[1] = list(range(idxs[1], idxs[0]))
            # idxs[2] = list(range(idxs[0]+1, idxs[2]+1))
            # # Check if we get exactly three previous and next frames.
            # # This is needed for batching.
            # # If we have more than three frames in either directons, trim the last three frames.
            # # Otherwise, return empty list.
            # if len(idxs[1]) > 3:
            #     idxs[1] = idxs[1][-3:]
            # else:
            #     t_matches.append([])
            #     continue
            # if len(idxs[2]) > 3:
            #     idxs[2] = idxs[2][-3:]
            # else:
            #     t_matches.append([])
            #     continue
            # Append the matched sequence
            t_matches.append(idxs)
        return t_matches


def find_nearest_mono_idx(t: int, mts: List[List[float]], last_search_idx: int) -> int:
    """Finds the nearest monocular timestamp for the given lidar timestamp

    Args:
        t (int): Timestamp of the target frame
        mts (List[List[float]]): List of monocular timestamps

    Returns:
        int: Index of the matched monocular frame
    """

    del_t = 0.050  # the match must be within 50ms of t
    # First check if t is outside monocular frames but still within thr close
    # Check if t comes before monocular frames
    if t < mts[last_search_idx]:
        return last_search_idx if mts[last_search_idx]-t < del_t else -1
    # Check if t comes after monocular frames
    if t > mts[-1]:
        return len(mts)-1 if t-mts[-1] < del_t else -1
    # Otherwise search within monocular frames
    for i in range(last_search_idx, len(mts)-1):
        if t > mts[i] and t < mts[i+1]:
            idx = i if (t-mts[i]) < (mts[i+1]-t) else i+1
            idx = idx if abs(mts[idx]-t) < del_t else -1
            return idx
    return -1
