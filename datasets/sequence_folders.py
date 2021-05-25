from typing import List
import torch.utils.data as data
import numpy as np
# from skimage import io
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from .robotcar_camera.camera_model import CameraModel
import utils_warp as utils
# import cv2
from pathlib import Path
from typing import List
import random
import radar


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.csv
        root/scene_1/0000001.csv
        ..
        root/scene_2/0000000.csv
        .
        transform functions must take in an image
    """

    def __init__(
            self, root, skip_frames, sequence_length, train, dataset,
            ro_params=None, load_mono=False,
            seed=None, transform=None, mono_transform=None, sequence=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.train = train
        self.load_mono = load_mono

        self.cart_resolution = ro_params['cart_resolution']
        self.cart_pixels = ro_params['cart_pixels']
        self.rangeResolutionsInMeter = ro_params['rangeResolutionsInMeter']
        self.angleResolutionInRad = ro_params['angleResolutionInRad']
        self.interpolate_crossover = True

        # if ro_params is not None:
        # self.modality = 'radar'
        self.isCartesian = ro_params['radar_format'] == 'cartesian'
        if self.isCartesian:
            self.radar_folder = 'radar_cart' if dataset == 'robotcar' else 'Navtech_Cartesian'
        else:
            self.radar_folder = 'radar' if dataset == 'robotcar' else 'Navtech_Polar'

        if sequence is not None:
            self.scenes = [self.root/sequence]
        else:
            scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
            self.scenes = [self.root/folder.strip() for folder in open(scene_list_path)
                           if not folder.strip().startswith("#")]

        if self.load_mono:
            self.mono_folder = 'zed_left' if dataset == 'radiate' else 'stereo/left'
            if dataset == 'robotcar':
                self.cam_model = CameraModel()

        self.transform = transform
        self.mono_transform = mono_transform
        self.dataset = dataset
        self.k = skip_frames
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
            intrinsics = utils.get_intrinsics_matrix(self.dataset)
            # f_type = '*.csv' if self.dataset == 'hand' else '*.png'
            f_type = '*.png'
            imgs = sorted(list((scene/self.radar_folder).glob(f_type)))
            if len(imgs) < sequence_length:
                continue

            if self.load_mono:
                # We need to collect the corresponding monocular frames within the same dataset class.
                # If we use separate classes for radar and mono, the order gets messy due to shuffling.

                # Temporary fix to read stereo folder from non-dataroot directory.
                # Delete the following lines once done. uncomment f_mt below.
                if self.dataset == 'robotcar':
                    temp_root = Path('/home/yasin/mmwave-raw/robotcar')
                    f_mt = temp_root/scene.name/'stereo.timestamps'
                    if not f_mt.is_file():
                        continue
                    imgs_mono = sorted(
                        list((temp_root/scene.name/self.mono_folder).glob(f_type)))
                elif self.dataset == 'radiate':
                    imgs_mono = sorted(
                        list((scene/self.mono_folder).glob(f_type)))
                if len(imgs) < sequence_length:
                    continue

                if self.dataset == 'radiate':
                    f_rt = scene/'Navtech_Polar.txt'
                    f_mt = scene/'zed_left.txt'

                    rts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_rt)]
                    mts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_mt)]
                    # Some scenes contain timestamps more than images. Drop the extra timestamps.
                    mts = mts[:len(imgs_mono)]

                elif self.dataset == 'robotcar':
                    f_rt = scene/'radar.timestamps'
                    # f_mt = scene/'stereo.timestamps'

                    # Robotcar timestamps are in microsecs.
                    # Read them in secs.
                    rts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_rt)]
                    mts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_mt)]
                else:
                    raise NotImplementedError(
                        'Currently, RADIATE and RobotCar datasets supported for VO')

                radar_idxs = list(
                    range(demi_length * self.k, len(imgs)-demi_length * self.k))
                mono_matches_all = self.find_mono_samples(
                    radar_idxs, rts, mts)

            for cnt, i in enumerate(range(demi_length * self.k, len(imgs)-demi_length * self.k)):
                # sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
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

        # Shuffle training dataset
        if self.train:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def load_radar_img_as_float(self, path):
        # Robotcar dataset has header of length 11 columns
        fft_data = radar.load_radar(str(path), self.dataset)
        cart_img = radar.radar_polar_to_cartesian(self.angleResolutionInRad, fft_data, self.rangeResolutionsInMeter,
                                                  self.cart_resolution, self.cart_pixels, self.dataset, self.interpolate_crossover)
        cart_img[cart_img < 0.2] = 0
        return cart_img

    def load_mono_img_as_float(self, path):
        img = Image.open(path)
        # img = img.resize((640, 384))
        if self.dataset == 'robotcar':
            img = demosaic(img, 'gbrg')
            img = self.cam_model.undistort(img)
        img = img.astype(np.uint8)
        # img = img.astype(np.float32) / 255.
        return img

    def load_cart_as_float(self, path):
        raw_data = Image.open(path)
        raw_data = np.array(raw_data)
        # raw_data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        cart_img = raw_data.astype(np.float32)[np.newaxis, :, :] / 255.
        cart_img[cart_img < 0.2] = 0

        return cart_img

    def load_csv_as_float(self, path):
        img = np.genfromtxt(path, delimiter=',', dtype=np.float32)  # [H, W]
        img = img.transpose()
        # img = img[np.newaxis,:,:] # [1, H, W] single channel image

        # Min-max normalization to [0,1]
        fft_data = img - img.min()
        fft_data = fft_data/fft_data.max()

        cart_img = radar.radar_polar_to_cartesian(
            self.angleResolutionInRad, fft_data, self.rangeResolutionsInMeter,
            self.cart_resolution, self.cart_pixels, self.dataset, self.interpolate_crossover)

        return cart_img

    def find_mono_samples(self, t_idxs: List[int], rts: List[List[float]], mts: List[List[float]]) -> List[List[int]]:
        """Returns indexes of monocular frames in the form of 
        [[tgt, [src-1,...,tgt], [tgt,...,src+1]], [tgt, [src-1,...,tgt], [tgt,...,src+1]],...]

        Args:
            t_idx (List[int]): Indices of the target radar frames
            rts (List[float]): List of radar timestamps
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

    def __getitem__(self, index):
        """Returns an item from the dataset

        Args:
            index (int): index of the element to return

        Returns:
            tgt_img (torch.Tensor): Target radar frame [B,1,H,W]
            ref_imgs (List[torch.Tensor]): Reference target frames [2,B,1,H,W]
            vo_tgt_img (torch.Tensor): Target monocular frame [B,3,H,W]
            vo_ref_imgs (List[List[torch.Tensor]]): Reference monocular frames 
            [num_sequence-1,num_matches,B,num_channels,H,W]=[2,3,B,3,H,W]
        """

        sample = self.samples[index]
        # Choose the loader function
        # if self.modality == 'radar':
        # if self.isCartesian:
        #     load_as_float = self.load_cart_as_float
        # else:
        #     # load_as_float = self.load_csv_as_float if self.dataset=='hand' else self.load_img_as_float
        #     load_as_float = self.load_radar_img_as_float
        load_as_float = self.load_cart_as_float if self.isCartesian else self.load_radar_img_as_float
        # else:
        #     load_as_float = self.load_mono_img_as_float
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.transform:
            tgt_img = self.transform(tgt_img)
            ref_imgs = [self.transform(ref_img) for ref_img in ref_imgs]

        if self.load_mono:
            if 'vo_tgt_img' in sample:
                vo_tgt_img = self.load_mono_img_as_float(sample['vo_tgt_img'])
                vo_ref_imgs = [self.load_mono_img_as_float(
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


def find_nearest_mono_idx(t: int, mts: List[List[float]], last_search_idx: int) -> int:
    """Finds the nearest monocular timestamp for the given radar timestamp

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
