import sys  # nopep8
sys.path.append('..')  # nopep8

from pathlib import Path
# import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Crop and resize CADCD camera images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int,
                    metavar='N', help='number of image processing workers')
save_folder = 'cam_preprocessed'
offset_x = 140
offset_y = 250
in_h = 1024
in_w = 1280
(out_w, out_h) = (320, 192)
(left, upper, right, lower) = (offset_x, 0, in_w-2*offset_x, in_h-offset_y)


def preprocess(path):
    save_name = save_dir/path.name
    # if save_name.is_file():
    #     return

    # try:
    #     img_saved = Image.open(save_name)
    #     img_saved.verify()
    # except Exception:
    #     # print(path)
    img = Image.open(path)
    img_prep = img.crop((left, upper, right, lower))
    # Resize
    img_prep = img_prep.resize((out_w, out_h))
    img_prep.save(save_name)


def pool_handler(imgs):
    p = Pool(args.workers)
    p.map_async(preprocess, imgs)
    p.close()


if __name__ == '__main__':
    args = parser.parse_args()

    root = Path(args.data)
    folders = [f for f in root.iterdir() if f.is_dir()]
    scenes = [f for folder in folders for f in folder.iterdir() if f.is_dir()]
    print('=> found {} scenes in {}'.format(len(scenes), root))

    cams = ['image_0{}'.format(i) for i in range(8)]

    for scene in tqdm(scenes):
        # scene = scenes[1]
        for cam in tqdm(cams):
            # cam = cams[0]
            imgs = list((scene/'raw'/cam/'data').glob('*.png'))
            # print(len(imgs))
            save_dir = scene/'raw'/cam/save_folder
            save_dir.mkdir(exist_ok=True)
            # print(save_dir)
            # pool_handler(imgs)
            for path in tqdm(imgs):
                # path = imgs[0]
                preprocess(path)
