from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time

import models
import utils
# import custom_transforms_mono as T

import torch
import torch.backends.cudnn as cudnn
import torchvision as tv
import torchvision.transforms as T
from datasets.sequence_folders_disp import ImageFolder

parser = argparse.ArgumentParser(description='Script for testing depth predictions with the corresponding ground truth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='hand', help='the dataset to train')
parser.add_argument('--with-preprocessed', type=int, default=1,
                    help='use the preprocessed undistorted images')
parser.add_argument('--with-testfile', type=int, default=0,
                    help='use the test.txt file containing test sequences')
parser.add_argument('--with-timing', type=int, default=0,
                    help='use the timing benchmark to evaluate the runtime speed')
parser.add_argument("--pretrained-disp", required=True,
                    type=str, help="pretrained DispNet path")
parser.add_argument('-j', '--workers', default=4, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument("--img-height", default=192, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
# parser.add_argument("--min-depth", default=1e-3)
# parser.add_argument("--max-depth", default=80)
# parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
# parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
# parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--results-dir', default='results', metavar='PATH',
                    help='directory where to save predicted depth maps and stats')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation')

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()

    results_dir = Path(args.results_dir)  # /args.sequence
    results_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    img_size = (args.img_height, args.img_width)

    if args.dataset == 'robotcar':
        if args.with_preprocessed:
            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            valid_transform = T.Compose([
                T.ToPILImage(),
                T.CropBottom(),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])

    elif args.dataset == 'radiate':
        if args.with_preprocessed:
            valid_transform = T.Compose([
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            valid_transform = T.Compose([
                # T.ToPILImage(),
                T.Resize(img_size),
                T.ToTensor(),
                # T.Normalize(imagenet_mean, imagenet_std)
            ])
    if args.with_testfile:
        root = Path(args.data)
        scene_list_path = root/'test.txt'
        scenes = [root/folder.strip()/'stereo_undistorted/left'
                  for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        scene_names = [folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
    else:
        scene_names = 'sequence'
        scenes = [Path(args.data)]

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(
        args.resnet_layers, False).to(device)

    # load parameters
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)

    # switch to evaluate mode
    disp_net.eval()

    for scene_name, scene in zip(scene_names, scenes):
        print("=> Processing:", scene_name)

        results_depth_dir = results_dir/scene_name/'depth'
        results_depth_dir.mkdir(parents=True)

        test_set = ImageFolder(
            path=scene, transform=valid_transform)
        nframes = len(test_set)
        print('{} samples found in {} '.format(
            nframes, scene_name))
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

        avg_time = 0
        for i, tgt_img in tqdm(enumerate(test_loader)):
            tgt_img = tgt_img.to(device)

            if args.with_timing:
                # compute speed
                torch.cuda.synchronize()
                t_start = time.time()

            tgt_depth = [disp_to_depth(disp) for disp in disp_net(tgt_img)]

            if args.with_timing:
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                avg_time += elapsed_time

            # tv.utils.save_image(
            #     tgt_depth[0], results_depth_dir/'{0:03d}.png'.format(i))

            for j, depth in enumerate(tgt_depth[0]):
                colour_depth = utils.tensor2array(
                    depth, max_value=None, colormap='inferno')
                colour_depth = colour_depth.transpose(1, 2, 0)*255
                colour_depth = colour_depth.astype(np.uint8)
                im = Image.fromarray(colour_depth)
                im.save(results_depth_dir /
                        '{0:03d}.png'.format(i*args.batch_size+j))

        if args.with_timing:
            avg_time /= nframes
            print('Avg Time: ', avg_time, ' seconds.')
            print('Avg Speed: ', 1.0 / avg_time, ' fps')


# def disp_to_depth(disp):
#     """Convert network's sigmoid output into depth prediction
#     The formula for this conversion is given in the 'additional considerations'
#     section of the paper.
#     """
#     # Disp is not scaled in DispResNet.
#     min_depth = 0.1
#     max_depth = 100.0
#     min_disp = 1 / max_depth
#     max_disp = 1 / min_depth
#     # disp = disp.clamp(min=1e-6)
#     scaled_disp = min_disp + (max_disp - min_disp) * disp
#     depth = 1 / scaled_disp
#     # depth = 1./disp
#     return depth


def disp_to_depth(disp):
    # depth_scale = 10.0
    # id_disp = torch.rand(disp.shape).to(device)*1e-12
    # disp = disp + id_disp
    depth = 1./disp
    # depth = depth/depth_scale
    depth = depth.clamp(min=1e-6)
    return depth


if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=False):
        main()
