import torch.utils.data as data
from pathlib import Path

from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, path, transform=None, nsamples=0):
        self.path = path
        self.image_paths = sorted(list(self.path.glob('*.png')))
        if nsamples > 0 and nsamples < len(self.image_paths):
            skip = len(self.image_paths)//nsamples
            # idx = list(range(0, skip*nsamples, skip))
            self.image_paths = self.image_paths[0:skip*nsamples:skip]
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform:
            x = self.transform(x)
        return x, self.image_paths[index].name

    def __len__(self):
        return len(self.image_paths)
