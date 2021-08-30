import torch.utils.data as data
from pathlib import Path

from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_paths = sorted(list(self.path.glob('*.png')))
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)
