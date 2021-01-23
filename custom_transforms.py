from __future__ import division
import torch
import random
import numpy as np
# from PIL import Image

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image= t(image)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image.sub_(self.mean).div_(self.std)
        return image


class ArrayToTensor(object):

    def __call__(self, image):
        tensor = torch.from_numpy(image).float()
        return tensor
