from __future__ import division
import random
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import numbers
import warnings
from collections.abc import Sequence

import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics, extrinsics):
        for t in self.transforms:
            images, intrinsics, extrinsics = t(images, intrinsics, extrinsics)
        return images, intrinsics, extrinsics

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensors: List[Tensor], intrinsics, extrinsics) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        tensors = [F.normalize(tensor, self.mean, self.std, self.inplace)
                   for tensor in tensors]
        return tensors, intrinsics, extrinsics

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor:
    """Convert a list of ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.
    .. _references: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __call__(self, images, intrinsics, extrinsics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        images = [F.to_tensor(pic) for pic in images]
        return images, intrinsics, extrinsics

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            images = [F.hflip(img) for img in images]
            w, h = images[0].size
            output_intrinsics = np.copy(intrinsics)
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
            extrinsics *= -1
            return images, output_intrinsics, extrinsics
        return images, intrinsics, extrinsics

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomScaleCrop(torch.nn.Module):
    """Randomly zooms images up to 15% and crop them to keep same size as before.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
    """

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        output_intrinsics = np.copy(intrinsics)

        in_w, in_h = images[0].size
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [F.resize(img, (scaled_h, scaled_w)) for img in images]

        # PIL uses a coordinate system with (0, 0) in the upper left corner.
        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [F.crop(img, offset_y, offset_x, in_h, in_w)
                          for img in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics, extrinsics

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropBottom(torch.nn.Module):
    """Randomly zooms images up to 15% and crop them to keep same size as before.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
    """

    def __init__(self, offset_y=160):
        super().__init__()
        self.offset_y = offset_y

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        in_w, in_h = images[0].size

        # PIL uses a coordinate system with (0, 0) in the upper left corner.
        # Bottom trimming does not change the principle points (cx, cy).
        # offset_y = 160

        cropped_images = [F.crop(img, 0, 0, in_h-self.offset_y, in_w)
                          for img in images]

        return cropped_images, intrinsics, extrinsics

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropSides(torch.nn.Module):
    """Randomly zooms images up to 15% and crop them to keep same size as before.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
    """

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        output_intrinsics = np.copy(intrinsics)

        in_w, in_h = images[0].size

        # PIL uses a coordinate system with (0, 0) in the upper left corner.
        # Left trimming shifts the principal point cx.
        offset_x = 140

        cropped_images = [F.crop(img, 0, offset_x, in_h, in_w-2*offset_x)
                          for img in images]

        output_intrinsics[0, 2] -= offset_x
        # output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics, extrinsics

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(torch.nn.Module):
    """Resize the input image to the given size. Input is list of PIL images.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it should have 1 or 2 values")
        self.size = size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        output_intrinsics = np.copy(intrinsics)

        in_w, in_h = images[0].size  # [800,1280]

        # scaled_h, scaled_w = 192, 320
        x_scaling = float(self.size[1])/in_w
        y_scaling = float(self.size[0])/in_h

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        images = [F.resize(img, self.size, self.interpolation)
                  for img in images]

        return images, output_intrinsics, extrinsics

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2})'.format(
            self.size, interpolate_str, self.max_size)


class ToPILImage:
    """Convert a tensor or an ndarray to PIL Image. This transform does not support torchscript.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
            ``short``).
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, images, intrinsics, extrinsics):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL Image: Image converted to PIL Image.
        """
        images = [F.to_pil_image(img, self.mode) for img in images]
        return images, intrinsics, extrinsics

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(
                    "{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(
            torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(
            torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(
            torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(
            torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, imgs, intrinsics, extrinsics):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast,
                            self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                imgs = [F.adjust_brightness(
                    img, brightness_factor) for img in imgs]
            elif fn_id == 1 and contrast_factor is not None:
                imgs = [F.adjust_contrast(img, contrast_factor)
                        for img in imgs]
            elif fn_id == 2 and saturation_factor is not None:
                imgs = [F.adjust_saturation(
                    img, saturation_factor) for img in imgs]
            elif fn_id == 3 and hue_factor is not None:
                imgs = [F.adjust_hue(img, hue_factor) for img in imgs]

        return imgs, intrinsics, extrinsics

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
