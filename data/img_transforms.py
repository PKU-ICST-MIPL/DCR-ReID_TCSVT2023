from torchvision.transforms import *
from PIL import Image
import random
import math
from torchvision.transforms import functional as F
import collections
import torch
import numpy as np


class ResizeWithEqualScale(object):
    """
    Resize an image with equal scale as the original image.

    Args:
        height (int): resized height.
        width (int): resized width.
        interpolation: interpolation manner.
        fill_color (tuple): color for padding.
    """
    def __init__(self, height, width, interpolation=Image.BILINEAR, fill_color=(0,0,0)):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):
        width, height = img.size
        if self.height / self.width >= height / width:
            height = int(self.width * (height / width))
            width = self.width
        else:
            width = int(self.height * (width / height))
            height = self.height 

        resized_img = img.resize((width, height), self.interpolation)
        new_img = Image.new('RGB', (self.width, self.height), self.fill_color)
        new_img.paste(resized_img, (int((self.width - width) / 2), int((self.height - height) / 2)))

        return new_img


class RandomCroping(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, p=0.5, interpolation=Image.BILINEAR):
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img1, img2, img3, contour):
        """
        Args:
            img (PIL Image): Image to be cropped.


        Returns:
            PIL Image: Cropped image.
        """
        width, height = img1.size
        if random.uniform(0, 1) >= self.p:
            return img1, img2, img3, contour
        
        new_width, new_height = int(round(width * 1.125)), int(round(height * 1.125))
        resized_img1 = img1.resize((new_width, new_height), self.interpolation)
        resized_img2 = img2.resize((new_width, new_height), self.interpolation)
        resized_img3 = img3.resize((new_width, new_height), self.interpolation)
        resized_contour = contour.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - width
        y_maxrange = new_height - height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img1 = resized_img1.crop((x1, y1, x1 + width, y1 + height))
        croped_img2 = resized_img2.crop((x1, y1, x1 + width, y1 + height))
        croped_img3 = resized_img3.crop((x1, y1, x1 + width, y1 + height))
        croped_contour = resized_contour.crop((x1, y1, x1 + width, y1 + height))

        return croped_img1, croped_img2, croped_img3, croped_contour


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.

    Reference:
        Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.

    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img1, img2, img3, contour):

        if random.uniform(0, 1) >= self.probability:
            return img1, img2, img3, contour

        for attempt in range(100):
            area = img1.size()[1] * img1.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img1.size()[2] and h < img1.size()[1]:
                x1 = random.randint(0, img1.size()[1] - h)
                y1 = random.randint(0, img1.size()[2] - w)
                if img1.size()[0] == 3:
                    img1[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img1[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img1[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img1[0, x1:x1+h, y1:y1+w] = self.mean[0]

                if img2.size()[0] == 3:
                    img2[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img2[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img2[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img2[0, x1:x1+h, y1:y1+w] = self.mean[0]

                if img3.size()[0] == 3:
                    img3[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img3[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img3[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img3[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    
                contour[0, x1:x1+h, y1:y1+w] = 1.0

                return img1, img2, img3, contour

        return img1, img2, img3, contour

class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img1, img2, img3, contour):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        img1 = F.resize(img1, self.size, self.interpolation)
        img2 = F.resize(img2, self.size, self.interpolation)
        img3 = F.resize(img3, self.size, self.interpolation)
        contour = F.resize(contour, self.size, self.interpolation)
        return img1, img2, img3, contour

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img1, img2, img3, contour):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img1), F.to_tensor(img2), F.to_tensor(img3), F.to_tensor(contour)

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, mean2=0.5, std2=0.5):
        self.mean1 = mean
        self.std1 = std
        self.mean2 = mean2
        self.std2 = std2
        self.mean_con = 0.5 # (0.5, 0.5, 0.5)
        self.std_con = 0.5 # (0.5, 0.5, 0.5)

    def __call__(self, tensor1, tensor2, tensor3, contour):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # print (torch.max(tensor1), torch.min(tensor1))
        # print (torch.max(tensor2), torch.min(tensor2))
        # print (torch.max(tensor3), torch.min(tensor3))
        # print (torch.max(contour), torch.min(contour))
        return F.normalize(tensor1, self.mean1, self.std1), F.normalize(tensor2, self.mean_con, self.std_con), F.normalize(tensor3, self.mean_con, self.std_con), 0.0 - F.normalize(contour, self.mean2, self.std2)

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, img3, contour):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img1), F.hflip(img2), F.hflip(img3), F.hflip(contour)
        return img1, img2, img3, contour

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, img3, contour):
        for idx,t in enumerate(self.transforms):
            img1, img2, img3, contour = t(img1, img2, img3, contour)
        return img1, img2, img3, contour