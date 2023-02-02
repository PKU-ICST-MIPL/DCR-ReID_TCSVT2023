import random
import math
import numbers
import collections
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, img3, contour):
        for t in self.transforms:
            img1, img2, img3, contour = t(img1, img2, img3, contour)
        return img1, img2, img3, contour

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic1, pic2, pic3, pic4):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        ret1, ret2, ret3, ret4 = False, False, False, False
        if isinstance(pic1, np.ndarray):
            img1 = torch.from_numpy(pic1.transpose((2, 0, 1)))
            img1 = img1.float().div(self.norm_value)
            ret1 = True
        if isinstance(pic2, np.ndarray):
            img2 = torch.from_numpy(pic2.transpose((2, 0, 1)))
            img2 = img2.float().div(self.norm_value)
            ret2 = True
        if isinstance(pic3, np.ndarray):
            img3 = torch.from_numpy(pic3.transpose((2, 0, 1)))
            img3 = img3.float().div(self.norm_value)
            ret3 = True
        if isinstance(pic4, np.ndarray):
            img4 = torch.from_numpy(pic4.transpose((2, 0, 1)))
            img4 = img4.float().div(self.norm_value)
            ret4 = True

        if accimage is not None and isinstance(pic1, accimage.Image) and ret1 == False:
            nppic1 = np.zeros(
                [pic1.channels, pic1.height, pic1.width], dtype=np.float32)
            pic1.copyto(nppic1)
            img1 = torch.from_numpy(nppic1)
            ret1 = True
        if accimage is not None and isinstance(pic2, accimage.Image) and ret2 == False:
            nppic2 = np.zeros(
                [pic2.channels, pic2.height, pic2.width], dtype=np.float32)
            pic2.copyto(nppic2)
            img2 = torch.from_numpy(nppic2)
            ret2 = True
        if accimage is not None and isinstance(pic3, accimage.Image) and ret3 == False:
            nppic3 = np.zeros(
                [pic3.channels, pic3.height, pic3.width], dtype=np.float32)
            pic3.copyto(nppic3)
            img3 = torch.from_numpy(nppic3)
            ret3 = True
        if accimage is not None and isinstance(pic4, accimage.Image) and ret4 == False:
            nppic4 = np.zeros(
                [pic4.channels, pic4.height, pic4.width], dtype=np.float32)
            pic4.copyto(nppic4)
            img4 = torch.from_numpy(nppic4)
            ret4 = True

        if ret1 == False:
            if pic1.mode == 'I':
                img1 = torch.from_numpy(np.array(pic1, np.int32, copy=False))
            elif pic1.mode == 'I;16':
                img1 = torch.from_numpy(np.array(pic1, np.int16, copy=False))
            else:
                img1 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic1.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic1.mode == 'YCbCr':
                nchannel1 = 3
            elif pic1.mode == 'I;16':
                nchannel1 = 1
            else:
                nchannel1 = len(pic1.mode)
            img1 = img1.view(pic1.size[1], pic1.size[0], nchannel1)
            
            img1 = img1.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img1, torch.ByteTensor):
                img1 = img1.float().div(self.norm_value)
            
        if ret2 == False:
            if pic2.mode == 'I':
                img2 = torch.from_numpy(np.array(pic2, np.int32, copy=False))
            elif pic2.mode == 'I;16':
                img2 = torch.from_numpy(np.array(pic2, np.int16, copy=False))
            else:
                img2 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic2.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic2.mode == 'YCbCr':
                nchannel2 = 3
            elif pic2.mode == 'I;16':
                nchannel2 = 1
            else:
                nchannel2 = len(pic2.mode)
            img2 = img2.view(pic2.size[1], pic2.size[0], nchannel2)
            
            img2 = img2.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img2, torch.ByteTensor):
                img2 = img2.float().div(self.norm_value)

        if ret3 == False:
            if pic3.mode == 'I':
                img3 = torch.from_numpy(np.array(pic3, np.int32, copy=False))
            elif pic3.mode == 'I;16':
                img3 = torch.from_numpy(np.array(pic3, np.int16, copy=False))
            else:
                img3 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic3.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic3.mode == 'YCbCr':
                nchannel3 = 3
            elif pic3.mode == 'I;16':
                nchannel3 = 1
            else:
                nchannel3 = len(pic3.mode)
            img3 = img3.view(pic3.size[1], pic3.size[0], nchannel3)
            
            img3 = img3.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img3, torch.ByteTensor):
                img3 = img3.float().div(self.norm_value)
        
        if ret4 == False:
            if pic4.mode == 'I':
                img4 = torch.from_numpy(np.array(pic4, np.int32, copy=False))
            elif pic4.mode == 'I;16':
                img4 = torch.from_numpy(np.array(pic4, np.int16, copy=False))
            else:
                img4 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic4.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic4.mode == 'YCbCr':
                nchannel4 = 3
            elif pic4.mode == 'I;16':
                nchannel4 = 1
            else:
                nchannel4 = len(pic4.mode)
            img4 = img4.view(pic4.size[1], pic4.size[0], nchannel4)
            
            img4 = img4.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img4, torch.ByteTensor):
                img4 = img4.float().div(self.norm_value)
        
        return img1, img2, img3, img4
        

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.mean_con = [0.5] # (0.5, 0.5, 0.5)
        self.std_con = [0.5] # (0.5, 0.5, 0.5)

    def __call__(self, tensor1, tensor2, tensor3, contour):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor1, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(tensor2, self.mean_con, self.std_con):
            t.sub_(m).div_(s)
        for t, m, s in zip(tensor3, self.mean_con, self.std_con):
            t.sub_(m).div_(s)
        for t, m, s in zip(contour, self.mean_con, self.std_con):
            t.sub_(m).div_(s)
        return tensor1, tensor2, tensor3, 0.0 - contour

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
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
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                img1 = img1
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                img1 = img1.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img1 = img1.resize((ow, oh), self.interpolation)

            w, h = img2.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                img2 = img2
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                img2 = img2.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img2 = img2.resize((ow, oh), self.interpolation)

            w, h = img3.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                img3 = img3
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                img3 = img3.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img3 = img3.resize((ow, oh), self.interpolation)

            w, h = contour.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                contour = contour
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                contour = contour.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                contour = contour.resize((ow, oh), self.interpolation)
            return img1, img2, img3, contour
            
        else:
            return img1.resize(self.size[::-1], self.interpolation), img2.resize(self.size[::-1], self.interpolation), img3.resize(self.size[::-1], self.interpolation), contour.resize(self.size[::-1], self.interpolation)

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img1, img2, img3, contour):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT), img3.transpose(Image.FLIP_LEFT_RIGHT), contour.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2, img3, contour

    def randomize_parameters(self):
        self.p = random.random()


class RandomCrop(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, size, p=0.5, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.height, self.width = self.size
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img1, img2, img3, contour):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if not self.cropping:
            return img1.resize((self.width, self.height), self.interpolation), img2.resize((self.width, self.height), self.interpolation), img3.resize((self.width, self.height), self.interpolation), contour.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        
        resized_img1 = img1.resize((new_width, new_height), self.interpolation)
        resized_img2 = img2.resize((new_width, new_height), self.interpolation)
        resized_img3 = img3.resize((new_width, new_height), self.interpolation)
        resized_contour = contour.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(self.tl_x * x_maxrange))
        y1 = int(round(self.tl_y * y_maxrange))

        return resized_img1.crop((x1, y1, x1 + self.width, y1 + self.height)), resized_img2.crop((x1, y1, x1 + self.width, y1 + self.height)), resized_img3.crop((x1, y1, x1 + self.width, y1 + self.height)), resized_contour.crop((x1, y1, x1 + self.width, y1 + self.height))

    def randomize_parameters(self):
        self.cropping = random.uniform(0, 1) < self.p
        self.tl_x = random.random()
        self.tl_y = random.random()


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
    
    def __init__(self, height=256, width=128, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.height = height
        self.width = width
       
    def __call__(self, img1, img2, img3, contour):
        if self.re:
            return img1, img2, img3, contour

        if img1.size()[0] == 3:
            img1[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
            img1[1, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[1]
            img1[2, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[2]
        else:
            img1[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
        
        if img2.size()[0] == 3:
            img2[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
            img2[1, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[1]
            img2[2, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[2]
        else:
            img2[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]

        if img3.size()[0] == 3:
            img3[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
            img3[1, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[1]
            img3[2, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[2]
        else:
            img3[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
        
        contour[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = 1.0

        return img1, img2, img3, contour

    def randomize_parameters(self):
        self.re = random.uniform(0, 1) < self.probability
        self.h, self.w, self.x1, self.y1 = 0, 0, 0, 0
        whether_re = False
        if self.re:
            for attempt in range(100):
                area = self.height*self.width

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)

                self.h = int(round(math.sqrt(target_area * aspect_ratio)))
                self.w = int(round(math.sqrt(target_area / aspect_ratio)))
                if self.w < self.width and self.h < self.height:
                    self.x1 = random.randint(0, self.height - self.h)
                    self.y1 = random.randint(0, self.width - self.w)
                    whether_re = True
                    break

        self.re = whether_re