import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_image_gray(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('L')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        
        if osp.exists(img_path.replace('/query/', '/query_cloth/')):
            cloth = read_image_gray(img_path.replace('/query/', '/query_cloth/'))########################
        else:
            cloth = read_image_gray(img_path)
        
        if osp.exists(img_path.replace('/query/', '/query_cloth_/')):
            _cloth = read_image_gray(img_path.replace('/query/', '/query_cloth_/'))#######################
        else:
            _cloth = read_image_gray(img_path)
        
        if osp.exists(img_path.replace('/query/', '/query_contour/')):
            contour = read_image_gray(img_path.replace('/query/', '/query_contour/'))
        else:
            contour = read_image_gray(img_path)
        
        if self.transform is not None:
            img, cloth, _cloth, contour = self.transform(img, cloth, _cloth, contour)
        return img, pid, camid, clothes_id, cloth, _cloth, contour#, img_path

class ImageDataset_Train(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id, cloth_path, _cloth_path, contour_path = self.dataset[index]
        img = read_image(img_path)
        cloth = read_image_gray(cloth_path)##################################
        _cloth = read_image_gray(_cloth_path)####################################
        contour = read_image_gray(contour_path)
        # print (np.max(cloth), np.max(_cloth), np.max(contour))
        if self.transform is not None:
            img, cloth, _cloth, contour = self.transform(img, cloth, _cloth, contour)
        return img, pid, camid, clothes_id, cloth, _cloth, contour


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def gray_pil_loader(gray_path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(gray_path, 'rb') as gray_f:
        with Image.open(gray_f) as gray_img:
            return gray_img.convert('L')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def get_gray_image_loader():
    return gray_pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def gray_image_loader(gray_path):
    return gray_pil_loader(gray_path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def gray_video_loader(gray_img_paths, gray_image_loader):
    gray_video = []
    for gray_image_path in gray_img_paths:
        if osp.exists(gray_image_path):
            gray_video.append(gray_image_loader(gray_image_path))
        else:
            return gray_video

    return gray_video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def get_gray_video_loader():
    gray_image_loader = get_gray_image_loader()
    return functools.partial(gray_video_loader, gray_image_loader=gray_image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.gray_loader = get_gray_video_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        # 获取衣服
        path_clip_cloth = [each.replace('session', 'cloth_session') for each in img_paths]
        # 获取非衣服
        path_clip_cloth_ = [each.replace('session', '_cloth_session') for each in img_paths]
        # 获取轮廓
        path_clip_contour = [each.replace('session', 'contour_session') for each in img_paths]

        clip = self.loader(img_paths)#####################
        clip_cloth = self.gray_loader(path_clip_cloth)
        clip_cloth_ = self.gray_loader(path_clip_cloth_)
        clip_contour = self.gray_loader(path_clip_contour)
        
        # print (len(clip), ",", clip[0], "|", len(clip_cloth), ",", clip_cloth[0], "|", len(clip_contour), clip_contour[0])
        # 8 , (128, 256) | 8 , (128, 256) | 8 (128, 256)

        # np.max(clip[0], np.max(clip_cloth), np.max(clip_contour))
        

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip_all = [self.spatial_transform(img1, img2, img3, img4) for img1, img2, img3, img4 in zip(clip, clip_cloth, clip_cloth_, clip_contour)]
        
        clip, clip_cloth, clip_cloth_, clip_contour = [], [], [], []
        for (a,b,c,d) in clip_all:
            clip.append(a)
            clip_cloth.append(b)
            clip_cloth_.append(c)
            clip_contour.append(d)

        # print ('==========================')
        # exit(0)

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip_cloth = torch.stack(clip_cloth, 0).permute(1, 0, 2, 3)
        clip_cloth_ = torch.stack(clip_cloth_, 0).permute(1, 0, 2, 3)
        clip_contour = torch.stack(clip_contour, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id, clip_cloth, clip_cloth_, clip_contour
        else:
            return clip, pid, camid
