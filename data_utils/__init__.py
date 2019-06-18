from torchvision.transforms import *
import numpy as np
import torch
import random
from PIL import ImageOps, ImageFilter, ImageEnhance
from .dataloader import VOCDataset, CityscapesDataset


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


class Relabel:
    def __init__(self, olabel, nlabel):
        '''
        Converts a particular label value in the tensor to another.
        Used to convert the background color value to 21 in the case of voc dataset

        Parameters
        ----------
        olabel : old label which needs to be changed
        nlabel : new label which will replace the old label
        '''
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        '''
        Used to change the image from dim(N x H x W) to dim(N x 1 x H x W)
        '''
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=22):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()

        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def PILaugment(img, mask):
    if random.random() > 0.2:
        (w, h) = img.size
        (w_, h_) = mask.size
        assert (w == w_ and h == h_), 'The size should be the same.'
        crop = random.uniform(0.45, 0.75)
        W = int(crop * w)
        H = int(crop * h)
        start_x = w - W
        start_y = h - H
        x_pos = int(random.uniform(0, start_x))
        y_pos = int(random.uniform(0, start_y))
        img = img.crop((x_pos, y_pos, x_pos + W, y_pos + H))
        mask = mask.crop((x_pos, y_pos, x_pos + W, y_pos + H))

    if random.random() > 0.2:
        img = ImageOps.flip(img)
        mask = ImageOps.flip(mask)

    if random.random() > 0.2:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)

    if random.random() > 0.2:
        angle = random.random() * 90 - 45
        img = img.rotate(angle)
        mask = mask.rotate(angle)
    if random.random() > 0.95:
        img = img.filter(ImageFilter.GaussianBlur(2))

    if random.random() > 0.95:
        img = ImageEnhance.Contrast(img).enhance(1)

    if random.random() > 0.95:
        img = ImageEnhance.Brightness(img).enhance(1)

    return img, mask


def get_transformation(size, resize=False):
    '''
    Used to return a transformation based on the size given, that is, returns an apt sized tensor
    If resize = True then Resizing else CenterCrop
    '''
    if resize:
        transfom_lst = [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            Normalize([.5, .5, .5], [.5, .5, .5])
        ]
    else:
        transfom_lst = [
            CenterCrop(size),
            ToTensor(),
            Normalize([.5, .5, .5], [.5, .5, .5])
        ]
    
    input_transform = Compose(transfom_lst)

    if resize:
        target_transform = Compose([
            Resize(size),
            CenterCrop(size),
            ToLabel(),
            Relabel(255, 21)    ## So as to replace the 255(boundaries) label as 21
        ])

    else:
        target_transform = Compose([
            CenterCrop(size),
            ToLabel(),
            Relabel(255, 21)    ## So as to replace the 255(boundaries) label as 21
        ])
    

    transform = {'img': input_transform, 'gt': target_transform}
    return transform
