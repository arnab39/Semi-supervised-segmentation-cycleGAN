import os
import pandas as pd
import numpy as np
import torch
import scipy.misc as m
import scipy.io as sio

from torch.utils.data import Dataset
from utils import recursive_glob
from PIL import Image
from .augmentations import *


class VOCDataset(Dataset):
    '''
    We assume that there will be txt to note all the image names

    
    color map:
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
    12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor,
    21=boundaries(self-defined)

    Also it will return an image and ground truth as it is present in the image form, so ground truth won't 
    be in one-hot form and rather would be a 2D tensor. To convert the labels in one-hot form in the training
    code we will be calling the function 'make_one_hot' function of utils.py

    '''

    split_ratio = [0.85, 0.15]
    '''
    this split ratio is for the train (including the labeled and unlabeled) and the val dataset
    '''

    @classmethod
    def reset_split_ratio(cls, new_ratio):
        assert new_ratio.__len__() == 2, "should input 2 float numbers indicating percentage for train and test dataset"
        assert np.array(new_ratio).sum() == 1, 'New split ratio should be normalized, given %s' % str(new_ratio)
        assert np.array(new_ratio).min() > 0 and np.array(new_ratio).max() < 1
        cls.split_ratio = new_ratio

    def __init__(self, root_path, name='label', ratio=0.5, transformation=None, augmentation=None):
        super(VOCDataset, self).__init__()
        self.root_path = root_path
        self.ratio = ratio
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        assert name in ('label', 'unlabel','val', 'test'), 'dataset name should be restricted in "label", "unlabel", "test" and "val", given %s' % name
        assert 0 <= ratio <= 1, 'the ratio between "labeled" and "unlabeled" should be between 0 and 1, given %.1f' % ratio
        np.random.seed(1)  ### Because of this we are not getting repeated images for labelled and unlabelled data

        if self.name != 'test':
            total_imgs = pd.read_table(os.path.join(self.root_path, 'ImageSets/Segmentation', 'trainvalAug.txt')).values.reshape(-1)
        
            train_imgs = np.random.choice(total_imgs, size=int(self.__class__.split_ratio[0] * total_imgs.__len__()),replace=False)
        
            val_imgs = [x for x in total_imgs if x not in train_imgs]
        
            labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)
        
            unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 0.5:
                new_ratio = round((self.ratio/(1-self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = unlabeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int((excess_ratio - int(excess_ratio))*unlabeled_imgs.__len__()), replace=False))
                unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 0.5:
                new_ratio = round(((1-self.ratio)/(self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = labeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int((excess_ratio - int(excess_ratio))*labeled_imgs.__len__()), replace=False))
                labeled_imgs += (new_list_1 + new_list_2)

        if self.name == 'test':
            test_imgs = pd.read_table(os.path.join(self.root_path, 'ImageSets/Segmentation', 'test.txt')).values.reshape(-1)
            # test_imgs = np.array(test_imgs)

        if self.name == 'label':
            self.imgs = labeled_imgs
        elif self.name == 'unlabel':
            self.imgs = unlabeled_imgs
        elif self.name == 'val':
            self.imgs = val_imgs
        elif self.name == 'test':
            self.imgs = test_imgs
        else:
            raise ('{} not defined'.format(self.name))
            
        self.gts = self.imgs

    def __getitem__(self, index):

        if self.name == 'test':
            img_path = os.path.join(self.root_path, 'JPEGImages', self.imgs[index] + '.jpg')

            img = Image.open(img_path).convert('RGB')

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)
            
            return img, self.imgs[index]

        else:
            img_path = os.path.join(self.root_path, 'JPEGImages', self.imgs[index] + '.jpg')
            gt_path = os.path.join(self.root_path, 'SegmentationClassAug', self.gts[index] + '.png')

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path) #.convert('P')

            if self.augmentation is not None:
                img, gt = self.augmentation(img, gt)

            if self.transformation:
                img = self.transformation['img'](img)
                gt = self.transformation['gt'](gt)

            return img, gt, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class CityscapesDataset(Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]
    ]

    label_colours = dict(zip(range(19), colors))

    split_ratio = [0.85, 0.15]
    '''
    this split ratio is for the train (including the labeled and unlabeled) and the val dataset
    '''

    def __init__(
        self,
        root_path,
        name="train",
        ratio=0.5,
        transformation=False,
        augmentation=None
    ):
        self.root = root_path
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        self.n_classes = 20
        self.ratio = ratio
        self.files = {}

        assert name in ('label', 'unlabel','val', 'test'), 'dataset name should be restricted in "label", "unlabel", "test" and "val", given %s' % name

        if self.name != 'test':
            self.images_base = os.path.join(self.root, "leftImg8bit", 'trainval')
            self.annotations_base = os.path.join(
                self.root, "gtFine", 'trainval'
            )
        else:
            self.images_base = os.path.join(self.root, "leftImg8bit", 'test')
            # self.annotations_base = os.path.join(
            #     self.root, "gtFine", 'test'
            # )

        np.random.seed(1) 

        if self.name != 'test':
            total_imgs = recursive_glob(rootdir=self.images_base, suffix=".png")
            total_imgs = np.array(total_imgs)

            train_imgs = np.random.choice(total_imgs, size=int(self.__class__.split_ratio[0] * total_imgs.__len__()),replace=False)
            val_imgs = [x for x in total_imgs if x not in train_imgs]
            
            labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)

            unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 0.5:
                new_ratio = round((self.ratio/(1-self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = unlabeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int((excess_ratio - int(excess_ratio))*unlabeled_imgs.__len__()), replace=False))
                unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 0.5:
                new_ratio = round(((1-self.ratio)/(self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = labeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int((excess_ratio - int(excess_ratio))*labeled_imgs.__len__()), replace=False))
                labeled_imgs += (new_list_1 + new_list_2)

        else:
            test_imgs = recursive_glob(rootdir=self.images_base, suffix=".png")

        if(self.name == 'label'):
            self.files[name] = list(labeled_imgs)
        elif(self.name == 'unlabel'):
            self.files[name] = list(unlabeled_imgs)
        elif(self.name == 'val'):
            self.files[name] = list(val_imgs)
        elif(self.name == 'test'):
            self.files[name] = list(test_imgs)
        
        '''
        This pattern for the various classes has been borrowed from the official repo for Cityscapes dataset
        You can see it here: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        '''
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "unlabelled"
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[name]:
            raise Exception(
                "No files for name=[%s] found in %s" % (name, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[name]), name))

    def __len__(self):
        """__len__"""
        return len(self.files[self.name])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        if self.name == 'test':
            img_path = self.files[self.name][index].rstrip()

            img = Image.open(img_path).convert('RGB')

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            return img, img_path[34:].rstrip('.png')   ### These numbers have been hard coded so as to get a suitable name for the model

        else:
            img_path = self.files[self.name][index].rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )

            img = Image.open(img_path).convert('RGB')
            lbl = Image.open(lbl_path)

            if self.augmentation is not None:
                img, lbl = self.augmentation(img, lbl)

            if self.transformation:
                img = self.transformation['img'](img)
                lbl = self.transformation['gt'](lbl)

            lbl = self.encode_segmap(lbl)

            return img, lbl, img_path[38:].rstrip('.png')   ### These numbers have been hard coded so as to get a suitable name for the model

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to ignore index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask[mask == self.ignore_index] = 19   ### Just a mapping between the two color values
        return mask


class ACDCDataset(Dataset):
    '''
    The dataloader for ACDC dataset
    '''

    split_ratio = [0.85, 0.15]
    '''
    this split ratio is for the train (including the labeled and unlabeled) and the val dataset
    '''

    def __init__(self, root_path, name='label', ratio=0.5, transformation=None, augmentation=None):
        super(ACDCDataset, self).__init__()
        self.root = root_path
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        self.ratio = ratio
        self.files = {}

        if self.name != 'test':
            self.images_base = os.path.join(self.root, 'training')
            self.annotations_base = os.path.join(self.root, 'training_gt')
        else:
            self.images_base = os.path.join(self.root, 'testing')
            # self.annotations_base = os.path.join(
            #     self.root, "gtFine", 'test'
            # )

        np.random.seed(1) 

        if self.name != 'test':
            total_imgs = os.listdir(self.images_base)
            total_imgs = np.array(total_imgs)

            train_imgs = np.random.choice(total_imgs, size=int(self.__class__.split_ratio[0] * total_imgs.__len__()),replace=False)
            val_imgs = [x for x in total_imgs if x not in train_imgs]
            
            labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)

            unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 0.5:
                new_ratio = round((self.ratio/(1-self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = unlabeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int((excess_ratio - int(excess_ratio))*unlabeled_imgs.__len__()), replace=False))
                unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 0.5:
                new_ratio = round(((1-self.ratio)/(self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = labeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int((excess_ratio - int(excess_ratio))*labeled_imgs.__len__()), replace=False))
                labeled_imgs += (new_list_1 + new_list_2)

        else:
            test_imgs = os.listdir(self.images_base)

        if(self.name == 'label'):
            self.files[name] = list(labeled_imgs)
        elif(self.name == 'unlabel'):
            self.files[name] = list(unlabeled_imgs)
        elif(self.name == 'val'):
            self.files[name] = list(val_imgs)
        elif(self.name == 'test'):
            self.files[name] = list(test_imgs)

        if not self.files[name]:
            raise Exception(
                "No files for name=[%s] found in %s" % (name, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[name]), name))

    def __len__(self):
        """__len__"""
        return len(self.files[self.name])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        if self.name == 'test':
            img_path = os.path.join(self.images_base, self.files[self.name][index])

            img = Image.open(img_path)

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            return img, self.files[self.name][index].rstrip('.jpg') 

        else:
            img_path = os.path.join(self.images_base, self.files[self.name][index])
            lbl_path = os.path.join(self.annotations_base, self.files[self.name][index].rstrip('.jpg') + '.png')

            img = Image.open(img_path)
            lbl = Image.open(lbl_path)

            if self.augmentation is not None:
                img, lbl = self.augmentation(img, lbl)

            if self.transformation:
                img = self.transformation['img'](img)
                lbl = self.transformation['gt'](lbl)

            return img, lbl, self.files[self.name][index].rstrip('.jpg')  
