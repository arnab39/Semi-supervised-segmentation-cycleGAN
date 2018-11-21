import os, pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class VOCDataset(Dataset):
    '''
    We assume that there will be txt to note all the image names

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

    def __init__(self, root_path, name, ratio=0.5, transformation=None, augmentation=None):
        super(VOCDataset, self).__init__()
        self.root_path = root_path
        self.ratio = ratio
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        assert name in ('label', 'unlabel',
                        'val'), 'dataset name should be restricted in "label", "unlabeled" and "val", given %s' % name
        assert 0 <= ratio <= 1, 'the ratio between "labeled" and "unlabeled" should be between 0 and 1, given %.1f' % ratio
        np.random.seed(1)
        total_imgs = pd.read_table(
            os.path.join(self.root_path, 'ImageSets', 'Segmentation', 'trainval.txt')).values.reshape(-1)
        train_imgs = np.random.choice(total_imgs, size=int(self.__class__.split_ratio[0] * total_imgs.__len__()),
                                      replace=False)
        val_imgs = [x for x in total_imgs if x not in train_imgs]
        labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
        unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

        if self.name == 'label':
            self.imgs = labeled_imgs
        elif self.name == "unlabel":
            self.imgs = unlabeled_imgs
        elif self.name =='val':
            self.imgs = val_imgs
        else:
            raise ('{} not defined'.format(self.name))
        self.gts = self.imgs

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, 'JPEGImages', self.imgs[index] + '.jpg')
        gt_path = os.path.join(self.root_path, 'SegmentationClass', self.gts[index] + '.png')

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('P')

        if self.augmentation is not None:
            img,gt = self.augmentation(img, gt)

        if self.transformation:
            img = self.transformation['img'](img)
            gt = self.transformation['gt'](gt)

        return img, gt, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    vocdataset = VOCDataset('/Users/jizong/workspace/Semi-supervised-cycleGAN/datasets/VOC2012', 'label')
