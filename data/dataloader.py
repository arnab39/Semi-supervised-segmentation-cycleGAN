import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CustomizedDataset(Dataset):
    split_ratio = [0.85, 0.15]

    @classmethod
    def reset_split_ratio(cls, new_ratio):
        assert new_ratio.__len__() == 2, "should input 2 float numbers indicating percentage for train and test dataset"
        assert np.array(new_ratio).sum() == 1, 'New split ratio should be normalized, given %s' % str(new_ratio)
        assert np.array(new_ratio).min() > 0 and np.array(new_ratio).max() < 1
        cls.split_ratio = new_ratio

    def __init__(self, root_path, name, ratio=0.5):
        super(CustomizedDataset, self).__init__()

        self.root_path = root_path
        self.ratio = ratio
        self.name = name

        assert name in ('label', 'unlab',
                        'val'), 'dataset name should be restricted in "label", "unlabeled" and "val", given %s' % name
        assert 0 <= ratio <= 1, 'the ratio between "labeled" and "unlabeled" should be between 0 and 1, given %.1f' % ratio
        np.random.seed(1)
        total_imgs = [x for x in os.listdir(os.path.join(self.root_path, 'train')) if
                      x.endswith('.jpg')]
        train_imgs = np.random.choice(total_imgs, size=int(self.__class__.split_ratio[0] * total_imgs.__len__()),
                                      replace=False)
        val_imgs = [x for x in total_imgs if x not in train_imgs]
        labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
        unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

        if self.name == 'label':
            self.imgs = labeled_imgs
        elif self.name == "unlabel":
            self.imgs = unlabeled_imgs
        else:
            self.imgs = val_imgs
        self.gts = [x.split('.')[0] for x in self.imgs]

    def __getitem__(self, index):
        img_path = os.path.join('train', self.imgs[index])
        gt_path = os.path.join('groundtruth', self.gts[index])

        img = Image.open(os.path.join(self.root_path, img_path)).convert('RGB')
        gt = Image.open(os.path.join(self.root_path, gt_path))

    def __len__(self):
        return len(self.imgs)


