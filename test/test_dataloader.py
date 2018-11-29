from datasets.dataloader import VOCDataset, CityscapesDataset
from datasets import get_transformation, PILaugment
from torchvision.transforms import ToPILImage

import numpy as np


def test_vocdataset():
    root = '../datasets/VOC2012'
    img_size = 256
    transform = get_transformation(img_size)
    voc = VOCDataset(root_path=root, name='label', ratio=1, transformation=transform, augmentation=None)
    ToPILImage()(voc[0][0]).show()
    print(voc[0][0].shape)
    assert voc[0][0].shape[0] == 3
    assert voc[0][0].shape[1] == img_size
    assert voc[0][0].shape[2] == img_size
    target = voc[0][1]
    print()

def test_cityscapes():
    root = '../datasets/Cityspaces'
    img_size = 256
    transform = get_transformation(img_size)
    cityscape = CityscapesDataset(root_path=root, name='train', transformation=transform, augmentation=None)

    # validate the length of images and ground-truth
    assert len(cityscape.imgs) == len(cityscape.gts)

    np.random.seed(1)
    idx = np.random.randint(len(cityscape.imgs))
    # validate file name in image and ground-truth
    img2validate = cityscape.imgs[idx]
    gt2validate = cityscape.gts[idx].replace('/gtFine', '/leftImg8bit').replace('_gtFine_labelIds', '_leftImg8bit')
    assert img2validate == gt2validate

    ToPILImage()(cityscape[0][0]).show()
    print(cityscape[0][0].shape)
    assert cityscape[0][0].shape[0] == 3
    assert cityscape[0][0].shape[1] == img_size
    assert cityscape[0][0].shape[2] == img_size
    target = cityscape[0][1]
    print()


if __name__ == '__main__':
    # test_vocdataset()
    test_cityscapes()

