import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

from data_utils import VOCDataset, CityscapesDataset
from data_utils import get_transformation
from data_utils.augmentations import *
from utils import make_one_hot
from torch.utils.data import DataLoader


def test_vocdataset():
    root = '../data/VOC2012'
    img_size = 256
    transform = get_transformation(img_size)
    voc = VOCDataset(root_path=root, name='label', ratio=1, transformation=transform, augmentation=None)
    transforms.ToPILImage()(voc[0][0]).show()
    print(voc[0][0].shape)
    assert voc[0][0].shape[0] == 3
    assert voc[0][0].shape[1] == img_size
    assert voc[0][0].shape[2] == img_size
    target = voc[0][1]


def test_onehot():
    root = '/Users/jizong/workspace/Semi-supervised-cycleGAN/data_utils/VOC2012'
    img_size = 256
    batchsize = 2
    transform = get_transformation(img_size)
    voc = VOCDataset(root_path=root, name='label', ratio=1, transformation=transform, augmentation=None)
    voc_loader = DataLoader(voc, batch_size=batchsize, shuffle=True)
    img = voc[0][0]
    assert img.size().__len__() == 3
    assert img.size(0) == 3
    assert img.size(1) == img_size
    assert img.size(2) == img_size
    img, gt = iter(voc_loader).__next__()[0:2]
    assert gt.shape.__len__() == 4
    assert gt.shape[0] == batchsize
    assert gt.shape[1] == 1
    assert gt.shape[2] == img_size
    assert gt.shape[3] == img_size
    onehot_gt = make_one_hot(gt, 'voc2012')

    # visulization for the first one image
    plt.imshow(img[0].squeeze()[0].numpy());
    plt.show()
    plt.imshow(gt[0].squeeze().numpy());
    plt.show()

    onehot_gt = onehot_gt[0]
    for c in range(onehot_gt.shape[0]):
        channel = onehot_gt[c]
        if channel.sum() > 0:
            plt.imshow(channel.squeeze().numpy(), cmap='gray')
            plt.show()

    pass


def test_cityscapes():
    img_size = (512, 1024)
    augmentations = Compose([Scale(2048), RandomRotate(10)])
    local_path = "../data/Cityspaces"
    cityscape = CityscapesDataset(local_path, is_transform=True, img_size=img_size, augmentation=augmentations)
    bs = 4
    trainloader = DataLoader(cityscape, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(cityscape.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()


if __name__ == '__main__':
    # test_vocdataset()
    test_cityscapes()
    print()
