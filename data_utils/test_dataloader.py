from datasets.dataloader import VOCDataset
from datasets.cityscapes_dataloader import CityscapesDatasetRefactored
from datasets import get_transformation, PILaugment
from datasets.augmentations import *
from torchvision.transforms import ToPILImage, Compose, ToTensor
from utils import make_one_hot
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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


def test_cityscapes():
    root = '../datasets/Cityspaces'
    img_size = 256
    transform = get_transformation(img_size)
    cityscape = CityscapesDatasetRefactored(root_path=root, name='train', transformation=transform, augmentation=None)

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


def test_onehot():
    root = '/Users/jizong/workspace/Semi-supervised-cycleGAN/datasets/VOC2012'
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


def testing_transformations():
    root = '../datasets/Cityspaces'
    img_size = 256
    transform = get_transformation(img_size)
    input_transform = Compose([ToTensor()])
    target_transform = Compose([ToTensor()])
    transform_ = {'img': input_transform, 'gt': target_transform}

    cityscape = CityscapesDatasetRefactored(root_path=root, name='train', transformation=transform_, augmentation=None)

    # validate the length of images and ground-truth
    assert len(cityscape.imgs) == len(cityscape.gts)

    fig = plt.figure()

    for i in range(len(cityscape)):
        imgs, gts, path = cityscape[i]
        print(path)
        print(i, imgs.shape, type(imgs), gts.squeeze(0).shape)
        npimgs = np.transpose(imgs.numpy(), (1, 2, 0))
        npgts = gts.squeeze(0).numpy()
        print(i, npimgs.shape, type(npgts), npgts.shape)

        ax = fig.add_subplot(1, 4, i + 1)
        imgplot = plt.imshow(npimgs, interpolation='nearest')
        plt.tight_layout()
        ax.set_title('Image #{}'.format(i))
        ax.axis('off')

        ax = fig.add_subplot(2, 4, i + 1)
        imgplot = plt.imshow(npgts, interpolation='nearest')
        plt.tight_layout()
        ax.set_title('GT #{}'.format(i))
        ax.axis('off')

        if i == 3:
            plt.show()
            break


def testing__refactored_cityscapes():
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10)])
    local_path = "../datasets/Cityspaces"
    dst = CityscapesDatasetRefactored(local_path, is_transform=True, img_size=(512, 1024), augmentations=augmentations)
    bs = 4
    trainloader = DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()


if __name__ == '__main__':
    testing__refactored_cityscapes()
    print()
