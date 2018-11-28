from datasets.dataloader import VOCDataset
from datasets import get_transformation, PILaugment
from torchvision.transforms import ToPILImage
from utils import make_one_hot
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    test_onehot()
