from datasets.dataloader import VOCDataset
from datasets import get_transformation, PILaugment
from torchvision.transforms import ToPILImage


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


if __name__ == '__main__':
    test_vocdataset()

