import copy
import os
import shutil

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image


'''
The palette is used to convert the segmentation map having values from 0-21 back to paletted image
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

cityscape_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                      250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 0, 130, 180, 220, 20, 60,
                      255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

acdc_palette = [0, 0, 0, 128, 64, 128, 70, 70, 70, 250, 170, 30]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

zero_pad = 256 * 3 - len(cityscape_palette)
for i in range(zero_pad):
    cityscape_palette.append(0)

zero_pad = 256 * 3 - len(acdc_palette)
for i in range(zero_pad):
    acdc_palette.append(0)


def colorize_mask(mask, dataset):
    '''
    Used to convert the segmentation of one channel(mask) back to a paletted image
    '''
    # mask: numpy array of the mask
    assert dataset in ('voc2012', 'cityscapes', 'acdc')
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if (dataset == 'voc2012'):
        new_mask.putpalette(palette)
    elif (dataset == 'cityscapes'):
        new_mask.putpalette(cityscape_palette)
    elif (dataset == 'acdc'):
        new_mask.putpalette(acdc_palette)

    return new_mask

### To convert a paletted image to a tensor image of 3 dimension
### This is because a simple paletted image cannot be viewed with all the details
def PIL_to_tensor(img, dataset):
    '''
    Here img is of the type PIL.Image
    '''
    assert dataset in ('voc2012', 'cityscapes', 'acdc')
    img_arr = np.array(img, dtype='float32')
    new_arr = np.zeros([3, img_arr.shape[0], img_arr.shape[1]], dtype='float32')

    if (dataset == 'voc2012'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = palette[index]
                new_arr[1, i, j] = palette[index+1]
                new_arr[2, i, j] = palette[index+2]
    elif (dataset == 'cityscapes'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = cityscape_palette[index]
                new_arr[1, i, j] = cityscape_palette[index+1]
                new_arr[2, i, j] = cityscape_palette[index+2]
    elif (dataset == 'acdc'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = acdc_palette[index]
                new_arr[1, i, j] = acdc_palette[index+1]
                new_arr[2, i, j] = acdc_palette[index+2]
    
    return_tensor = torch.tensor(new_arr)

    return return_tensor

def smoothen_label(label, alpha, gpu_id):
    '''
    For smoothening of the classification labels
    
    labels : tensor having dimensrions: batch_size*21*H*W filled with zeroes and ones
    '''
    torch.manual_seed(0)
    try:
        smoothen_array = -1*alpha + torch.rand([label.shape[0], label.shape[1], label.shape[2], label.shape[3]]) * (2*alpha)
        smoothen_array = cuda(smoothen_array, gpu_id)
        label = label + smoothen_array
    except:
        smoothen_array = -1*alpha + torch.rand([label.shape[0], label.shape[1], label.shape[2], label.shape[3]]) * (2*alpha)
        label = label + smoothen_array

    return label

'''
To be used to apply gaussian noise in the input to the discriminator
'''
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.zeros(x.size()).normal_() * scale
            x = x + sampled_noise
        return x


# To make directories
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


# To make cuda tensor
def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs


# For Pytorch datasets loader
def create_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], 'Link'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], 'Link'))

    return dirs


def get_traindata_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    return dirs


def get_testdata_link(dataset_dir):
    dirs = {}
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    return dirs


# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
  
def make_one_hot(labels, dataname, gpu_id):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    assert dataname in ('voc2012', 'cityscapes', 'acdc'), 'dataset name should be one of the following: \'voc2012\',given {}'.format(dataname)

    if dataname == 'voc2012':
        C = 21
    elif dataname == 'cityscapes':
        C = 20
    elif dataname == 'acdc':
        C = 4
    else:
        raise NotImplementedError

    labels = labels.long()
    try:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        one_hot = cuda(one_hot, gpu_id)
    except:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target


# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')
