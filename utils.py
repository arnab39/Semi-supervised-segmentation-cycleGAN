import copy
import os
import shutil

import numpy as np
import torch
from PIL import Image


'''
The palette is used to convert the segmentation map having values from 0-21 back to paletted image
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    '''
    Used to convert the segmentation of one channel(mask) back to a paletted image
    '''
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


# To make directories
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]
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
  
def make_one_hot(labels, dataname):
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
    assert dataname in ('voc2012'), 'dataset name should be one of the following: \'voc2012\',given {}'.format(dataname)

    if dataname == 'voc2012':
        C = 22
    else:
        raise NotImplementedError

    labels = labels.long()
    try:
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
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


import numpy as np
import torch
from torch.autograd import Variable


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return mean_iu, cls_iu


def val(model, valoader, nclass=21, nogpu=False):
    model.eval()
    gts, preds = [], []
    for img_id, (img, gt_mask, _) in enumerate(valoader):
        gt_mask = gt_mask.numpy()[0]
        if nogpu:
            img = Variable(img, volatile=True)
        else:
            img = Variable(img.cuda(), volatile=True)
        out_pred_map = model(img)

        # Get hard prediction
        if nogpu:
            soft_pred = out_pred_map.data.numpy()[0]
        else:
            soft_pred = out_pred_map.data.cpu().numpy()[0]

        soft_pred = soft_pred[:, :gt_mask.shape[0], :gt_mask.shape[1]]
        hard_pred = np.argmax(soft_pred, axis=0).astype(np.uint8)
        for gt_, pred_ in zip(gt_mask, hard_pred):
            gts.append(gt_)
            preds.append(pred_)
    miou, _ = scores(gts, preds, n_class=nclass)
    model.train()

    return miou

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
