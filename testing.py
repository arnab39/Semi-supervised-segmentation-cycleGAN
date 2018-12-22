import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from arch import define_Gen
from datasets import VOCDataset, get_transformation

root = '/home/AP84830/Semi-supervised-cycleGAN/datasets/VOC2012'

def test(args):
    utils.cuda_devices(args.gpu_ids)
    transform = get_transformation((args.img_height, args.img_width))

    ## let the choice of dataset configurable
    test_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform, augmentation=None)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    Gsi = define_Gen(input_nc=3, output_nc=22, ngf=args.ngf, netG='unet_256', 
                                    norm=args.norm, use_dropout= not args.no_dropout)

    utils.cuda([Gsi])


    try:
        ckpt = utils.load_checkpoint('%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))
        Gsi.load_state_dict(ckpt['Gsi'])
    except:
        print(' [*] No checkpoint!')


    """ run """
    image_test = Variable(iter(test_loader).next()[0], requires_grad=True)
    image_test = utils.cuda([image_test])
    print(image_test.shape)
            

    Gsi.eval()

    with torch.no_grad():
        seg_test = Gsi(image_test)

    pic = (torch.cat([image_test, seg_test], dim=0).data + 1) / 2.0

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    torchvision.utils.save_image(pic, args.results_dir+'/sample.jpg', nrow=1)

