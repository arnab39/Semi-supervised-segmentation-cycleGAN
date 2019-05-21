import os
import torch
import numpy as np
from metric import metric
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from PIL import Image
from arch import define_Gen
from data_utils import VOCDataset, get_transformation

root = '/home/AP84830/Semi-supervised-cycleGAN/data/VOC2012'

def test(args):
    transform = get_transformation((args.crop_height, args.crop_width), resize=True)

    ## let the choice of dataset configurable
    test_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform, augmentation=None)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    Gsi = define_Gen(input_nc=3, output_nc=22, ngf=args.ngf, netG='unet_128', 
                                    norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)

    Gsi = utils.cuda(Gsi)

    ### dict containing IoU of every test image
    IoU = {}

    if(args.model == 'supervised_model'):

        ### loading the checkpoint
        try:
            ckpt = utils.load_checkpoint('%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))
            Gsi.load_state_dict(ckpt['Gsi'])
        except:
            print(' [*] No checkpoint!')

        ### run
        Gsi.eval()
        for i, (image_test, real_segmentation, image_name) in enumerate(test_loader):
            image_test = utils.cuda(image_test)
            seg_map = Gsi(image_test)

            prediction = seg_map.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()   ### To convert from 22 --> 1 channel
            for j in range(prediction.shape[0]):
                new_img = prediction[j]     ### Taking a particular image from the batch
                new_img = utils.colorize_mask(new_img)   ### So as to convert it back to a paletted image

                real_segmentation_img = Image.fromarray(real_segmentation[j].squeeze_(0).cpu().numpy().astype(np.uint8))

                ### getting IoU of this particular image
                res = metric(real_segmentation_img, new_img)

                IoU[image_name[j]] = res

                ### Now the new_img is PIL.Image
                new_img.save(os.path.join(args.results_dir+'/supervised/'+image_name[j]+'.png'))

            
            print('Epoch-', str(i+1), ' Done!')
        
        torch.save(os.path.join(args.results_dir+'/supervised/'+'accuracy.ckpt'))

    elif(args.model == 'semisupervised_cycleGAN'):

        ### loading the checkpoint
        try:
            ckpt = utils.load_checkpoint('%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))
            Gsi.load_state_dict(ckpt['Gsi'])
        except:
            print(' [*] No checkpoint!')
        
        ### run
        Gsi.eval()
        for i, (image_test, real_segmentation, image_name) in enumerate(test_loader):
            image_test = utils.cuda(image_test)
            seg_map = Gsi(image_test)

            prediction = seg_map.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()   ### To convert from 22 --> 1 channel
            for j in range(prediction.shape[0]):
                new_img = prediction[j]     ### Taking a particular image from the batch
                new_img = utils.colorize_mask(new_img)   ### So as to convert it back to a paletted image

                real_segmentation_img = Image.fromarray(real_segmentation[j].squeeze_(0).cpu().numpy().astype(np.uint8))

                ### getting IoU of this particular image
                res = metric(real_segmentation_img, new_img)

                IoU[image_name[j]] = res

                ### Now the new_img is PIL.Image
                new_img.save(os.path.join(args.results_dir+'/unsupervised/'+image_name[j]+'.png'))
            
            print('Epoch-', str(i+1), ' Done!')
        
        torch.save(os.path.join(args.results_dir+'/unsupervised/'+'accuracy.ckpt'))
