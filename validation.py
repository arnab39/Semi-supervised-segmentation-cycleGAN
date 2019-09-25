import os
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from utils import make_one_hot
from PIL import Image
from arch import define_Gen
from data_utils import VOCDataset, CityscapesDataset, ACDCDataset, get_transformation

root = './data/VOC2012'
root_cityscapes = './data/Cityscape'
root_acdc = './data/ACDC'

def validation(args):

    ### For selecting the number of channels
    if args.dataset == 'voc2012':
        n_channels = 21
    elif args.dataset == 'cityscapes':
        n_channels = 20
    elif args.dataset == 'acdc':
        n_channels = 4

    transform = get_transformation((args.crop_height, args.crop_width), resize=True, dataset=args.dataset)

    ## let the choice of dataset configurable
    if args.dataset == 'voc2012':
        val_set = VOCDataset(root_path=root, name='val', ratio=0.5, transformation=transform, augmentation=None)
    elif args.dataset == 'cityscapes':
        val_set = CityscapesDataset(root_path=root_cityscapes, name='val', ratio=0.5, transformation=transform, augmentation=None)
    elif args.dataset == 'acdc':
        val_set = ACDCDataset(root_path=root_acdc, name='val', ratio=0.5, transformation=transform, augmentation=None)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    Gsi = define_Gen(input_nc=3, output_nc=n_channels, ngf=args.ngf, netG='deeplab', 
                                    norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)

    Gis = define_Gen(input_nc=n_channels, output_nc=3, ngf=args.ngf, netG='deeplab',
                              norm=args.norm, use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)

    ### best_iou
    best_iou = 0

    ### Interpolation
    interp = nn.Upsample(size = (args.crop_height, args.crop_width), mode='bilinear', align_corners=True)

    ### Softmax activation
    activation_softmax = nn.Softmax2d()
    activation_tanh = nn.Tanh()

    if(args.model == 'supervised_model'):

        ### loading the checkpoint
        try:
            ckpt = utils.load_checkpoint('%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))
            Gsi.load_state_dict(ckpt['Gsi'])
            best_iou = ckpt['best_iou']

        except:
            print(' [*] No checkpoint!')

        ### run
        Gsi.eval()
        for i, (image_test, real_segmentation, image_name) in enumerate(val_loader):
            image_test = utils.cuda(image_test, args.gpu_ids)
            seg_map = Gsi(image_test)
            seg_map = interp(seg_map)
            seg_map = activation_softmax(seg_map)

            prediction = seg_map.data.max(1)[1].squeeze_(1).cpu().numpy()   ### To convert from 22 --> 1 channel
            for j in range(prediction.shape[0]):
                new_img = prediction[j]     ### Taking a particular image from the batch
                new_img = utils.colorize_mask(new_img, args.dataset)   ### So as to convert it back to a paletted image

                ### Now the new_img is PIL.Image
                new_img.save(os.path.join(args.validation_dir+'/supervised/'+image_name[j]+'.png'))

            
            print('Epoch-', str(i+1), ' Done!')
        
        print('The iou of the resulting segment maps: ', str(best_iou))


    elif(args.model == 'semisupervised_cycleGAN'):

        ### loading the checkpoint
        try:
            ckpt = utils.load_checkpoint('%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))
            Gsi.load_state_dict(ckpt['Gsi'])
            Gis.load_state_dict(ckpt['Gis'])
            best_iou = ckpt['best_iou']

        except:
            print(' [*] No checkpoint!')

        ### run
        Gsi.eval()
        for i, (image_test, real_segmentation, image_name) in enumerate(val_loader):
            image_test, real_segmentation = utils.cuda([image_test, real_segmentation], args.gpu_ids)
            seg_map = Gsi(image_test)
            seg_map = interp(seg_map)
            seg_map = activation_softmax(seg_map)
            fake_img = Gis(seg_map).detach()
            fake_img = interp(fake_img)
            fake_img = activation_tanh(fake_img)

            fake_img_from_labels = Gis(make_one_hot(real_segmentation, args.dataset, args.gpu_ids).float()).detach()
            fake_img_from_labels = interp(fake_img_from_labels)
            fake_img_from_labels = activation_tanh(fake_img_from_labels)
            fake_label_regenerated = Gsi(fake_img_from_labels).detach()
            fake_label_regenerated = interp(fake_label_regenerated)
            fake_label_regenerated = activation_softmax(fake_label_regenerated)

            prediction = seg_map.data.max(1)[1].squeeze_(1).cpu().numpy()   ### To convert from 22 --> 1 channel
            fake_regenerated_label = fake_label_regenerated.data.max(1)[1].squeeze_(1).cpu().numpy()

            fake_img = fake_img.cpu()
            fake_img_from_labels = fake_img_from_labels.cpu()

            ### Now i am going to revert back the transformation on these images
            if args.dataset == 'voc2012' or args.dataset == 'cityscapes':
                trans_mean = [0.5, 0.5, 0.5]
                trans_std = [0.5, 0.5, 0.5]
                for k in range(3):
                    fake_img[:, k, :, :] = ((fake_img[:, k, :, :] * trans_std[k]) + trans_mean[k])
                    fake_img_from_labels[:, k, :, :] = ((fake_img_from_labels[:, k, :, :] * trans_std[k]) + trans_mean[k])

            elif args.dataset == 'acdc':
                trans_mean = [0.5]
                trans_std = [0.5]
                for k in range(1):
                    fake_img[:, k, :, :] = ((fake_img[:, k, :, :] * trans_std[k]) + trans_mean[k])
                    fake_img_from_labels[:, k, :, :] = ((fake_img_from_labels[:, k, :, :] * trans_std[k]) + trans_mean[k])

            for j in range(prediction.shape[0]):
                new_img = prediction[j]     ### Taking a particular image from the batch
                new_img = utils.colorize_mask(new_img, args.dataset)   ### So as to convert it back to a paletted image

                regen_label = fake_regenerated_label[j]
                regen_label = utils.colorize_mask(regen_label, args.dataset)

                ### Now the new_img is PIL.Image
                new_img.save(os.path.join(args.validation_dir+'/unsupervised/generated_labels/'+image_name[j]+'.png'))
                regen_label.save(os.path.join(args.validation_dir+'/unsupervised/regenerated_labels/'+image_name[j]+'.png'))
                torchvision.utils.save_image(fake_img[j], os.path.join(args.validation_dir+'/unsupervised/regenerated_image/'+image_name[j]+'.jpg'))
                torchvision.utils.save_image(fake_img_from_labels[j], os.path.join(args.validation_dir+'/unsupervised/image_from_labels/'+image_name[j]+'.jpg'))
            
            print('Epoch-', str(i+1), ' Done!')
        
        print('The iou of the resulting segment maps: ', str(best_iou))
        
