import os
from itertools import cycle

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis
from datasets import VOCDataset, get_transformation
from utils import make_one_hot

'''
Class for CycleGAN with train() as a member function

'''
root = '/home/AP84830/Semi-supervised-cycleGAN/datasets/VOC2012'

class supervised_model(object):
    def __init__(self, args):

        utils.cuda_devices(args.gpu_ids)

        # Define the network 
        self.Gsi = define_Gen(input_nc=3, output_nc=21, ngf=args.ngf, netG='unet_256', norm=args.norm,
                                                 use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)  # for image to segmentation
        self.CE = nn.CrossEntropyLoss()
        self.gsi_optimizer = torch.optim.Adam(self.Gsi.parameters(), lr=args.lr, betas=(0.9, 0.999))

        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Gsi.load_state_dict(ckpt['Gsi'])
            self.gsi_optimizer.load_state_dict(ckpt['gsi_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self, args):
        transform = get_transformation((args.img_height, args.img_width))

        ## let the choice of dataset configurable
        labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform, augmentation=None)

        labeled_loader= DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):
            for i, (l_img, l_gt, _) in enumerate(labeled_loader):
                # step
                step = epoch * len(labeled_loader) + i + 1

                # set train
                self.Gsi.train()

                l_img, l_gt = utils.cuda([l_img, l_gt])

                lab_gt = self.Gsi(l_img)

                # Total generators losses
                fullsupervisedloss = self.CE(lab_gt, l_gt.squeeze(1))
                # Update generators
                self.Gsi.zero_grad()
                fullsupervisedloss.backward()
                self.gsi_optimizer.step()


                print("Epoch: (%3d) (%5d/%5d) | Crossentropy Loss:%.2e" %
                      (epoch, i + 1, len(labeled_loader), fullsupervisedloss.item()))

            # Override the latest checkpoint 
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Gsi': self.Gsi.state_dict(),
                                   'gsi_optimizer': self.gsi_optimizer.state_dict()},
                                  '%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))


class semisuper_cycleGAN(object):
    def __init__(self, args):

        utils.cuda_devices(args.gpu_ids)

        # Define the network 
        # for segmentaion to image
        self.Gis = define_Gen(input_nc=21, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', 
                        norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        # for image to segmentation
        self.Gsi = define_Gen(input_nc=3, output_nc=21, ngf=args.ngf, netG='unet_256', 
                        norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids) 
        self.Di = define_Dis(input_nc=3, ndf=args.ndf, netD= 'n_layers', n_layers_D=3,
                                                     norm=args.norm, gpu_ids=args.gpu_ids)
        self.Ds = define_Dis(input_nc=22, ndf=args.ndf, netD= 'n_layers', n_layers_D=3,
                                                     norm=args.norm, gpu_ids=args.gpu_ids)  # for voc 2012, there are 22 classes 

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.CE = nn.CrossEntropyLoss()

        self.di_optimizer = torch.optim.Adam(self.Di.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.ds_optimizer = torch.optim.Adam(self.Ds.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.gsi_optimizer = torch.optim.Adam(self.Gsi.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.gis_optimizer = torch.optim.Adam(self.Gis.parameters(), lr=args.lr, betas=(0.5, 0.999))

        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Di.load_state_dict(ckpt['Di'])
            self.Ds.load_state_dict(ckpt['Ds'])
            self.Gis.load_state_dict(ckpt['Gis'])
            self.Gsi.load_state_dict(ckpt['Gsi'])
            self.di_optimizer.load_state_dict(ckpt['di_optimizer'])
            self.ds_optimizer.load_state_dict(ckpt['ds_optimizer'])
            self.gis_optimizer.load_state_dict(ckpt['gis_optimizer'])
            self.gsi_optimizer.load_state_dict(ckpt['gsi_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self, args):
        transform = get_transformation((args.img_height, args.img_width))

        ## let the choice of dataset configurable
        labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform, augmentation=None)
        unlabeled_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform,
                                   augmentation=None)
        val_set = VOCDataset(root_path=root, name='val', ratio=0.5, transformation=transform,
                             augmentation=None)

        assert (set(labeled_set.imgs) & set(unlabeled_set.imgs)).__len__() == 0


        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):


            for i, ((l_img, l_gt, _),(unl_img, _, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):


                # step
                step = epoch * min(len(labeled_loader), len(unlabeled_loader)) + i + 1
                # set train
                self.Gis.train()
                self.Gsi.train()

                l_img, unl_img, l_gt = utils.cuda([l_img, unl_img, l_gt])

                ## ================ generator part================================

                # Forward pass through generators
                fake_img = self.Gis(make_one_hot(l_gt, args.dataset).float())
                assert fake_img.shape[1] == 3
                assert fake_img.shape[2] == args.img_height
                assert fake_img.shape[3] == args.img_width

                fake_gt = self.Gsi(unl_img.float())
                lab_gt = self.Gsi(l_img)
                
                assert fake_gt.shape[1] == 22
                assert fake_img.shape[2] == args.img_height
                assert fake_img.shape[3] == args.img_width

                recon_img = self.Gis(fake_gt.float())
                recon_gt = self.Gsi(fake_img.float())


                # Adversarial losses
                fake_img_dis = self.Di(fake_img)
                fake_gt_dis = self.Ds(fake_gt)

                real_label = utils.cuda(Variable(torch.ones(fake_gt_dis.size())))

                ## here is much better to have a cross entropy loss for classification.
                img_gen_loss = self.MSE(fake_img_dis, real_label)
                gt_gen_loss = self.MSE(fake_gt_dis, real_label)

                # Cycle consistency losses


                img_cycle_loss = self.L1(recon_img, unl_img)
                gt_cycle_loss = self.L1(recon_gt, make_one_hot(l_gt, args.dataset ))


                # Total generators losses
                fullsupervisedloss = self.CE(lab_gt, l_gt.squeeze(1)) + self.MSE(fake_img, l_img)
                unsupervisedloss = img_gen_loss + gt_gen_loss + img_cycle_loss * args.lamda + gt_cycle_loss * args.lamda

                gen_loss = args.omega * fullsupervisedloss + unsupervisedloss

                # Update generators
                self.Gis.zero_grad()
                self.Gsi.zero_grad()
                gen_loss.backward()
                self.gis_optimizer.step()
                self.gsi_optimizer.step()

                # Sample from history of generated images
                fake_img = Variable(torch.Tensor(img_fake_sample([fake_img.cpu().data.numpy()])[0]))
                fake_gt = Variable(torch.Tensor(gt_fake_sample([fake_gt.cpu().data.numpy()])[0]))
                fake_img, fake_gt = utils.cuda([fake_img, fake_gt])

                # ================ Discrminator training===============================

                # Forward pass through discriminators 
                unl_img_dis = self.Di(unl_img)
                fake_img_dis = self.Di(fake_img)

                real_gt_dis = self.Ds(make_one_hot(l_gt, args.dataset ))

                fake_gt_dis = self.Ds(fake_gt)
                real_label = utils.cuda(Variable(torch.ones(unl_img_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(fake_img_dis.size())))

                # Discriminator losses
                img_dis_real_loss = self.MSE(unl_img_dis, real_label)
                img_dis_fake_loss = self.MSE(fake_img_dis, fake_label)
                gt_dis_real_loss = self.MSE(real_gt_dis, real_label)
                gt_dis_fake_loss = self.MSE(fake_gt_dis, fake_label)

                # Total discriminators losses
                img_dis_loss = img_dis_real_loss + img_dis_fake_loss
                gt_dis_loss = gt_dis_real_loss + gt_dis_fake_loss

                # Update discriminators
                self.Di.zero_grad()
                self.Ds.zero_grad()
                img_dis_loss.backward()
                gt_dis_loss.backward()
                self.di_optimizer.step()
                self.ds_optimizer.step()


            miou = utils.val(self.Gis, val_loader, nclass=21, nogpu=False)

            print("Epoch: (%3d) (%5d/%5d) | Dis Loss:%.2e | Unlab Gen Loss:%.2e | Lab Gen loss:%.2e" %
                      (epoch, i + 1, min(len(labeled_loader), len(unlabeled_loader)),
                       img_dis_loss+gt_dis_loss, unsupervisedloss, fullsupervisedloss))
            # Override the latest checkpoint 
            utils.save_checkpoint({'miou': miou, 'epoch': epoch + 1,
                                   'Di': self.Di.state_dict(),
                                   'Ds': self.Ds.state_dict(),
                                   'Gis': self.Gis.state_dict(),
                                   'Gsi': self.Gsi.state_dict(),
                                   'di_optimizer': self.di_optimizer.state_dict(),
                                   'ds_optimizer': self.ds_optimizer.state_dict(),
                                   'gis_optimizer': self.gis_optimizer.state_dict(),
                                   'gsi_optimizer': self.gsi_optimizer.state_dict()},
                                  '%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))
