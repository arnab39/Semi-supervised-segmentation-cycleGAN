import os
from itertools import cycle

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from arch import Generator, Discriminator
from datasets import PILaugment, VOCDataset, get_transformation
from utils import make_one_hot

'''
Class for CycleGAN with train() as a member function

'''


class cycleGAN(object):
    def __init__(self, args):

        utils.cuda_devices([args.gpu_id])

        # Define the network 
        self.Di = Discriminator(in_dim=3)
        self.Ds = Discriminator(in_dim=22)  # for voc 2012, there are 22 classes
        self.Gis = Generator(in_dim=22, out_dim=3)  # for segmentaion to image
        self.Gsi = Generator(in_dim=3, out_dim=22)  # for image to segmentation

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.CE = nn.CrossEntropyLoss()

        utils.cuda([self.Di, self.Ds, self.Gis, self.Gsi])

        self.di_optimizer = torch.optim.Adam(self.Di.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.ds_optimizer = torch.optim.Adam(self.Ds.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.gsi_optimizer = torch.optim.Adam(self.Gsi.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.gis_optimizer = torch.optim.Adam(self.Gis.parameters(), lr=args.lr, betas=(0.5, 0.999))

        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
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
        dataname = args.dataset
        root = '/Users/jizong/workspace/Semi-supervised-cycleGAN/datasets/VOC2012'
        transform = get_transformation((args.img_height, args.img_width))

        ## let the choice of dataset configurable
        labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform, augmentation=None)
        unlabeled_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform,
                                   augmentation=None)

        ##
        assert (set(labeled_set.imgs) & set(unlabeled_set.imgs)).__len__() == 0

        labeled_loader_CE = DataLoader(labeled_set, batch_size=1, shuffle=True)
        labeled_loader = DataLoader(labeled_set, batch_size=1, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=1, shuffle=True)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):
            for i, ((_, real_gt, _), (real_img, _, _), (l_img, l_gt, _)) in enumerate(
                    zip(labeled_loader, unlabeled_loader, labeled_loader_CE)):
                # step
                step = epoch * min(len(labeled_loader), len(unlabeled_loader)) + i + 1
                print(real_gt.max())
                # set train
                self.Gis.train()
                self.Gsi.train()

                real_img, real_gt = utils.cuda([real_img, real_gt])

                ## ================ generator part================================
                # Forward pass through generators
                fake_img = self.Gis(make_one_hot(real_gt, dataname).float())
                assert fake_img.shape[1] == 3
                assert fake_img.shape[2] == args.img_height
                assert fake_img.shape[3] == args.img_width

                fake_gt = self.Gsi(real_img.float())
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
                a_gen_loss = self.MSE(fake_img_dis, real_label)
                b_gen_loss = self.MSE(fake_gt_dis, real_label)

                # Cycle consistency losses
                img_cycle_loss = self.L1(recon_img, real_img)
                gt_cycle_loss = self.L1(recon_gt, make_one_hot(real_gt, dataname ))

                # Total generators losses
                gen_loss = a_gen_loss + b_gen_loss + img_cycle_loss * args.lamda + gt_cycle_loss * args.lamda

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

                # Forward pass through discriminators 
                real_img_dis = self.Di(real_img)
                fake_img_dis = self.Di(fake_img)
                real_gt_dis = self.Ds(make_one_hot(real_gt, dataname ))
                fake_gt_dis = self.Ds(fake_gt)
                real_label = utils.cuda(Variable(torch.ones(real_img_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(fake_img_dis.size())))

                # Discriminator losses
                img_dis_real_loss = self.MSE(real_img_dis, real_label)
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

                # ================fully supervised training=================
                l_img, l_gt = utils.cuda([l_img, l_gt])
                fake_gt = self.Gsi(l_img)
                fake_img = self.Gis(make_one_hot(l_gt, dataname ))
                fullsupervisedloss = self.CE(fake_gt, l_gt.squeeze(1)) + self.MSE(fake_img, l_img)
                ## here the categorical loss should be set as CE.
                self.Gis.zero_grad()
                self.Gsi.zero_grad()
                fullsupervisedloss.backward()
                self.gis_optimizer.step()
                self.gsi_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e | FS loss:%.2e" %
                      (epoch, i + 1, min(len(labeled_loader), len(unlabeled_loader)),
                       gen_loss, img_dis_loss + gt_dis_loss, fullsupervisedloss.item()))

            # Override the latest checkpoint 
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Di': self.Di.state_dict(),
                                   'Ds': self.Ds.state_dict(),
                                   'Gis': self.Gis.state_dict(),
                                   'Gsi': self.Gsi.state_dict(),
                                   'di_optimizer': self.di_optimizer.state_dict(),
                                   'ds_optimizer': self.ds_optimizer.state_dict(),
                                   'gis_optimizer': self.gis_optimizer.state_dict(),
                                   'gsi_optimizer': self.gsi_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
