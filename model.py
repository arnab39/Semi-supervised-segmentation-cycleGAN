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

'''
Class for CycleGAN with train() as a member function

'''


class cycleGAN(object):
    def __init__(self, args):

        utils.cuda_devices([args.gpu_id])

        # Define the network 
        self.Di = Discriminator(in_dim=3)
        self.Ds = Discriminator(in_dim=1)
        self.Gis = Generator(in_dim=1, out_dim=3)  # for segmentaion to image
        self.Gsi = Generator(in_dim=3, out_dim=1)  # for image to segmentation

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

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
        # For transforming the input image
        # transform = transforms.Compose(
        #     [transforms.RandomHorizontalFlip(),
        #      transforms.Resize((args.img_height,args.img_width)),
        #      transforms.ToTensor(),
        #      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        root = '/Users/jizong/workspace/Semi-supervised-cycleGAN/datasets/VOC2012'
        img_size = (128,128)
        transform = get_transformation(img_size)
        labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform, augmentation=None)

        unlabeled_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform,
                                   augmentation=None)
        assert (set(labeled_set.imgs) & set(unlabeled_set.imgs)).__len__() == 0

        labeled_loader_CE = DataLoader(labeled_set, batch_size=1, shuffle=True)
        labeled_loader = DataLoader(labeled_set, batch_size=1, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=1, shuffle=True)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):
            for i, ((_, real_gt, _), (real_img, _, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
                # step
                step = epoch * min(len(labeled_loader), len(unlabeled_loader)) + i + 1

                # set train
                self.Gis.train()
                self.Gsi.train()

                real_img, real_gt = utils.cuda([real_img, real_gt])

                # Forward pass through generators
                fake_img = self.Gis(real_gt.float())
                fake_gt = self.Gsi(real_img.float())

                recon_img = self.Gis(fake_gt.float())
                recon_gt = self.Gsi(fake_img.float())

                # Adversarial losses
                fake_img_dis = self.Di(fake_img)
                fake_gt_dis = self.Ds(fake_gt)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # Cycle consistency losses
                a_cycle_loss = self.L1(a_recon, a_real)
                b_cycle_loss = self.L1(b_recon, b_real)

                # Total generators losses
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss * args.lamda + b_cycle_loss * args.lamda

                # Update generators
                self.Gab.zero_grad()
                self.Gba.zero_grad()
                gen_loss.backward()
                self.gab_optimizer.step()
                self.gba_optimizer.step()

                # Sample from history of generated images
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators 
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # Discriminator losses
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = a_dis_real_loss + a_dis_fake_loss
                b_dis_loss = b_dis_real_loss + b_dis_fake_loss

                # Update discriminators
                self.Da.zero_grad()
                self.Db.zero_grad()
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.da_optimizer.step()
                self.db_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint 
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'da_optimizer': self.da_optimizer.state_dict(),
                                   'db_optimizer': self.db_optimizer.state_dict(),
                                   'gab_optimizer': self.gab_optimizer.state_dict(),
                                   'gba_optimizer': self.gba_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
