import os
import itertools
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis, set_grad
from data_utils import VOCDataset, CityscapesDataset, get_transformation
from utils import make_one_hot
from tensorboardX import SummaryWriter

'''
Class for CycleGAN with train() as a member function

'''
root = '/home/AP84830/Semi-supervised-cycleGAN/data/VOC2012'
root_cityscapes = "Cityspaces"

### The location for tensorboard visualizations
tensorboard_loc = '/home/AP84830/Semi-supervised-cycleGAN-aniket/tensorboard_results/first_run'

class supervised_model(object):
    def __init__(self, args):

        # Define the network 
        self.Gsi = define_Gen(input_nc=3, output_nc=22, ngf=args.ngf, netG='unet_128', norm=args.norm,
                              use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)  # for image to segmentation

        utils.print_networks([self.Gsi], ['Gsi'])

        self.CE = nn.CrossEntropyLoss()
        self.gsi_optimizer = torch.optim.Adam(self.Gsi.parameters(), lr=args.lr, betas=(0.9, 0.999))

        ### writer for tensorboard
        self.writer_supervised = SummaryWriter(tensorboard_loc + '_supervised')

        self.args = args

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
        transform = get_transformation((self.args.crop_height, self.args.crop_width), resize=True)

        # let the choice of dataset configurable
        if self.args.dataset == 'voc2012':
            labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform,
                                     augmentation=None)
            labeled_loader = DataLoader(labeled_set, batch_size=self.args.batch_size, shuffle=True)
        elif self.args.dataset == 'cityscapes':
            labeled_set = CityscapesDataset(root_path=root_cityscapes, split='train', is_transform=True,
                                            augmentation=None)
            labeled_loader = DataLoader(labeled_set, batch_size=self.args.batch_size, shuffle=True)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, self.args.epochs):
            for i, (l_img, l_gt, _) in enumerate(labeled_loader):
                # step
                step = epoch * len(labeled_loader) + i + 1

                self.gsi_optimizer.zero_grad()

                l_img, l_gt = utils.cuda([l_img, l_gt])

                lab_gt = self.Gsi(l_img)
                #print(lab_gt.shape,lab_gt.max(),lab_gt.min())
                #l=l_gt.squeeze(1)
                #print(l.shape,np.amax(l),np.amin(l))
                #print(l.shape,l.unique(),l.min())

                # CE losses
                fullsupervisedloss = self.CE(lab_gt, l_gt.squeeze(1))

                fullsupervisedloss.backward()
                self.gsi_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Crossentropy Loss:%.2e" %
                      (epoch, i + 1, len(labeled_loader), fullsupervisedloss.item()))

                self.writer_supervised.add_scalar('Supervised Loss', fullsupervisedloss, i)

            # Override the latest checkpoint 
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Gsi': self.Gsi.state_dict(),
                                   'gsi_optimizer': self.gsi_optimizer.state_dict()},
                                  '%s/latest_supervised_model.ckpt' % (self.args.checkpoint_dir))
        
        self.writer_supervised.close()


class semisuper_cycleGAN(object):
    def __init__(self, args):

        # Define the network 
        #####################################################
        # for segmentaion to image
        self.Gis = define_Gen(input_nc=22, output_nc=3, ngf=args.ngf, netG='resnet_9blocks',
                              norm=args.norm, use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
        # for image to segmentation
        self.Gsi = define_Gen(input_nc=3, output_nc=22, ngf=args.ngf, netG='unet_128',
                              norm=args.norm, use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Di = define_Dis(input_nc=3, ndf=args.ndf, netD='n_layers', n_layers_D=3,
                             norm=args.norm, gpu_ids=args.gpu_ids)
        self.Ds = define_Dis(input_nc=22, ndf=args.ndf, netD='n_layers', n_layers_D=3,
                             norm=args.norm, gpu_ids=args.gpu_ids)  # for voc 2012, there are 22 classes

        utils.print_networks([self.Gis,self.Gsi,self.Di,self.Ds], ['Gis','Gsi','Di','Ds'])

        self.args = args

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.CE = nn.CrossEntropyLoss()

        ### Tensorboard writer
        self.writer_semisuper = SummaryWriter(tensorboard_loc + '_semisuper')

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gis.parameters(),self.Gsi.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Di.parameters(),self.Ds.parameters()), lr=args.lr, betas=(0.5, 0.999))
        

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        
        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Di.load_state_dict(ckpt['Di'])
            self.Ds.load_state_dict(ckpt['Ds'])
            self.Gis.load_state_dict(ckpt['Gis'])
            self.Gsi.load_state_dict(ckpt['Gsi'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0



    def train(self, args):
        transform = get_transformation((args.crop_height, args.crop_width), resize=True)

        # let the choice of dataset configurable
        if self.args.dataset == 'voc2012':
            labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform,
                                     augmentation=None)
            unlabeled_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform,
                                       augmentation=None)
            val_set = VOCDataset(root_path=root, name='val', ratio=0.5, transformation=transform,
                                 augmentation=None)
        elif self.args.dataset == 'cityscapes':
            labeled_set = CityscapesDataset(root_path=root_cityscapes, split='train', is_transform=True,
                                            augmentation=None)
            unlabeled_set = CityscapesDataset(root_path=root_cityscapes, split='val', is_transform=True,
                                              augmentation=None)
            val_set = CityscapesDataset(root_path=root_cityscapes, split='test', is_transform=True,
                                        augmentation=None)

        assert (set(labeled_set.imgs) & set(unlabeled_set.imgs)).__len__() == 0

        '''
        https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510
        ^^ The reason for using drop_last=True so as to obtain an even size of all the batches and
        deleting the last batch with less images
        '''
        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        img_dis_loss, gt_dis_loss, unsupervisedloss, fullsupervisedloss = 0, 0, 0, 0

        for epoch in range(self.start_epoch, args.epochs):
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, ((l_img, l_gt, _), (unl_img, _, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
                # step
                step = epoch * min(len(labeled_loader), len(unlabeled_loader)) + i + 1

                l_img, unl_img, l_gt = utils.cuda([l_img, unl_img, l_gt])

                # Generator Computations
                ##################################################

                set_grad([self.Di, self.Ds], False)
                self.g_optimizer.zero_grad()

                # Forward pass through generators
                ##################################################
                fake_img = self.Gis(make_one_hot(l_gt, args.dataset).float())
                fake_gt = self.Gsi(unl_img.float())


                lab_gt = self.Gsi(l_img)

                recon_img = self.Gis(fake_gt.float())
                recon_lab_img = self.Gis(lab_gt.float())
                recon_gt = self.Gsi(fake_img.float())

                img_idt = self.Gis(make_one_hot(l_gt, args.dataset).float())
                gt_idt = self.Gsi(unl_img.float())

                assert img_idt.shape == unl_img.shape, ('img_idt: '+str(img_idt.shape)+' unl_img: '+str(unl_img.shape))
                # Identity losses
                ###################################################
                img_idt_loss = self.L1(img_idt, unl_img) * args.lamda * args.idt_coef
                gt_idt_loss = self.L1(gt_idt, make_one_hot(l_gt, args.dataset)) * args.lamda * args.idt_coef

                # Adversarial losses
                ###################################################
                fake_img_dis = self.Di(fake_img)
                fake_gt_dis = self.Ds(fake_gt)

                real_label = utils.cuda(Variable(torch.ones(fake_gt_dis.size())))

                # here is much better to have a cross entropy loss for classification.
                img_gen_loss = self.MSE(fake_img_dis, real_label)
                gt_gen_loss = self.MSE(fake_gt_dis, real_label)


                # Cycle consistency losses
                ###################################################
                img_cycle_loss = self.L1(recon_img, unl_img) * args.lamda
                gt_cycle_loss = self.L1(recon_gt, make_one_hot(l_gt, args.dataset)) * args.lamda

                # Total generators losses
                ###################################################
                fullsupervisedloss = self.CE(lab_gt, l_gt.squeeze(1)) + self.MSE(fake_img, l_img)

                unsupervisedloss = img_gen_loss + gt_gen_loss + img_cycle_loss + gt_cycle_loss + img_idt_loss + gt_idt_loss 

                gen_loss = args.omega * fullsupervisedloss + unsupervisedloss

                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()


                # Discriminator Computations
                #################################################

                set_grad([self.Di, self.Ds], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                fake_img = Variable(torch.Tensor(img_fake_sample([fake_img.cpu().data.numpy()])[0]))
                fake_gt = Variable(torch.Tensor(gt_fake_sample([fake_gt.cpu().data.numpy()])[0]))
                fake_img, fake_gt = utils.cuda([fake_img, fake_gt])

                # Forward pass through discriminators
                #################################################
                unl_img_dis = self.Di(unl_img)
                fake_img_dis = self.Di(fake_img)

                real_gt_dis = self.Ds(make_one_hot(l_gt, args.dataset))
                fake_gt_dis = self.Ds(fake_gt)

                real_label = utils.cuda(Variable(torch.ones(unl_img_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(fake_img_dis.size())))

                # Discriminator losses
                ##################################################
                img_dis_real_loss = self.MSE(unl_img_dis, real_label)
                img_dis_fake_loss = self.MSE(fake_img_dis, fake_label)
                gt_dis_real_loss = self.MSE(real_gt_dis, real_label)
                gt_dis_fake_loss = self.MSE(fake_gt_dis, fake_label)

                # Total discriminators losses
                img_dis_loss = img_dis_real_loss + img_dis_fake_loss
                gt_dis_loss = gt_dis_real_loss + gt_dis_fake_loss

                # Update discriminators
                ##################################################
                img_dis_loss.backward()
                gt_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Dis Loss:%.2e | Unlab Gen Loss:%.2e | Lab Gen loss:%.2e" %
                  (epoch, i + 1, min(len(labeled_loader), len(unlabeled_loader)),
                   img_dis_loss + gt_dis_loss, unsupervisedloss, fullsupervisedloss))
                
                self.writer_semisuper.add_scalar('Dis Loss', img_dis_loss + gt_dis_loss, i)
                self.writer_semisuper.add_scalar('Unlabelled Loss', unsupervisedloss, i)
                self.writer_semisuper.add_scalar('Labelled Loss', fullsupervisedloss)

            #miou = utils.val(self.Gis, val_loader, nclass=21, nogpu=False)

            
            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Di': self.Di.state_dict(),
                                   'Ds': self.Ds.state_dict(),
                                   'Gis': self.Gis.state_dict(),
                                   'Gsi': self.Gsi.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
        
        self.writer_semisuper.close()
