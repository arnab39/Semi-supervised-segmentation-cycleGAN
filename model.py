import os
import itertools
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis, set_grad
from data_utils import VOCDataset, CityscapesDataset, ACDCDataset, get_transformation
from utils import make_one_hot
from tensorboardX import SummaryWriter

'''
Class for CycleGAN with train() as a member function

'''
root = './data/VOC2012'
root_cityscapes = "./data/Cityscape"
root_acdc = './data/ACDC'

### The location for tensorboard visualizations
tensorboard_loc = './tensorboard_results/first_run'

class supervised_model(object):
    def __init__(self, args):

        if args.dataset == 'voc2012':
            self.n_channels = 21
        elif args.dataset == 'cityscapes':
            self.n_channels = 20
        elif args.dataset == 'acdc':
            self.n_channels = 4

        # Define the network 
        self.Gsi = define_Gen(input_nc=3, output_nc=self.n_channels, ngf=args.ngf, netG='resnet_9blocks_softmax', norm=args.norm,
                              use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)  # for image to segmentation

        utils.print_networks([self.Gsi], ['Gsi'])

        self.CE = nn.CrossEntropyLoss()
        self.activation_softmax = nn.Softmax2d()
        self.gsi_optimizer = torch.optim.Adam(self.Gsi.parameters(), lr=args.lr, betas=(0.9, 0.999))

        ### writer for tensorboard
        self.writer_supervised = SummaryWriter(tensorboard_loc + '_supervised')
        self.running_metrics_val = utils.runningScore(self.n_channels)   ## Here are the number of classes in the output

        self.args = args

        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Gsi.load_state_dict(ckpt['Gsi'])
            self.gsi_optimizer.load_state_dict(ckpt['gsi_optimizer'])
            self.best_iou = ckpt['best_iou']
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.best_iou = -100

    def train(self, args):

        transform = get_transformation((self.args.crop_height, self.args.crop_width), resize=True, dataset=args.dataset)

        # let the choice of dataset configurable
        if self.args.dataset == 'voc2012':
            labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform,
                                     augmentation=None)
            val_set = VOCDataset(root_path=root, name='val', ratio=0.5, transformation=transform,
                                 augmentation=None)
            labeled_loader = DataLoader(labeled_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True)
        elif self.args.dataset == 'cityscapes':
            labeled_set = CityscapesDataset(root_path=root_cityscapes, name='label', ratio=0.5, transformation=transform,
                                            augmentation=None)
            val_set = CityscapesDataset(root_path=root_cityscapes, name='val', ratio=0.5, transformation=transform,
                                            augmentation=None)
            labeled_loader = DataLoader(labeled_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        elif self.args.dataset == 'acdc':
            labeled_set = ACDCDataset(root_path=root_acdc, name='label', ratio=0.5, transformation=transform,
                                            augmentation=None)
            val_set = ACDCDataset(root_path=root_acdc, name='val', ratio=0.5, transformation=transform,
                                            augmentation=None)
            labeled_loader = DataLoader(labeled_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, self.args.epochs):
            self.Gsi.train()
            for i, (l_img, l_gt, img_name) in enumerate(labeled_loader):
                # step
                step = epoch * len(labeled_loader) + i + 1

                self.gsi_optimizer.zero_grad()

                l_img, l_gt = utils.cuda([l_img, l_gt], args.gpu_ids)

                lab_gt = self.Gsi(l_img)
                # print('The maximum of the l_gt: ',l_gt.max(),' and the minimum is: ',l_gt.min())
                # print('The labels name: ', img_name)
                #l=l_gt.squeeze(1)
                #print(l.shape,np.amax(l),np.amin(l))
                #print(l.shape,l.unique(),l.min())

                # CE losses
                fullsupervisedloss = self.CE(lab_gt, l_gt.squeeze(1))

                fullsupervisedloss.backward()
                self.gsi_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Crossentropy Loss:%.2e" %
                      (epoch, i + 1, len(labeled_loader), fullsupervisedloss.item()))

                self.writer_supervised.add_scalars('Supervised Loss', {'CE Loss ':fullsupervisedloss}, len(labeled_loader)*epoch + i)

            ### For getting the IoU for the image
            self.Gsi.eval()
            with torch.no_grad():
                for i, (val_img, val_gt, _) in enumerate(val_loader):
                    val_img, val_gt = utils.cuda([val_img, val_gt], args.gpu_ids)

                    outputs = self.Gsi(val_img)
                    
                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = val_gt.squeeze().data.cpu().numpy()

                    self.running_metrics_val.update(gt, pred)
            
            score, class_iou = self.running_metrics_val.get_scores()
           
            self.running_metrics_val.reset()

            ### For displaying the images generated by generator on tensorboard
            val_img, val_gt, _ = iter(val_loader).next()
            val_img, val_gt = utils.cuda([val_img, val_gt], args.gpu_ids)
            with torch.no_grad():
                fake = self.Gsi(val_img).detach()
            fake = self.activation_softmax(fake)
            fake_prediction = fake.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            val_gt = val_gt.cpu()

            ### display_tensor is the final tensor that will be displayed on tensorboard
            display_tensor = torch.zeros([fake.shape[0], 3, fake.shape[2], fake.shape[3]])
            display_tensor_gt = torch.zeros([val_gt.shape[0], 3, val_gt.shape[2], val_gt.shape[3]])
            for i in range(fake_prediction.shape[0]):
                new_img = fake_prediction[i]
                new_img = utils.colorize_mask(new_img, self.args.dataset)   ### So this is the generated image in PIL.Image format
                img_tensor = utils.PIL_to_tensor(new_img, self.args.dataset)
                display_tensor[i, :, :, :] = img_tensor

                display_tensor_gt[i, :, :, :] = val_gt[i]

            self.writer_supervised.add_image('Generated segmented image', torchvision.utils.make_grid(display_tensor, nrow=2, normalize=True), epoch)
            self.writer_supervised.add_image('Ground truth for the image', torchvision.utils.make_grid(display_tensor_gt, nrow=2, normalize=True), epoch)


            if score["Mean IoU : \t"] >= self.best_iou:
                self.best_iou = score["Mean IoU : \t"]
                # Override the latest checkpoint 
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'Gsi': self.Gsi.state_dict(),
                                       'gsi_optimizer': self.gsi_optimizer.state_dict(),
                                       'best_iou': self.best_iou,
                                       'class_iou': class_iou},
                                       '%s/latest_supervised_model.ckpt' % (self.args.checkpoint_dir))
        
        self.writer_supervised.close()


class semisuper_cycleGAN(object):
    def __init__(self, args):

        if args.dataset == 'voc2012':
            self.n_channels = 21
        elif args.dataset == 'cityscapes':
            self.n_channels = 20
        elif args.dataset == 'acdc':
            self.n_channels = 4

        # Define the network 
        #####################################################
        # for segmentaion to image
        self.Gis = define_Gen(input_nc=self.n_channels, output_nc=3, ngf=args.ngf, netG='resnet_9blocks',
                              norm=args.norm, use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
        # for image to segmentation
        self.Gsi = define_Gen(input_nc=3, output_nc=self.n_channels, ngf=args.ngf, netG='resnet_9blocks_softmax',
                              norm=args.norm, use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Di = define_Dis(input_nc=3, ndf=args.ndf, netD='pixel', n_layers_D=3,
                             norm=args.norm, gpu_ids=args.gpu_ids)
        self.Ds = define_Dis(input_nc=1, ndf=args.ndf, netD='pixel', n_layers_D=3,
                             norm=args.norm, gpu_ids=args.gpu_ids)  # for voc 2012, there are 21 classes

        utils.print_networks([self.Gis,self.Gsi,self.Di,self.Ds], ['Gis','Gsi','Di','Ds'])

        self.args = args

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.CE = nn.CrossEntropyLoss()
        self.activation_softmax = nn.Softmax2d()

        ### Tensorboard writer
        self.writer_semisuper = SummaryWriter(tensorboard_loc + '_semisuper')
        self.running_metrics_val = utils.runningScore(self.n_channels)   ## Here 21 is the number of classes

        ### For adding gaussian noise
        self.gauss_noise = utils.GaussianNoise(sigma = 0.2)

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
            self.best_iou = ckpt['best_iou']
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.best_iou = -100



    def train(self, args):
        transform = get_transformation((args.crop_height, args.crop_width), resize=True, dataset=args.dataset)

        # let the choice of dataset configurable
        if self.args.dataset == 'voc2012':
            labeled_set = VOCDataset(root_path=root, name='label', ratio=0.5, transformation=transform,
                                     augmentation=None)
            unlabeled_set = VOCDataset(root_path=root, name='unlabel', ratio=0.5, transformation=transform,
                                       augmentation=None)
            val_set = VOCDataset(root_path=root, name='val', ratio=0.5, transformation=transform,
                                 augmentation=None)
        elif self.args.dataset == 'cityscapes':
            labeled_set = CityscapesDataset(root_path=root_cityscapes, name='label', ratio=0.5, transformation=transform,
                                            augmentation=None)
            unlabeled_set = CityscapesDataset(root_path=root_cityscapes, name='unlabel', ratio=0.5, transformation=transform,
                                            augmentation=None)
            val_set = CityscapesDataset(root_path=root_cityscapes, name='val', ratio=0.5, transformation=transform,
                                            augmentation=None)
        elif self.args.dataset == 'acdc':
            labeled_set = ACDCDataset(root_path=root_acdc, name='label', ratio=0.5, transformation=transform,
                                            augmentation=None)
            unlabeled_set = ACDCDataset(root_path=root_acdc, name='unlabel', ratio=0.5, transformation=transform,
                                            augmentation=None)
            val_set = ACDCDataset(root_path=root_acdc, name='val', ratio=0.5, transformation=transform,
                                            augmentation=None)

        # assert (set(labeled_set.imgs) & set(unlabeled_set.imgs)).__len__() == 0

        '''
        https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510
        ^^ The reason for using drop_last=True so as to obtain an even size of all the batches and
        deleting the last batch with less images
        '''
        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

        img_fake_sample = utils.Sample_from_Pool()
        gt_fake_sample = utils.Sample_from_Pool()

        img_dis_loss, gt_dis_loss, unsupervisedloss, fullsupervisedloss = 0, 0, 0, 0

        ### Variable to regulate the frequency of update between Discriminators and Generators
        counter = 0

        for epoch in range(self.start_epoch, args.epochs):
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            self.Gsi.train()
            self.Gis.train()

            for i, ((l_img, l_gt, _), (unl_img, _, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
                # step
                step = epoch * min(len(labeled_loader), len(unlabeled_loader)) + i + 1

                l_img, unl_img, l_gt = utils.cuda([l_img, unl_img, l_gt], args.gpu_ids)

                # Generator Computations
                ##################################################

                set_grad([self.Di, self.Ds], False)
                self.g_optimizer.zero_grad()

                # Forward pass through generators
                ##################################################
                fake_img = self.Gis(make_one_hot(l_gt, args.dataset, args.gpu_ids).float())
                fake_gt = self.Gsi(unl_img.float())  ### having 21 channels
                lab_gt = self.Gsi(l_img)  ### having 21 channels

                # fake_gt = fake_gt.data.max(1)[1].squeeze_(1).squeeze_(0)  ### will get into no channels
                # fake_gt = fake_gt.unsqueeze(1)   ### will get into 1 channel only
                # fake_gt = make_one_hot(fake_gt, args.dataset, args.gpu_ids)

                lab_loss_CE = self.CE(lab_gt, l_gt.squeeze(1))

                ### Again applying activations in the label
                lab_gt = self.activation_softmax(lab_gt)
                fake_gt = self.activation_softmax(fake_gt)

                recon_img = self.Gis(fake_gt.float())
                recon_lab_img = self.Gis(lab_gt.float())
                recon_gt = self.Gsi(fake_img.float())

                ### Again applying the activations in the label
                ###### JUST APPLY THIS WHEN USING RECONSTRUCTION LOSS OF LABELS AS L1 AGAIN
                # recon_gt = self.activation_softmax(recon_gt)

                ### Printing out the ranges of all the various images
                # print('The range for unl_img: Max: ',unl_img.max(), ' , Min: ', unl_img.min())
                # print('The range for l_gt: Max: ',make_one_hot(l_gt, args.dataset, args.gpu_ids).float().max(), ' , Min: ', make_one_hot(l_gt, args.dataset, args.gpu_ids).float().min())
                # print('The range for fake_gt: Max: ',fake_gt.max(), ' , Min: ', fake_gt.min())
                # print('The range for fake_img: Max: ',fake_img.max(), ' , Min: ', fake_img.min())
                # print('The range for recon_img: Max: ',recon_img.max(), ' , Min: ', recon_img.min())
                # print('The range for recon_gt: Max: ',recon_gt.max(), ' , Min: ', recon_gt.min())

                # img_idt = self.Gis(make_one_hot(l_gt, args.dataset).float())
                # gt_idt = self.Gsi(unl_img.float())

                # assert img_idt.shape == unl_img.shape, ('img_idt: '+str(img_idt.shape)+' unl_img: '+str(unl_img.shape))
                # Identity losses
                ###################################################
                # img_idt_loss = self.L1(img_idt, unl_img) * args.lamda * args.idt_coef
                # gt_idt_loss = self.L1(gt_idt, make_one_hot(l_gt, args.dataset, args.gpu_ids)) * args.lamda * args.idt_coef

                # Adversarial losses
                ###################################################
                fake_img_dis = self.Di(fake_img)

                ### For passing different type of input to Ds
                fake_gt_discriminator = fake_gt.data.max(1)[1].squeeze_(1).squeeze_(0)
                fake_gt_discriminator = fake_gt_discriminator.unsqueeze(1)
                # fake_gt_discriminator = make_one_hot(fake_gt_discriminator, args.dataset, args.gpu_ids)
                fake_gt_dis = self.Ds(fake_gt_discriminator.float())
                # lab_gt_dis = self.Ds(lab_gt)

                real_label = utils.cuda(Variable(torch.ones(fake_gt_dis.size())), args.gpu_ids)

                # here is much better to have a cross entropy loss for classification.
                img_gen_loss = self.MSE(fake_img_dis, real_label)
                gt_gen_loss = self.MSE(fake_gt_dis, real_label)
                # gt_label_gen_loss = self.MSE(lab_gt_dis, real_label)


                # Cycle consistency losses
                ###################################################
                img_cycle_loss = self.L1(recon_img, unl_img)
                gt_cycle_loss = self.CE(recon_gt, l_gt.squeeze(1))
                # lab_img_cycle_loss = self.L1(recon_lab_img, l_img) * args.lamda

                # Total generators losses
                ###################################################
                # lab_loss_CE = self.CE(lab_gt, l_gt.squeeze(1))
                lab_loss_MSE = self.MSE(fake_img, l_img)

                fullsupervisedloss = lab_loss_CE + lab_loss_MSE

                unsupervisedloss = args.adversarial_weight*(img_gen_loss + gt_gen_loss) + img_cycle_loss*args.lamda_img + gt_cycle_loss*args.lamda_gt

                gen_loss = args.gen_weight*(fullsupervisedloss) + unsupervisedloss

                # Update generators
                ###################################################
                gen_loss.backward()

                ### To check the gradient norms
                # Gsi_list = []
                # Gis_list = []
                # for p in list(filter(lambda p: p.grad is not None, self.Gsi.parameters())):
                    # Gsi_list.append(p.grad.data.norm(2).item())
                # for p in list(filter(lambda p: p.grad is not None, self.Gis.parameters())):
                    # Gis_list.append(p.grad.data.norm(2).item())
                # for p in list(filter(lambda p: p.grad is not None, self.Di.parameters())):
                    # Di_list.append(p.grad.data.norm(2).item())
                # for p in list(filter(lambda p: p.grad is not None, self.Ds.parameters())):
                    # Ds_list.append(p.grad.data.norm(2).item())

                # print('During Generator Computations, for Gsi : ', max(Gsi_list))
                # print('During Generator Computations, for Gis : ', max(Gis_list))
                # print('During Generator Computations, for Di : ', max(Di_list))
                # print('During Generator Computations, for Ds : ', max(Ds_list))

                ### clipping the gradients
                # nn.utils.clip_grad_norm_(self.Gsi.parameters(), max_norm=80)                
                # nn.utils.clip_grad_norm_(self.Gis.parameters(), max_norm=80)

                self.g_optimizer.step()


                if counter % 1 == 0:
                    # Discriminator Computations
                    #################################################

                    set_grad([self.Di, self.Ds], True)
                    self.d_optimizer.zero_grad()

                    # Sample from history of generated images
                    #################################################
                    if torch.rand(1) < 0.0:
                        fake_img = self.gauss_noise(fake_img.cpu())
                        fake_gt = self.gauss_noise(fake_gt.cpu())

                    fake_img = Variable(torch.Tensor(img_fake_sample([fake_img.cpu().data.numpy()])[0]))
                    # lab_gt = Variable(torch.Tensor(gt_fake_sample([lab_gt.cpu().data.numpy()])[0]))
                    fake_gt = Variable(torch.Tensor(gt_fake_sample([fake_gt.cpu().data.numpy()])[0]))

                    fake_img, lab_gt, fake_gt = utils.cuda([fake_img, lab_gt, fake_gt], args.gpu_ids)

                    # Forward pass through discriminators
                    #################################################
                    unl_img_dis = self.Di(unl_img)
                    fake_img_dis = self.Di(fake_img)

                    # lab_gt_dis = self.Ds(lab_gt)

                    real_gt_dis = self.Ds(l_gt.float())

                    fake_gt_discriminator = fake_gt.data.max(1)[1].squeeze_(1).squeeze_(0)
                    fake_gt_discriminator = fake_gt_discriminator.unsqueeze(1)
                    # fake_gt_discriminator = make_one_hot(fake_gt_discriminator, args.dataset, args.gpu_ids)
                    fake_gt_dis = self.Ds(fake_gt_discriminator.float())
    
                    real_label = utils.cuda(Variable(torch.ones(unl_img_dis.size())), args.gpu_ids)
                    fake_label = utils.cuda(Variable(torch.zeros(fake_img_dis.size())), args.gpu_ids)

                    # Discriminator losses
                    ##################################################
                    img_dis_real_loss = self.MSE(unl_img_dis, real_label)
                    img_dis_fake_loss = self.MSE(fake_img_dis, fake_label)
                    gt_dis_real_loss = self.MSE(real_gt_dis, real_label)
                    gt_dis_fake_loss = self.MSE(fake_gt_dis, fake_label)
                    # lab_gt_dis_fake_loss = self.MSE(lab_gt_dis, fake_label)

                    # Total discriminators losses
                    img_dis_loss = (img_dis_real_loss + img_dis_fake_loss)*0.5
                    gt_dis_loss = (gt_dis_real_loss + gt_dis_fake_loss)*0.5
                    # lab_gt_dis_loss = (gt_dis_real_loss + lab_gt_dis_fake_loss)*0.33

                    # Update discriminators
                    ##################################################
                    discriminator_loss = args.discriminator_weight * (img_dis_loss + gt_dis_loss)
                    discriminator_loss.backward()

                    ### To check the gradient norms
                    # Di_list = []
                    # Ds_list = []
                    # for p in list(filter(lambda p: p.grad is not None, self.Gsi.parameters())):
                        # Gsi_list.append(p.grad.data.norm(2).item())
                    # for p in list(filter(lambda p: p.grad is not None, self.Gis.parameters())):
                        # Gis_list.append(p.grad.data.norm(2).item())
                    # for p in list(filter(lambda p: p.grad is not None, self.Di.parameters())):
                        # Di_list.append(p.grad.data.norm(2).item())
                    # for p in list(filter(lambda p: p.grad is not None, self.Ds.parameters())):
                        # Ds_list.append(p.grad.data.norm(2).item())

                    # print('During Discriminator Computations, for Gsi : ', max(Gsi_list))
                    # print('During Discriminator Computations, for Gis : ', max(Gis_list))
                    # print('During Discriminator Computations, for Di : ', max(Di_list))
                    # print('During Discriminator Computations, for Ds : ', max(Ds_list))

                    ### clipping the gradient norms
                    # nn.utils.clip_grad_norm_(self.Di.parameters(), max_norm = 8)
                    # nn.utils.clip_grad_norm_(self.Ds.parameters(), max_norm = 8)

                    # lab_gt_dis_loss.backward()
                    self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Dis Loss:%.2e | Unlab Gen Loss:%.2e | Lab Gen loss:%.2e" %
                  (epoch, i + 1, min(len(labeled_loader), len(unlabeled_loader)),
                   img_dis_loss + gt_dis_loss, unsupervisedloss, fullsupervisedloss))
                
                self.writer_semisuper.add_scalars('Dis Loss', {'img_dis_loss':img_dis_loss, 'gt_dis_loss':gt_dis_loss}, len(labeled_loader)*epoch + i)
                self.writer_semisuper.add_scalars('Unlabelled Loss', {'img_gen_loss': img_gen_loss, 'gt_gen_loss':gt_gen_loss, 'img_cycle_loss':img_cycle_loss, 'gt_cycle_loss':gt_cycle_loss}, len(labeled_loader)*epoch + i)
                self.writer_semisuper.add_scalars('Labelled Loss', {'lab_loss_CE':lab_loss_CE, 'lab_loss_MSE':lab_loss_MSE}, len(labeled_loader)*epoch + i)

                counter += 1

            ### For getting the mean IoU
            self.Gsi.eval()
            self.Gis.eval()
            with torch.no_grad():
                for i, (val_img, val_gt, _) in enumerate(val_loader):
                    val_img, val_gt = utils.cuda([val_img, val_gt], args.gpu_ids)

                    outputs = self.Gsi(val_img)

                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = val_gt.squeeze().data.cpu().numpy()

                    self.running_metrics_val.update(gt, pred)

            score, class_iou = self.running_metrics_val.get_scores()

            self.running_metrics_val.reset()

            ### For displaying the images generated by generator on tensorboard using validation images
            val_image, val_gt, _ = iter(val_loader).next()
            val_image, val_gt = utils.cuda([val_image, val_gt], args.gpu_ids)
            with torch.no_grad():
                fake_label = self.Gsi(val_image).detach()
                fake_label = self.activation_softmax(fake_label)
                fake_img = self.Gis(fake_label).detach()

                fake_img_from_labels = self.Gis(make_one_hot(val_gt, args.dataset, args.gpu_ids).float()).detach()
                fake_label_regenerated = self.Gsi(fake_img_from_labels).detach()
                fake_label_regenerated = self.activation_softmax(fake_label_regenerated)
            fake_prediction_label = fake_label.data.max(1)[1].squeeze_(1).cpu().numpy()
            fake_regenerated_label = fake_label_regenerated.data.max(1)[1].squeeze_(1).cpu().numpy()
            val_gt = val_gt.cpu()

            fake_img = fake_img.cpu()
            fake_img_from_labels = fake_img_from_labels.cpu()
            ### Now i am going to revert back the transformation on these images
            if self.args.dataset == 'voc2012' or self.args.dataset == 'cityscapes':
                trans_mean = [0.5, 0.5, 0.5]
                trans_std = [0.5, 0.5, 0.5]
                for i in range(3):
                    fake_img[:, i, :, :] = ((fake_img[:, i, :, :] * trans_std[i]) + trans_mean[i])
                    fake_img_from_labels[:, i, :, :] = ((fake_img_from_labels[:, i, :, :] * trans_std[i]) + trans_mean[i])
            
            elif self.args.dataset == 'acdc':
                trans_mean = [0.5]
                trans_std = [0.5]
                for i in range(1):
                    fake_img[:, i, :, :] = ((fake_img[:, i, :, :] * trans_std[i]) + trans_mean[i])
                    fake_img_from_labels[:, i, :, :] = ((fake_img_from_labels[:, i, :, :] * trans_std[i]) + trans_mean[i])

            ### display_tensor is the final tensor that will be displayed on tensorboard
            display_tensor_label = torch.zeros([fake_label.shape[0], 3, fake_label.shape[2], fake_label.shape[3]])
            display_tensor_gt = torch.zeros([val_gt.shape[0], 3, val_gt.shape[2], val_gt.shape[3]])
            display_tensor_regen_label = torch.zeros([fake_label_regenerated.shape[0], 3, fake_label_regenerated.shape[2], fake_label_regenerated.shape[3]])
            for i in range(fake_prediction_label.shape[0]):
                new_img_label = fake_prediction_label[i]
                new_img_label = utils.colorize_mask(new_img_label, self.args.dataset)   ### So this is the generated image in PIL.Image format
                img_tensor_label = utils.PIL_to_tensor(new_img_label, self.args.dataset)
                display_tensor_label[i, :, :, :] = img_tensor_label

                display_tensor_gt[i, :, :, :] = val_gt[i]

                regen_label = fake_regenerated_label[i]
                regen_label = utils.colorize_mask(regen_label, self.args.dataset)
                regen_tensor_label = utils.PIL_to_tensor(regen_label, self.args.dataset)
                display_tensor_regen_label[i, :, :, :] = regen_tensor_label

            self.writer_semisuper.add_image('Generated segmented image: ', torchvision.utils.make_grid(display_tensor_label, nrow=2, normalize=True), epoch)
            self.writer_semisuper.add_image('Generated image back from segmentation: ', torchvision.utils.make_grid(fake_img, nrow=2, normalize=True), epoch)
            self.writer_semisuper.add_image('Ground truth for the image: ', torchvision.utils.make_grid(display_tensor_gt, nrow=2, normalize=True), epoch)
            self.writer_semisuper.add_image('Image generated from val labels: ', torchvision.utils.make_grid(fake_img_from_labels, nrow=2, normalize=True), epoch)
            self.writer_semisuper.add_image('Labels generated back from the cycle: ', torchvision.utils.make_grid(display_tensor_regen_label, nrow=2, normalize=True), epoch)

            
            if score["Mean IoU : \t"] >= self.best_iou:
                self.best_iou = score["Mean IoU : \t"]

                # Override the latest checkpoint
                #######################################################
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'Di': self.Di.state_dict(),
                                       'Ds': self.Ds.state_dict(),
                                       'Gis': self.Gis.state_dict(),
                                       'Gsi': self.Gsi.state_dict(),
                                       'd_optimizer': self.d_optimizer.state_dict(),
                                       'g_optimizer': self.g_optimizer.state_dict(),
                                       'best_iou': self.best_iou,
                                       'class_iou': class_iou},
                                       '%s/latest_semisuper_cycleGAN.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
        
        self.writer_semisuper.close()
