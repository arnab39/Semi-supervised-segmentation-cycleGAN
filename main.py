import os
from argparse import ArgumentParser
import model as md
from utils import create_link
from testing import test


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=256)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--omega', type=int, default=1)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--model', type=str, default='supervised_model')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/horse2zebra')
    parser.add_argument('--dataset',type=str,choices=['voc2012'],default='voc2012')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--use_sigmoid', action='store_true', help='use sigmoid for discriminator')
    args = parser.parse_args()
    return args


def main():
  args = get_args()
  # set gpu ids
  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      args.gpu_ids.append(id)

  if args.training:
    if args.model == "semisupervised_cycleGAN":
      print("Training semi-supervised cycleGAN")
      model = md.semisuper_cycleGAN(args)
      model.train(args)
    if args.model == "supervised_model":
      print("Training base model")
      model = md.supervised_model(args)
      model.train(args)
  if args.testing:
      print("Testing")
      test(args)


if __name__ == '__main__':
    main()