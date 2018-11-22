import os
from argparse import ArgumentParser
import model as md
from utils import create_link
import test as tst


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--img_height', type=int, default=128)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--omega', type=int, default=1)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--model', type=str, default='supervised_model')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/horse2zebra')
    parser.add_argument('--dataset',type=str,choices=['voc2012'],default='voc2012')
    args = parser.parse_args()
    return args


def main():
  args = get_args()

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
      tst.test(args)


if __name__ == '__main__':
    main()