import argparse
import time

from . import datasets


def create_parser():
    parser = argparse.ArgumentParser(
        description='Training or Transfer Learning of Models')
    parser.add_argument('--dataset', metavar='DATASET', default='star',
                        choices=datasets.__all__,
                        help='dataset: ' +
                        ' | '.join(datasets.__all__) +
                        ' (default: imagenet)')
    parser.add_argument('--model', type=str,
                        help='the model architecture to use')
    parser.add_argument('--train_algo', type=str, default='vat',
                        help='the training algorithm to use, vat or mixup')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='test',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--label-split', default=10, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', default=True, type=bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled instances')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='mixup alpha for beta dis')
    parser.add_argument('--aug-num', default=1,
                        type=int, help="number of augs")
    parser.add_argument('--vat_lambda', type=float, default=1.)
    parser.add_argument('--vat_epsilon', type=float, default=0.3)
    parser.add_argument('--progress', default=False,
                        type=bool, help='progress bar on or off')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', default='runs/{}'.format(time.time()))
    parser.add_argument('--weights_path', help="wieght path to eval")
    parser.add_argument('--experiment', 
                        help="experiment type: rndweights, finetunefc, initweights (this is a temporary argument, will be removed later)")
    return parser


def parse_commandline_args():
    return create_parser().parse_args()
