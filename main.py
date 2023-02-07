from pathlib import Path
import random

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import optim

from config import cli, datasets
import helpers
from torchgear.layers import VAT


def run():
    # Get the command line arguments
    args = cli.parse_commandline_args()
    args = helpers.load_args(args)

    if args.seed is not None:
        # RNG control
        random.seed(args.seed)

        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load the dataset
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    args.num_classes = num_classes

    # Create loaders
    train_loader, eval_loader = helpers.create_data_loaders(
        **dataset_config, args=args)

    # Create Model and Optimiser
    args.device = torch.device('cuda')
    model = helpers.create_model(num_classes, args)
    optimizer = torch.optim.SGD(model.parameters(),
		args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    # Define ramp down epoch
    args.lr_rampdown_epochs = 50
    # print("Epochs: ", args.lr_rampdown_epochs)

    # Information store in epoch results and then saved to file
    epoch_results = np.zeros((args.epochs, 6))

    # Save model params
    writer = SummaryWriter(args.logdir)
    with (Path(args.logdir) / 'params.txt').open('w') as file:
        print('### model ###', file=file)
        print(model, file=file)
        print('### train params ###', file=file)
        print(args, file=file)

    for epoch in range(args.epochs):
        # Supervised main loop
        print("Supervised Training Epoch:", (epoch+1), "/", args.epochs)

        # select training algorithm
        if args.train_algo == "mixup":
            helpers.train_mixup_rampdown(
                train_loader, model, optimizer, epoch, args)
            lr_scheduler.step()
            print("Evaluating the model:", end=" ")
            prec1, prec5 = helpers.validate_mixup_rampdown(
                eval_loader, model, args)
        elif args.train_algo == "vat":
            vat = VAT(epsilon=args.vat_epsilon)
            helpers.train_vat(train_loader, model, optimizer, epoch, vat, args)
            lr_scheduler.step()
            print("Evaluating the model:", end=" ")
            prec1, prec5 = helpers.validate_vat(
                eval_loader, model, args)
        elif args.train_algo == "mixup+vat":
            vat = VAT(epsilon=args.vat_epsilon)
            helpers.train_mixup_vat(train_loader, model, optimizer, epoch, vat, args)
            lr_scheduler.step()
            print("Evaluating the model:", end=" ")
            prec1, prec5 = helpers.validate_mixup_vat(
                eval_loader, model, args)

        epoch_results[epoch, 1] = prec1
        epoch_results[epoch, 2] = prec5

        writer.add_scalar('prec1', prec1, epoch)
        writer.add_scalar('prec5', prec5, epoch)

        # Saving Model
        # pylint: disable=consider-using-f-string
        # torch.save(model.state_dict(), Path(args.logdir) / 'state_dict_epoch_{0:03d}.pth'.format(epoch))
        # if (epoch % 10) == 0 and epoch >= 4900:
        if ((epoch+1) % 10) == 0:
            torch.save(model.state_dict(), Path(args.logdir) /
                       'state_dict_epoch_{0:03d}.pth'.format(epoch+1))


if __name__ == '__main__':
    run()
