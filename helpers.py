import itertools
import os
import time

import numpy as np
import torchvision.models as models
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from config.utils import *
import lp.db_eval as db_eval
import lp.db_train as db_train
from models import resnet10


class StreamBatchSampler(Sampler):

    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_eternally(self.primary_indices)
        return (primary_batch for (primary_batch)
                in grouper(primary_iter, self.primary_batch_size)
                )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def create_data_loaders(weak_transformation, strong_transformation,
                        eval_transformation,
                        datadir,
                        args):

    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    with open(args.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())

    dataset = db_train.DBSL(traindir, labels, False, args.aug_num,
                            eval_transformation, weak_transformation, strong_transformation)
    sampler = SubsetRandomSampler(dataset.labeled_idx)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=False)

    eval_dataset = db_eval.DBE(evaldir, False, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False)

    return train_loader, eval_loader


# Create Model
def create_model(num_classes, args):
    model = resnet10(num_classes)
    # model = resnet50(num_classes)
    # model = build_wideresnet(28,2,0,num_classes)

    # model_choice = args.model

    # if model_choice == "resnet18":
    #     model = resnet18(num_classes)

    # elif model_choice == "resnet50":
    #     model = resnet50(num_classes)

    # elif model_choice == "wrn-28-2":
    #     model = build_wideresnet(28,2,0,num_classes)

    # elif model_choice == "wrn-28-8":
    #     model = build_wideresnet(28,8,0,num_classes)

    # model = nn.DataParallel(model)
    
    experiment = args.experiment

    # now training efficientnet
    # if experiment == "rndweights":
    #     model = torch.hub.load(
    #         'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
    # elif experiment == "finetunefc":
    #     model = torch.hub.load(
    #         'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     model.classifier.fc.requires_grad_(True)
    # elif experiment == "initweights":
    #     model = torch.hub.load(
    #         'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    # num_ftrs = model.classifier.fc.in_features
    # model.classifier.fc = nn.Linear(num_ftrs, num_classes)

    # # now training resnet50
    # if experiment == "rndweights":
    #     model = models.resnet50(pretrained=False)
    # elif experiment == "finetunefc":
    #     model = models.resnet50(pretrained=True)
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     model.fc.requires_grad_(True)
    # elif experiment == "initweights":
    #     model = models.resnet50(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)

    model.to(args.device)
    cudnn.benchmark = True
    return model


def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2))/np.sqrt(2)


def mixup_data(x_1, index, lam):
    mixed_x_1 = lam * x_1 + (1 - lam) * x_1[index, :]
    return mixed_x_1


def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

##### ------- training with mixup + cosine rampdown (a warm-up technique) ------- #####


def train_mixup_rampdown(train_loader, model, optimizer, epoch, args):
    model.train()

    if args.progress == True:
        from tqdm import tqdm
        tk0 = tqdm(train_loader, desc="Training Progress " +
                   str(epoch+1) + "/" + str(args.epochs), unit="batch")

    for i, (aug_images, target) in enumerate(tk0):
        target = target.to(args.device)
        # Create the mix
        alpha = args.alpha
        index = torch.randperm(args.batch_size, device=args.device)
        lam = np.random.beta(alpha, alpha)
        target_a, target_b = target, target[index]

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        # Loop over the batches
        count = 0
        for batch in aug_images:
            batch = batch.to(args.device)
            m_batch = mixup_data(batch, index, lam)
            # class_logit , _  = model(m_batch)
            class_logit = model(m_batch)
            if count == 0:
                loss_sum = mixup_criterion(
                    class_logit.double(), target_a, target_b, lam).mean()
            else:
                loss_sum += mixup_criterion(class_logit.double(),
                                            target_a, target_b, lam).mean()

            count += 1

        loss = loss_sum / (args.aug_num)
        loss.backward()
        optimizer.step()


def validate_mixup_rampdown(eval_loader, model, args):
    meters = AverageMeterSet()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # outputs,_ = model(inputs)
            outputs = model(inputs)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            meters.update('top1', prec1.item(), batch_size)
            meters.update('error1', 100.0 - prec1.item(), batch_size)
            meters.update('top5', prec5.item(), batch_size)
            meters.update('error5', 100.0 - prec5.item(), batch_size)

        print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
              .format(top1=meters['top1'], top5=meters['top5']))

    return meters['top1'].avg, meters['top5'].avg


##### ------- training with VAT (Virtual Adversarial Training, our CR default) ------- #####
def train_vat(train_loader, model, optimizer, epoch, vat, args):
    model.train()

    if args.progress == True:
        from tqdm import tqdm
        tk0 = tqdm(train_loader, desc="Training Progress " +
                   str(epoch+1) + "/" + str(args.epochs), unit="batch")

    for i, (aug_images, target) in enumerate(tk0):
        target = target.to(args.device)
        optimizer.zero_grad()
        loss_sum = 0
        for batch in aug_images:
            batch = batch.to(args.device)
            output = F.log_softmax(model(batch), dim=1)
            likelihood_loss = F.nll_loss(output, target)
            vat_loss = vat.forward(model, batch)
            loss_sum += likelihood_loss + args.vat_lambda * vat_loss
        loss = loss_sum / (args.aug_num)
        loss.backward()
        optimizer.step()

        del aug_images, output, likelihood_loss, vat_loss, loss, target


def validate_vat(eval_loader, model, args):
    meters = AverageMeterSet()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = F.log_softmax(model(inputs), dim=1)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5)) 
            meters.update('top1', prec1.item(), batch_size)
            meters.update('error1', 100.0 - prec1.item(), batch_size)
            meters.update('top5', prec5.item(), batch_size)
            meters.update('error5', 100.0 - prec5.item(), batch_size)

        print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
              .format(top1=meters['top1'], top5=meters['top5']))

    return meters['top1'].avg, meters['top5'].avg

##### ------- training with mixup + vat ------- #####
def train_mixup_vat(train_loader, model, optimizer, epoch, vat, args):
    model.train()

    if args.progress == True:
        from tqdm import tqdm
        tk0 = tqdm(train_loader, desc="Training Progress " +
                   str(epoch+1) + "/" + str(args.epochs), unit="batch")

    for i, (aug_images, target) in enumerate(tk0):
        target = target.to(args.device)
        # Create the mix
        alpha = args.alpha
        index = torch.randperm(args.batch_size, device=args.device)
        lam = np.random.beta(alpha, alpha)
        target_a, target_b = target, target[index]

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)              

        # Loop over the batches
        loss_sum_mixup = 0
        loss_sum_vat = 0
        count = 0
        for batch in aug_images:
            batch = batch.to(args.device)
            # mixup part
            m_batch = mixup_data(batch, index, lam)            
            class_logit = model(m_batch)
            if count == 0:
                loss_sum_mixup = mixup_criterion(
                    class_logit.double(), target_a, target_b, lam).mean()
            else:
                loss_sum_mixup += mixup_criterion(class_logit.double(),
                                            target_a, target_b, lam).mean()
            count += 1

            # vat part
            output = F.log_softmax(model(batch), dim=1)
            likelihood_loss = F.nll_loss(output, target)
            vat_loss = vat.forward(model, batch)
            loss_sum_vat += likelihood_loss + args.vat_lambda * vat_loss

        loss = (loss_sum_mixup * 0.5) + (loss_sum_vat * 0.5) / (args.aug_num)
        loss.backward()
        optimizer.step()


def validate_mixup_vat(eval_loader, model, args):
    meters = AverageMeterSet()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)            
            outputs_mixup = model(inputs)
            outputs_vat = F.log_softmax(outputs_mixup, dim=1)
            w_outputs_mixup = outputs_mixup *0.25
            w_outputs_vat = outputs_vat * 0.75
            outputs = torch.add(w_outputs_mixup, w_outputs_vat)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            meters.update('top1', prec1.item(), batch_size)
            meters.update('error1', 100.0 - prec1.item(), batch_size)
            meters.update('top5', prec5.item(), batch_size)
            meters.update('error5', 100.0 - prec5.item(), batch_size)

        print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
              .format(top1=meters['top1'], top5=meters['top5']))

    return meters['top1'].avg, meters['top5'].avg

####################### End of Training / Validation algorithms #######################


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    # assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):

    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        # assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def extract_features_simp(train_loader, model, args):
    model.eval()
    embeddings_all = []

    with torch.no_grad():
        for i, (batch_input) in enumerate(train_loader):
            X_n = batch_input[0].to(args.device)
            _, feats = model(X_n)
            embeddings_all.append(feats.data.cpu())
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    return embeddings_all


def load_args(args):
    args.workers = 4 * torch.cuda.device_count()
    label_dir = 'data-local/'

    if int(args.label_split) < 10:
        args.label_split = args.label_split.zfill(2)

    args.test_batch_size = args.batch_size
    args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (
        label_dir, args.dataset, args.num_labeled, args.label_split)
    
    # if args.dataset == "star":
    #     args.test_batch_size = args.batch_size
    #     args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (
    #         label_dir, args.dataset, args.num_labeled, args.label_split)

    # elif args.dataset == "r304" or args.dataset == "area3" or args.dataset == "area5":
    #     args.test_batch_size = args.batch_size
    #     args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (
    #         label_dir, args.dataset, args.num_labeled, args.label_split)
    
    # elif args.dataset == "gakken_demo" or args.dataset == "sands8":
    #     args.test_batch_size = args.batch_size
    #     args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (
    #         label_dir, args.dataset, args.num_labeled, args.label_split)

    # else:
    #     sys.exit('Undefined dataset!')

    return args
