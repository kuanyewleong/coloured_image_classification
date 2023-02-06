import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import cli, datasets
import helpers
import lp.db_eval as db_eval

from models import resnet10

matplotlib.use('Agg')


def plot_confusion_matrix(cm, classes, result_path, correct, total, acc, args):
    # Unnormalized
    u_cm = cm
    # print('Confusion matrix, without normalization')
    # print(cm)
    # Normalized confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # print(cm)

    plt.figure(figsize=(12, 9))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGnBu)
    accuracy = '\nAccuracy: {}/{} ({:.5f})\n'.format(correct, total, acc)

    if args.experiment == "initweights":
        title = 'Initialize conv layer with pretrained EfficientNet on TheStar Area5 (normalized, small samples)' + \
            accuracy
    elif args.experiment == "finetunefc":
        title = 'Fine-tuned fc layer EfficientNet on TheStar Area5 (normalized, small samples)' + \
            accuracy
    elif args.experiment == "rndweights":
        title = 'Combination of 4 chipsets* (EfficientNet, normalized, 81,200 balanced samples)' + accuracy

    plt.title(title, fontsize=16, color="black")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, horizontalalignment="right", rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        plt.axhline(tick_marks[i]+0.5, color='gray', linewidth=0.5)
        plt.axvline(tick_marks[i]+0.5, color='gray', linewidth=0.5)

    thresh = cm.max() / 2.
    float_formatter = "{:.1f}".format
    int_formatter = "{:d}".format
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            normalized_value = float_formatter(cm[i, j]*100)
            unnormalized_value = int_formatter(u_cm[i, j])
            plt.text(j, i, '{}%\n {}'.format(normalized_value, unnormalized_value),
                     verticalalignment="center", horizontalalignment="center",fontsize=5, color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ax = plt.gca()
    ax.set_xlabel('Predicted Class', fontsize=12, color="cornflowerblue")
    ax.xaxis.set_label_coords(1.07, -0.1)
    plt.ylabel('Actual Class', fontsize=12, color="cornflowerblue")
    # plt.xlabel('Predicted class', fontsize=14)
    if args.experiment == "initweights":
        savepath = result_path + "/efficientnet_initweights_area5_small.png"
        plt.savefig(savepath)
    elif args.experiment == "finetunefc":
        savepath = result_path + "/efficientnet_finetunefc_area5_small.png"
        plt.savefig(savepath)
    elif args.experiment == "rndweights":
        savepath = result_path + "/efficientnet_rndweights_all_4datasets.png"
        plt.savefig(savepath)

# --------------


def create_model(num_classes, args):
    model = resnet10(num_classes)
    args.device = torch.device('cuda')
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(args.device)
    model.load_state_dict(torch.load(
        args.weights_path, map_location=device_string))
    cudnn.benchmark = True
    return model


def validate(eval_loader, model, args):
    predictions = []
    correct = 0
    y_true = []
    y_pred = []
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # mixup
            if args.train_algo == "mixup":
                outputs = model(inputs)
            # vat
            elif args.train_algo == "vat":
                outputs = F.log_softmax(model(inputs), dim=1)

            # measure accuracy
            pred = outputs.max(1, keepdim=True)[1]
            pred_eq_target = pred.eq(
                targets.view_as(pred)).squeeze(1).cpu().numpy()
            predictions += list(pred.squeeze(1).cpu().numpy())
            correct += pred_eq_target.sum()
            total += batch_size

            # print("pred, target: {}, {}".format(pred[0].item(), targets[0].item()))
            for i in range(batch_size):
                y_true.append(targets[i].item())
                y_pred.append(pred[i].item())

        cm = confusion_matrix(y_true, y_pred)        
        print(cm)
    return cm, correct, total, correct/total


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # print(pred.shape)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval(weak_transformation, strong_transformation,
         eval_transformation,
         datadir,
         args):
    evaldir = os.path.join(datadir, args.eval_subdir)
    eval_dataset = db_eval.DBE(evaldir, False, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False)

    return eval_loader


def eval_main():
    # Get the command line arguments
    args = cli.parse_commandline_args()
    args = helpers.load_args(args)
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')    
    args.device = torch.device('cuda')
    # model = create_model(num_classes,args)

    ####### Resnet50
    # model = models.resnet50()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)

    ###### EfficientNet
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                           'nvidia_efficientnet_b0', pretrained=False)
    num_ftrs = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(num_ftrs, num_classes)

    model.to(args.device)
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(
        args.weights_path, map_location=device_string))
    model.eval()

    eval_loader = eval(**dataset_config, args=args)

    # Plot confusion matrix
    classes_names = ['AngelBG8color_ROUND-DEFAULT-YELLOW', \
        'AngelBG8color_ROUND-DEFAULT-GREEN', \
        'AngelBG8color_ROUND-DEFAULT-PURPLE', \
        'AngelBG8color_ROUND-DEFAULT-LIGHT_BLUE', \
        'AngelBG8color_ROUND-DEFAULT-ORANGE', \
        'AngelBG8color_ROUND-DEFAULT-RED', \
        'AngelBG8color_ROUND-DEFAULT-BLACK', \
        'AngelBG8color_ROUND-DEFAULT-BLUE', \
        'PoC2019Scc_ROUND-DEFAULT-RED', \
        'PoC2019Scc_ROUND-DEFAULT-ORANGE', \
        'PoC2019Scc_ROUND-DEFAULT-BLUE', \
        'PoC2019Scc_ROUND-DEFAULT-BLACK', \
        'PoC2019Scc_ROUND-DEFAULT-GREEN', \
        'PoC2019Scc_ROUND-DEFAULT-YELLOW', \
        'PoC2019Scc_ROUND-DEFAULT-PINK', \
        'PoC2019Scc_ROUND-DEFAULT-BROWN', \
        'g2e_nonAEC_ROUND-DEFAULT-YELLOW', \
        'g2e_nonAEC_ROUND-DEFAULT-BLACK', \
        'g2e_nonAEC_ROUND-DEFAULT-BLUE', \
        'g2e_nonAEC_ROUND-DEFAULT-GREEN', \
        'g2e_nonAEC_ROUND-DEFAULT-RED', \
        'sands8_ROUND-DEFAULT-ORANGE', \
        'sands8_ROUND-DEFAULT-BLUE', \
        'sands8_ROUND-DEFAULT-RED', \
        'sands8_ROUND-DEFAULT-YELLOW', \
        'sands8_ROUND-DEFAULT-BLACK', \
        'sands8_ROUND-DEFAULT-LIGHT_BLUE', \
        'sands8_ROUND-DEFAULT-VIOLET']

    # classes_names = ['MBSPoC2022_ROUND-DEFAULT-RED', \
    #     'MBSPoC2022_ROUND-NN-GREEN', \
    #     'MBSPoC2022_ROUND-DEFAULT-LIGHT_BLUE', \
    #     'MBSPoC2022_ROUND-NN-BLACK', \
    #     'MBSPoC2022_ROUND-DEFAULT-ORANGE', \
    #     'MBSPoC2022_ROUND-NN-VIOLET', \
    #     'MBSPoC2022_ROUND-NN-YELLOW', \
    #     'MBSPoC2022_ROUND-DEFAULT-BLACK', \
    #     'MBSPoC2022_ROUND-DEFAULT-GREEN', \
    #     'MBSPoC2022_ROUND-NN-ORANGE', \
    #     'MBSPoC2022_ROUND-NN-PINK', \
    #     'MBSPoC2022_ROUND-DEFAULT-YELLOW', \
    #     'MBSPoC2022_ROUND-DEFAULT-VIOLET', \
    #     'MBSPoC2022_ROUND-DEFAULT-NONAEC_GRAY', \
    #     'MBSPoC2022_ROUND-NN-LIGHT_BLUE', \
    #     'MBSPoC2022_ROUND-NN-BLUE', \
    #     'MBSPoC2022_ROUND-DEFAULT-PINK', \
    #     'MBSPoC2022_ROUND-DEFAULT-NONAEC_LIGHT_GREEN', \
    #     'MBSPoC2022_ROUND-DEFAULT-BLUE', \
    #     'MBSPoC2022_ROUND-DEFAULT-NONAEC_BROWN']

    # classes_names = ['AngelBG8color_ROUND-DEFAULT-YELLOW', \
    #     'AngelBG8color_ROUND-DEFAULT-GREEN', \
    #     'AngelBG8color_ROUND-DEFAULT-PURPLE', \
    #     'AngelBG8color_ROUND-DEFAULT-LIGHT_BLUE', \
    #     'AngelBG8color_ROUND-DEFAULT-ORANGE', \
    #     'AngelBG8color_ROUND-DEFAULT-RED', \
    #     'AngelBG8color_ROUND-DEFAULT-BLACK', \
    #     'AngelBG8color_ROUND-DEFAULT-BLUE']

    # classes_names = ['ROUND-DEFAULT-PINK', 'ROUND-DEFAULT-BLUE', 'ROUND-DEFAULT-VIOLET',
    #                  'ROUND-DEFAULT-RED', 'ROUND-DEFAULT-LIGHT_BLUE', 'ROUND-DEFAULT-GREEN',
    #                  'ROUND-DEFAULT-BLACK', ' ROUND-DEFAULT-ORANGE', 'ROUND-DEFAULT-YELLOW',
    #                  'ROUND-DEFAULT-DARK_BLUE', 'ROUND-NN-RED', 'ROUND-NN-GREEN',
    #                  'ROUND-NN-BLACK', 'ROUND-NN-ORANGE', 'ROUND-NN-YELLOW']
    
    cm, correct, total, acc = validate(eval_loader, model, args)
    plot_confusion_matrix(cm, classes_names, args.logdir,
                          correct, total, acc, args)


if __name__ == '__main__':
    eval_main()
