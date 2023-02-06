import os

import torchvision.transforms as transforms

from .augmentations import RandAugment
from .utils import export

  
@export
def sands8():
    channel_stats = dict(mean=[0.3277, 0.2849, 0.2611],
                         std=[0.1397, 0.1152, 0.1052])

    weak_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20, 200), padding=4, padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    strong_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20, 200), padding=4, padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformation = transforms.Compose([
        transforms.Resize((20, 200)),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]
    data_dir = '/../../media/16TBHDD/leong/dataset/cr_mixdataset_from_diff_projects'

    print("Using sands8 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 7
    }

@export
def all_5datasets():
    channel_stats = dict(mean=[0.2781, 0.2479, 0.2317],
                         std=[0.1668, 0.1408, 0.1268])

    weak_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((20, 200)),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    strong_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((20, 200)),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformation = transforms.Compose([
        transforms.Resize((20, 200)),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]
    data_dir = '/../../media/16TBHDD/leong/dataset/cr_mixdataset_from_diff_projects'

    print("Using 5 datasets from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 48
    }