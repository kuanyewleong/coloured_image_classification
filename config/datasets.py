import os

import torchvision.transforms as transforms

from .augmentations import RandAugment
from .utils import export


@export
def star():
    channel_stats = dict(mean=[0.0396, 0.0273, 0.0230],
                         std=[0.0917, 0.0671, 0.0535])

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
        # to solve the issue of tensor size mismatch
        transforms.Resize((20, 200)),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised'

    print("Using theStar from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }


@export
def r304():
    channel_stats = dict(mean=[0.0536, 0.0469, 0.0472],
                         std=[0.1033, 0.0853, 0.0898])

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
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised/Room304'

    print("Using 304 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }


@export
def area3():
    channel_stats = dict(mean=[0.0359, 0.0248, 0.0214],
                         std=[0.0648, 0.0414, 0.0293])

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
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised/Area3'

    print("Using Area3 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }


@export
def area5():
    channel_stats = dict(mean=[0.0161, 0.0142, 0.0146],
                         std=[0.0132, 0.0090, 0.0103])

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
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised/Area5'

    print("Using Area5 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }


@export
def AngelBG8color():
    channel_stats = dict(mean=[0.4406, 0.3471, 0.3143],
                         std=[0.1592, 0.1277, 0.1461])

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

    print("Using AngelBG8color from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 8
    }

    
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
def AngelBG8color_sands8():
    channel_stats = dict(mean=[0.3611, 0.3033, 0.2769],
                         std=[0.1546, 0.1224, 0.1213])

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

    print("Using AngelBG8color_sands8 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }

@export
def g2e_nonAEC():
    channel_stats = dict(mean=[0.1911, 0.1939, 0.1954],
                         std=[0.1407, 0.1244, 0.1004])

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

    print("Using g2e_nonAEC from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 5
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