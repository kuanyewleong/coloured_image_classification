import numpy as np
import torch


def to_tensor(img):
    '''
    convert
        (H, W, C) np.ndarray to (C, H, W) torch.Tensor
        (N, H, W, C) np.ndarray to (N, C, H, W) torch.Tensor
    where
        H: Height
        W: Width
        C: Channel
        N: Batch size
    '''
    if len(img.shape) == 3:
        trans_param = (2, 0, 1)
    elif len(img.shape) == 4:
        trans_param = (0, 3, 1, 2)

    if img.dtype in (np.uint16, np.uint8):
        img = img.astype(np.float32) / np.iinfo(img.dtype).max

    img_tensor = torch.from_numpy(img.transpose(*trans_param))

    return img_tensor
