import torch
import torch.nn.functional as F
from torch import nn


class LogSobel(nn.Module):  # pylint: disable=abstract-method

    def __init__(self, padding_mode, clamp_min=1/255):
        super().__init__()

        if padding_mode == 'replication':
            # aaa | abcde | eee
            self.padding = nn.ReplicationPad2d(1)
        elif padding_mode == 'reflection':
            # dcb | abcde | dcb
            self.padding = nn.ReflectionPad2d(1)
        else:
            raise NotImplementedError(padding_mode)

        self.clamp_min = clamp_min
        dx_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        dy_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        zero_kernel = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        weight = torch.Tensor([
            [dx_kernel, zero_kernel, zero_kernel],
            [zero_kernel, dx_kernel, zero_kernel],
            [zero_kernel, zero_kernel, dx_kernel],
            [dy_kernel, zero_kernel, zero_kernel],
            [zero_kernel, dy_kernel, zero_kernel],
            [zero_kernel, zero_kernel, dy_kernel],
        ])
        self.weight = nn.Parameter(weight, requires_grad=False)

    def __call__(self, x):  # pylint: disable=arguments-differ
        x = torch.clamp(x, min=self.clamp_min)
        x = torch.log(x)
        x = self.padding(x)
        x = F.conv2d(x, weight=self.weight, padding=0)
        return x
