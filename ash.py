import torch
import numpy as np


def get_tensor_device(x):
    device = x.get_device()
    if device == -1:
        return 'cpu'
    return device


def ash_b(x, percentile=65):
    assert x.dim() == 4

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    # calculate pruning threshold per sample
    threshold = np.percentile(x.cpu().numpy(), percentile, axis=[1, 2, 3])

    # convert threshold from numpy array to pytorch tensor
    device = get_tensor_device(x)
    threshold = torch.from_numpy(threshold).to(device)

    # set every input unit less then threshold to 0.0
    mask = torch.lt(x, threshold[:, None, None, None])
    x[mask] = 0.0

    # apply sharpening
    alive = torch.logical_not(mask).float()
    scale = (s1 / alive.sum(dim=[1, 2, 3])).float()
    x = alive * scale[:, None, None, None]
    return x


def ash_p(x, percentile=65):
    assert x.dim() == 4

    # calculate pruning threshold per sample
    threshold = np.percentile(x.cpu().numpy(), percentile, axis=[1, 2, 3])

    # convert threshold from numpy array to pytorch tensor
    device = get_tensor_device(x)
    threshold = torch.from_numpy(threshold).to(device)

    # set every input unit less then threshold to 0.0
    mask = torch.lt(x, threshold[:, None, None, None])
    x[mask] = 0.0

    return x


def ash_s(x, percentile=65):
    assert x.dim() == 4

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    # calculate pruning threshold per sample
    threshold = np.percentile(x.cpu().numpy(), percentile, axis=[1, 2, 3])

    # convert threshold from numpy array to pytorch tensor
    device = get_tensor_device(x)
    threshold = torch.from_numpy(threshold).to(device)

    # set every input unit less then threshold to 0.0
    mask = torch.lt(x, threshold[:, None, None, None])
    # x[mask] = 0.0
    x = x * mask.float()
    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x
