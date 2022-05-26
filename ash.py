import os

import numpy as np
import torch
import torch.nn.functional as F


def get_msp_score(logits):
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_energy_score(logits):
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_score(logits, method):
    if method == "msp":
        return get_msp_score(logits)
    if method == "energy":
        return get_energy_score(logits)
    exit('Unsupported scoring method')


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
    x = x * (1 - mask.float())  # x[mask] = 0.0

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
    x = x * (1 - mask.float())  # x[mask] = 0.0
    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_rand(x, percentile=65, a=0, b=10):
    assert x.dim() == 4

    # calculate pruning threshold per sample
    threshold = np.percentile(x.detach().cpu().numpy(), percentile, axis=[1, 2, 3])

    # convert threshold from numpy array to pytorch tensor
    device = get_tensor_device(x)
    threshold = torch.from_numpy(threshold).to(device)

    # set every input unit less then threshold to 0.0
    mask = torch.lt(x, threshold[:, None, None, None])
    x = x * (1 - mask.float())  # x[mask] = 0.0

    # apply sharpening
    alive = torch.logical_not(mask).float()
    c = (b - a) * torch.rand_like(x, device=device) + a
    x = alive * c[:, None, None, None]
    return x


def apply_ash(x):
    method = os.getenv('ash_method')
    [fn, p] = method.split('@')
    return eval(fn)(x, int(p))
