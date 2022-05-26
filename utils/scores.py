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
