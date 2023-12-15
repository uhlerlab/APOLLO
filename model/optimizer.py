import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np


def optimizer_kl(mu, logvar, nodemask=None,reduction='mean'):
    if reduction=='mean':
        f=torch.mean
        if nodemask is None:
            s=mu.size()[0]
        else:
            s=nodemask.size()[0]
    elif reduction=='sum':
        f=torch.sum
        s=1
    if nodemask is None:
        kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return kl
    kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar[nodemask] - mu[nodemask].pow(2) - logvar[nodemask].exp().pow(2), 1))
    return kl