from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal
import time


device = torch.device('cuda:{}'.format(2))

def loss_warmup(y_1,t, forget_rate, ind,noise_or_not):
    outputs = F.softmax(y_1, dim=1)
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]


    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()

        num_remember = ind_1_update.shape[0]

    loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    pure_ratio_1 = np.sum(noise_or_not[ind]) / ind.shape[0]

    return torch.sum(loss_1_update) / num_remember,pure_ratio_1

# Loss functions





def loss_RDIA(y_1, t, forget_rate, ind, noise_or_not,epoch,noise_rate,gamma):

    outputs = F.softmax(y_1, dim=1)
    _, pred1 = torch.max(y_1.data, 1)
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]


    remember_rate = 1 - forget_rate
    #noisy rate should be 0.05 for 20% noisy rate
    #noisy rate should be 0.2 for 50% noisy rate
    negative_rate = noise_rate/2
    num_remember = int(remember_rate * len(loss_1_sorted))
    negative_number  = int(negative_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_1_negative=ind_1_sorted[-negative_number:].cpu()
    ind_1_unsure = ind_1_sorted[num_remember:-negative_number].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]])/float(num_remember)


    #cross_entropy
    loss_1_update = F.cross_entropy(outputs[ind_1_update], t[ind_1_update])
    #RDIA

    s1_neg = torch.log(torch.clamp(1.-outputs, min=1e-7, max=1.))
    loss_1_negative = F.nll_loss(s1_neg[ind_1_unsure],t[ind_1_unsure])
    return gamma*torch.sum(loss_1_update)+(1-gamma)*torch.sum(loss_1_negative)/len(loss_1_sorted)-negative_number, pure_ratio_1
