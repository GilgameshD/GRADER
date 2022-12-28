'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-05-30 14:20:04
LastEditTime: 2022-12-28 12:41:41
@Description: 
'''

import numpy as np
import math
import torch.nn.init as init
import torch
import torch.nn as nn


def CPU(var):
    return var.detach().cpu().numpy()


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def xavier_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=math.sqrt(3))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
