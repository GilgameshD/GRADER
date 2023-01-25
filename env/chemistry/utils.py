"""Utility functions."""

import os
from collections import defaultdict

import numpy as np

import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

EPS = 1e-17


def get_cmap(cmap, mode):
    length = 9
    if cmap == 'Sets':
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Set1')
        else:
            cmap = [plt.get_cmap('Set1'), plt.get_cmap('Set3')]
            length = [9,12]
    else :
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Pastel1')
        else:
            cmap = [plt.get_cmap('Pastel1'), plt.get_cmap('Pastel2')]
            length = [9,8]

    return cmap, length

def observed_colors(num_colors, mode):
    if mode == 'ZeroShot':
        c = np.sort(np.random.uniform(0.0, 1.0, size=num_colors))
    else:
        c = (np.arange(num_colors)) / (num_colors-1)
        diff = 1.0 / (num_colors - 1)
        if mode == 'Train':
            diff = diff / 8.0
        elif mode == 'Test-v1':
            diff = diff / 4.0
        elif mode == 'Test-v2':
            diff = diff / 3.0
        elif mode == 'Test-v3':
            diff = diff / 2.0

        unif = np.random.uniform(-diff+EPS, diff-EPS, size=num_colors)
        unif[0] = abs(unif[0])
        unif[-1] = -abs(unif[-1])

        c = c + unif

    return c

def unobserved_colors(cmap, num_colors, mode, new_colors=None):
    if mode in ['Train', 'ZeroShotShape']:
        cm, length = get_cmap(cmap, mode)
        weights = np.sort(np.random.choice(length, num_colors, replace=False))
        colors = [cm(i/length) for i in weights]
    else:
        cm, length = get_cmap(cmap, mode)
        cm1, cm2 = cm
        length1, length2 = length
        l = length1 + len(new_colors)
        w = np.sort(np.random.choice(l, num_colors, replace=False))
        colors = []
        weights = []
        for i in w:
            if i < length1:
                colors.append(cm1(i/length1))
                weights.append(i)
            else:
                colors.append(cm2(new_colors[i - length1] / length2))
                weights.append(new_colors[i - length1] + 0.5)

    return colors, weights

def get_colors_and_weights(cmap='Set1', num_colors=9, observed=True, mode='Train', new_colors=None):
    """Get color array from matplotlib colormap."""
    if observed:
        c = observed_colors(num_colors, mode)
        cm = plt.get_cmap(cmap)

        colors = []
        for i in reversed(range(num_colors)):
            colors.append((cm(c[i])))

        weights = [num_colors - idx for idx in range(num_colors)]
    else:
        colors, weights = unobserved_colors(cmap, num_colors, mode, new_colors)
    return colors, weights


def pairwise_distance_matrix(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU(inplace=True)
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    #print(indices.size())
    #print(zeros.size())
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result
