'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-06-09 23:39:36
LastEditTime: 2022-12-27 20:00:52
@Description: 
'''

import numpy as np
import torch
import time

from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, action_dim, horizon, popsize):
        super().__init__()
        self.horizon = horizon
        self.popsize = popsize
        self.action_dim = action_dim
        self.solution = None
        self.cost_function = None

    def setup(self, cost_function):
        self.cost_function = cost_function

    def reset(self):
        pass

    def obtain_solution_tower(self, *args, **kwargs):     
        #start = time.time()
        color_dim = 5
        shape_dim = 3
        # convert int to onehot
        color = np.random.randint(0, color_dim-1, size=(self.popsize, self.horizon))
        color = (np.arange(color_dim-1) == color[..., None]).astype(int)
        shape = np.random.randint(0, shape_dim-1, size=(self.popsize, self.horizon))
        shape = (np.arange(shape_dim-1) == shape[..., None]).astype(int)
        stop = np.random.randint(0, 2, size=(self.popsize, self.horizon, 1))

        action = np.concatenate([color, shape, stop], axis=2)

        #end_1 = time.time()

        costs = self.cost_function(action)
        solution = action[np.argmin(costs)]
        return solution

    def generate_one_action(self, low, high, size):
        shape = torch.Size(size)
        if torch.cuda.is_available():
            move = torch.cuda.LongTensor(shape)
        else:
            move = torch.LongTensor(shape)

        torch.randint(0, high, size=shape, out=move)
        move = torch.nn.functional.one_hot(move)
        return move

    def obtain_solution_chemistry(self, action_dim):     
        # convert int to onehot
        action = np.random.randint(0, action_dim, size=(self.popsize, self.horizon))
        action = (np.arange(action_dim) == action[..., None]).astype(int)
        costs = self.cost_function(action)
        solution = action[np.argmin(costs)]
        return solution