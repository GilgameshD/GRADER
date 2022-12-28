'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2020-06-09 23:39:29
LastEditTime: 2021-12-22 18:47:19
Description: 
'''

class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")
