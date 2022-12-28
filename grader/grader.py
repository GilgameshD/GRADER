'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2022-12-27 22:21:02
Description: 
'''

import numpy as np
from grader.planner import Planner
from grader.discover import Discover


class GRADER(object):
    name = 'GRADER'

    def __init__(self, args):
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        args['planner']['env_params'] = args['env_params']
        args['planner']['grader_model'] = args['grader_model']
        args['discover']['env_params'] = args['env_params']

        # use discovered graph or not. use gt if not use discovered graph
        self.use_discover = True
        
        # only use causal when we use causal model
        if args['grader_model'] != 'causal':
            self.use_discover = False

        args['planner']['use_discover'] = self.use_discover
        args['planner']['use_gt'] = not self.use_discover

        # two modules
        self.planner = Planner(args['planner'])
        self.discover = Discover(args['discover'])

        # decide the ratio between generation and discovery (generation is always longer)
        self.stage = 'generation'
        self.episode_counter = 0
        self.discovery_interval = args['discover']['discovery_interval']

    def stage_scheduler(self):
        if (self.episode_counter + 1) % self.discovery_interval == 0:
            self.stage = 'discovery'
        else:
            self.stage = 'generation'
        self.episode_counter += 1

    def select_action(self, env, state, deterministic):
        return self.planner.select_action(env, state, deterministic)

    def store_transition(self, data):
        self.planner.store_transition(data)
        self.discover.store_transition(data)

    def train(self):
        # discovery
        if self.stage == 'discovery' and self.use_discover:
            self.discover.update_causal_graph()
            self.planner.set_causal_graph(self.discover.get_adj_matrix_graph())

        # generation
        self.planner.train()

        # in the end, update the stage
        self.stage_scheduler()

    def save_model(self):
        self.planner.save_model(self.model_path, self.model_id)
        self.discover.save_model(self.model_path, self.model_id)

    def load_model(self):
        self.planner.load_model(self.model_path, self.model_id)
        self.discover.load_model(self.model_path, self.model_id)
