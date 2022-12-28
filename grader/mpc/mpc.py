'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-04-16 21:29:40
LastEditTime: 2022-12-27 20:16:32
@Description: 
'''

import numpy as np
from grader.mpc.optimizers import RandomOptimizer
import copy


class MPC_Chemistry(object):
    def __init__(self, mpc_args):
        self.type = mpc_args['type']
        self.horizon = mpc_args['horizon']
        self.gamma = mpc_args['gamma']
        self.popsize = mpc_args['popsize']

        # parameters from the environment
        self.action_dim = mpc_args['env_params']['action_dim']
        self.goal_dim = mpc_args['env_params']['goal_dim']

        self.optimizer = RandomOptimizer(action_dim=self.action_dim, horizon=self.horizon, popsize=self.popsize)
        self.optimizer.setup(self.cost_function)
        self.reset()

    def reset(self):
        self.optimizer.reset()

    def act(self, model, state):
        # process the state to get pure state and goal
        goal = state[len(state)-self.goal_dim:]
        pure_state = state[:len(state)-self.goal_dim] # remove the goal info at very beginning

        self.model = model
        self.state = pure_state
        self.goal = state = np.repeat(goal[None], self.popsize, axis=0)

        best_solution = self.optimizer.obtain_solution_chemistry(self.action_dim)

        # task the first step as our action
        action = best_solution[0]
        return action

    def preprocess(self, state):
        state = np.repeat(self.state[None], self.popsize, axis=0)
        return state

    def cost_function(self, actions):
        # the observation need to be processed since we use a common model
        state = self.preprocess(self.state)
        stop_flag = np.ones(self.popsize,)

        assert actions.shape == (self.popsize, self.horizon, self.action_dim)
        costs = np.zeros(self.popsize)
        for t_i in range(self.horizon):
            action = actions[:, t_i, :]  # (batch_size, timestep, action dim)
            # the output of the prediction model is [state_next - state]
            state_next = self.model.predict(state, action) + state
            cost, stop_mask = self.chemistry_objective(state_next)  # compute cost
            stop_flag = stop_flag * stop_mask # Bit AND, stopped trajectory will have 0 cost
            costs += (1-stop_flag) * cost
            state = copy.deepcopy(state_next)

        return costs

    def chemistry_objective(self, state):
        mse = np.sum((state - self.goal) ** 2, axis=1) ** 0.5
        final_cost = mse
        stop_mask = mse < 0.1 # goal achieved
        return final_cost, stop_mask
