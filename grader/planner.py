'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2022-12-28 13:31:19
Description: 
'''

from copy import deepcopy
import numpy as np
import os

import torch
import torch.nn as nn
import networkx as nx

from grader.mpc.mpc import MPC_Chemistry
from grader.gnn_modules import RGCN, MLP, GRU_SCM
from grader.grader_utils import CUDA, kaiming_init


class WorldModel(object):
    def __init__(self, args):
        self.state_dim = args['env_params']['state_dim']
        self.action_dim = args['env_params']['action_dim']
        self.goal_dim = args['env_params']['goal_dim']
        self.env_name = args['env_params']['env_name']
        self.grader_model = args['grader_model']
        self.use_discover = args['use_discover']
        self.use_gt = args['use_gt']

        assert self.grader_model in ['causal', 'full', 'mlp', 'offline', 'gnn']

        self.n_epochs = args['n_epochs']
        self.lr = args['lr']
        self.batch_size = args['batch_size']

        self.validation_flag = args['validation_flag']
        self.validate_freq = args['validation_freq']
        self.validation_ratio = args['validation_ratio']
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # process things that are different in environments
        if self.env_name == 'chemistry':
            self.build_node_and_edge = self.build_node_and_edge_chemistry
            self.organize_nodes = self.organize_nodes_chemistry
            self.num_objects = args['env_params']['num_objects']
            self.num_colors = args['env_params']['num_colors']
            self.width = args['env_params']['width']
            self.height = args['env_params']['height']
            self.adjacency_matrix = args['env_params']['adjacency_matrix']
            self.adjacency_matrix += np.eye(self.adjacency_matrix.shape[0]) # add diagonal elements
            self.state_dim_list = [self.num_colors * self.width * self.height] * self.num_objects 
            self.action_dim_list = [self.num_objects * self.num_colors] # action does not have causal variables
        else:
            raise ValueError('Unknown environment name')

        self.use_full = False
        self.use_mlp = False
        if self.grader_model == 'mlp':
            self.model_name = 'mlp'
            self.use_mlp = True
        elif self.grader_model == 'causal':
            self.model_name = 'gru'
        elif self.grader_model == 'full':
            self.model_name = 'gru'
            self.use_full = True
        elif self.grader_model == 'gnn':
            self.model_name = 'gnn'
            self.use_full = True

        random = False
        if self.model_name == 'mlp':
            input_dim = self.state_dim - self.goal_dim + self.action_dim
            output_dim = self.state_dim - self.goal_dim
            self.model = CUDA(MLP(input_dim, output_dim, args["hidden_dim"], args["hidden_size"], dropout_p=0.0))
            hidden_dim = args["hidden_size"]
        elif self.model_name == 'gru' or self.model_name == 'gnn':
            edge_dim = 1
            hidden_num = 1
            if self.env_name == 'chemistry':
                args["hidden_dim"] = 64
                
            hidden_dim = args["hidden_dim"]
            self.node_num = len(self.action_dim_list) + len(self.state_dim_list)
            self.node_dim = int(np.max(self.state_dim_list+self.action_dim_list))
            if self.model_name == 'gnn':
                self.model = CUDA(RGCN(self.node_dim, self.node_num, 'mean', args["hidden_dim"], self.node_dim, edge_dim, hidden_num))
            else:
                self.model = CUDA(GRU_SCM(self.action_dim_list, self.state_dim_list, self.node_num, 'mean', args["hidden_dim"], edge_dim, hidden_num, dropout=0.0, random=random))

        print('----------------------------')
        print('Env:', self.env_name)
        print('GRADER model:', self.grader_model)
        print('Model_name:', self.model_name)
        print('Full:', self.use_full)
        print('SCM noise:', random)
        print('Hidden dim:', hidden_dim)
        print('----------------------------')

        self.model.apply(kaiming_init)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.buffer_length = 0
        self.criterion = self.mse_loss

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data = None
        self.label = None
        self.eps = 1e-30

        if self.grader_model == 'causal':
            # the initial graph is a lower triangular graph
            self.causal_graph = np.zeros((self.node_num, self.node_num))
            for i in range(self.causal_graph.shape[0]):
                for j in range(self.causal_graph.shape[1]):
                    if i >= j:
                        self.causal_graph[i, j] = 1
        self.best_test_loss = np.inf

    def build_node_and_edge_chemistry(self, data):
        # create the node matrix. the last node is the output node therefore should always be 0.
        batch_size = data.shape[0]
        x = torch.zeros((batch_size, self.node_num, self.node_dim), device=torch.device(self.device)) # [B, 125]

        # build the nodes of action
        action = data[:, sum(self.state_dim_list):]
        start_ = 0
        for a_i in range(len(self.action_dim_list)):
            end_ = self.action_dim_list[a_i] + start_
            x[:, a_i, 0:end_-start_] = action[:, start_:end_] # pad 0 for remaining places
            start_ = end_

        # build the nodes of state
        state = data[:, 0:sum(self.state_dim_list)]

        # [B, N*C*W*H] -> [B, N, C*W*H]
        state = state.reshape(batch_size, self.num_objects * self.num_colors, self.width, self.height)
        state = state.reshape(batch_size, self.num_objects, self.num_colors * self.width * self.height)
        start_ = 0
        for s_i in range(len(self.state_dim_list)):
            end_ = self.state_dim_list[s_i] + start_
            x[:, s_i+len(self.action_dim_list), 0:end_-start_] = state[:, s_i, :] # pad 0 for remaining places
            start_ = end_

        if self.use_full:
            # full graph (states are fully connected)
            full = np.ones((self.node_num, self.node_num))
            action_row = np.zeros((1, self.node_num))
            action_row[0] = 1
            full[0, :] = action_row
            adj = full

        if self.use_discover:
            adj = self.causal_graph

        if self.use_gt:
            # using GT causal graph
            gt_adj = np.zeros((self.node_num, self.node_num))
            gt_adj[1:, 1:] = self.adjacency_matrix
            gt_adj[:, 0] = 1.0
            adj = gt_adj

        adj = np.array(adj)[None, None, :, :].repeat(batch_size, axis=0)
        adj = CUDA(torch.from_numpy(adj.astype(np.float32)))
        return x, adj

    def organize_nodes_chemistry(self, x):
        # x - [B, node_num, node_dim], the nodes of next_state are in the end
        delta_state_node = x[:, -len(self.state_dim_list):, :]
        delta_state = []
        for s_i in range(len(self.state_dim_list)):
            state_i = delta_state_node[:, s_i:s_i+1, 0:self.state_dim_list[s_i]] 
            delta_state.append(state_i)

        # NOTE: since the embedding of state has beed reordered, we should do that thing again
        delta_state = torch.cat(delta_state, dim=1) # [B, N, C*W*H]
        delta_state = delta_state.reshape(delta_state.shape[0], self.num_objects * self.num_colors * self.width * self.height)
        return delta_state

    def data_process(self, data, max_buffer_size):
        x = data[0][None]
        label = data[1][None]
        self.buffer_length += 1
    
        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.from_numpy(x.astype(np.float32)))
            self.label = CUDA(torch.from_numpy(label.astype(np.float32)))
        else:
            if self.data.shape[0] < max_buffer_size:
                self.data = torch.cat((self.data, CUDA(torch.from_numpy(x.astype(np.float32)))), dim=0)
                self.label = torch.cat((self.label, CUDA(torch.from_numpy(label.astype(np.float32)))), dim=0)
            else:
                # replace the old buffer
                #index = self.buffer_length % max_buffer_size # sequentially replace buffer
                index = np.random.randint(0, max_buffer_size) # randomly replace buffer
                self.data[index] = CUDA(torch.from_numpy(x.astype(np.float32)))
                self.label[index] = CUDA(torch.from_numpy(label.astype(np.float32)))

    def split_train_validation(self):
        num_data = len(self.data)

        # use validation
        if self.validation_flag:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]

            train_set = [[self.data[idx], self.label[idx]] for idx in train_idx]
            test_set = [[self.data[idx], self.label[idx]] for idx in test_idx]

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_set = [[self.data[idx], self.label[idx]] for idx in range(num_data)]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self):
        self.model.train()
        train_loader, test_loader = self.split_train_validation()

        self.best_test_loss = np.inf
        for epoch in range(self.n_epochs):
            for datas, labels in train_loader:
                self.optimizer.zero_grad()

                if self.use_mlp:
                    delta = self.model(datas)
                    loss = self.criterion(delta, labels)
                else:
                    x, adj = self.build_node_and_edge(datas)
                    x = self.model(x, adj)
                    delta = self.organize_nodes(x)
                    loss = self.criterion(delta, labels)
                loss.backward()
                self.optimizer.step()

            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                with torch.no_grad():
                    loss_test = self.validate_model(test_loader)
                if loss_test < self.best_test_loss:
                    self.best_test_loss = loss_test
                    self.best_model = deepcopy(self.model.state_dict())

        # load the best model if we use validation
        if self.validation_flag:
            self.model.load_state_dict(self.best_model)
        return self.best_test_loss

    def validate_model(self, testloader):
        self.model.eval()
        loss_list = []
        for datas, labels in testloader:
            if self.use_mlp:
                delta = self.model(datas)
                loss = self.criterion(delta, labels)
            else:
                x, adj = self.build_node_and_edge(datas)
                x = self.model(x, adj)
                delta = self.organize_nodes(x)
                loss = self.criterion(delta, labels)

            loss_list.append(loss.item())
        self.model.train()
        return np.mean(loss_list)

    def predict(self, s, a):
        self.model.eval()
        # convert to torch format
        if isinstance(s, np.ndarray):
            s = CUDA(torch.from_numpy(s.astype(np.float32)))
        if isinstance(a, np.ndarray):
            a = CUDA(torch.from_numpy(a.astype(np.float32)))

        inputs = torch.cat((s, a), axis=1)

        with torch.no_grad():
            if self.use_mlp:
                delta = self.model(inputs)
            else:
                x, adj = self.build_node_and_edge(inputs)
                x = self.model(x, adj)
                delta = self.organize_nodes(x)

            delta = delta.cpu().detach().numpy()
        return delta

    def save_model(self, model_path, model_id):
        states = {'model': self.model.state_dict()}
        filepath = os.path.join(model_path, 'grade.'+str(model_id)+'.torch')
        with open(filepath, 'wb') as f:
            torch.save(states, f)

    def load_model(self, model_path, model_id):
        filepath = os.path.join(model_path, 'grade.'+str(model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise Exception('No GRADE model found!')


class Planner(object):
    def __init__(self, args):
        self.pretrain_buffer_size = args['pretrain_buffer_size']
        self.max_buffer_size = args['max_buffer_size']
        self.epsilon = args['epsilon']
        self.goal_dim = args['env_params']['goal_dim']

        args['mpc']['env_params'] = args['env_params']
        self.mpc_controller = MPC_Chemistry(args['mpc'])
        self.mpc_controller.reset()

        self.model = WorldModel(args)

    def select_action(self, env, state, deterministic):
        if self.model.data is None or self.model.data.shape[0] < self.pretrain_buffer_size:
            action = env.random_action()
        else:
            if np.random.uniform(0, 1) > self.epsilon or deterministic:
                action = self.mpc_controller.act(model=self.model, state=state)
            else:
                action = env.random_action()
        return action

    def store_transition(self, data):
        # [state, action, next_state]
        # we should remove the goal infomation from x and label
        pure_state = data[0][:len(data[0])-self.goal_dim]
        action = data[1]
        pure_next_state = data[2][:len(data[0])-self.goal_dim]
        x = np.concatenate([pure_state, action])
        label = pure_next_state - pure_state 
        self.model.data_process([x, label], self.max_buffer_size)

    def train(self):
        # when data has been collected enough, train model
        if self.model.data.shape[0] < self.pretrain_buffer_size:
            self.best_test_loss = 0
        else:
            self.best_test_loss = self.model.fit()

    def set_causal_graph(self, causal_graph):
        self.model.causal_graph = causal_graph

    def save_model(self, model_path, model_id):
        self.model.save_model(model_path, model_id)

    def load_model(self, model_path, model_id):
        self.model.load_model(model_path, model_id)
