'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2022-12-27 22:32:14
Description: 
'''

import os
import numpy as np
import pickle as pkl

import pandas as pd
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from fcit import fcit  # https://github.com/kjchalup/fcit, https://arxiv.org/pdf/1804.02747.pdf
import copy


class Discover(object):
    def __init__(self, args):
        self.goal_dim = args['env_params']['goal_dim']
        self.action_dim = args['env_params']['action_dim']
        self.env_name = args['env_params']['env_name']

        # parameters
        self.num_perm = 8
        self.prop_test = 0.5

        if self.env_name == 'chemistry':
            self.pvalue_threshold = 0.01
            self.num_objects = args['env_params']['num_objects']
            self.num_colors = args['env_params']['num_colors']
            self.width = args['env_params']['width']
            self.height = args['env_params']['height']
            self.adjacency_matrix = args['env_params']['adjacency_matrix']
            self.state_dim_list = [self.num_colors * self.width * self.height] * self.num_objects 
            self.action_dim_list = [self.num_objects * self.num_colors] # action does not have causal variables
            self.adj_node_num = len(self.action_dim_list) + len(self.state_dim_list) 
            self.state_dim_list = self.state_dim_list * 2 
            self.ground_truth = self.adjacency_matrix + np.eye(self.adjacency_matrix.shape[0]) # add diagonal elements

            # nodes (Action x1, state x20)
            if self.num_objects == 10:
                self.next_state_offset = 11
                self.node_name_mapping = {
                    0: 'A_i',  
                    1: 'S_0',  2: 'S_1', 3: 'S_2', 4: 'S_3', 5: 'S_4', 6: 'S_5', 7: 'S_6', 8: 'S_7', 9: 'S_8', 10: 'S_9',
                    11: 'NS_0', 12: 'NS_1', 13: 'NS_2', 14: 'NS_3', 15: 'NS_4', 16: 'NS_5', 17: 'NS_6', 18: 'NS_7', 19: 'NS_8', 20: 'NS_9'
                }
            elif self.num_objects == 5:
                self.next_state_offset = 6
                self.node_name_mapping = {
                    0: 'A_i',  
                    1: 'S_0',  2: 'S_1', 3: 'S_2', 4: 'S_3', 5: 'S_4',
                    6: 'NS_0', 7: 'NS_1', 8: 'NS_2', 9: 'NS_3', 10: 'NS_4'
                }

            # remove the dimension that has no influence
            self.remove_list = [[] for _ in self.node_name_mapping.keys()]
            # variable type is discrete or not
            self.discrete_var = {
                0: False, 
                1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False, 10: False, 
                11: False, 12: False, 13: False, 14: False, 15: False, 16: False, 17: False, 18: False, 19: False, 20: False
            }
        else:
            raise ValueError('unknown env name')

        # build the variable list
        self.node_dim_list = self.action_dim_list + self.state_dim_list 
        self.var_list = [self.node_name_mapping[n_i] for n_i in range(len(self.node_name_mapping.keys()))]
        self.intervene_var_list = self.var_list.copy()

        # build causal graph
        self.reset_causal_graph()

        # build dataset 
        self.dataset_dict = {i: [] for i in range(len(self.node_dim_list))}

    def reset_causal_graph(self):
        self.causal_graph = nx.DiGraph()
        # add nodes to causal graph
        for var_i in self.var_list:
            self.causal_graph.add_node(var_i) 

    def load_data(self, path, used_ratio):
        if used_ratio > 1:
            raise ValueError('used_ratio should be smaller than 1')

        self.dataset_dict = np.load(path, allow_pickle=True).item()
        data_size = len(self.dataset_dict[0])
        print('loaded data size:', data_size)
        used_size = int(used_ratio * data_size)
        for k_i in self.dataset_dict.keys():
            self.dataset_dict[k_i] = self.dataset_dict[k_i][0:used_size]

    def store_transition(self, data):
        # find the height of the delta state
        def find_current_state(state):
            state = state.reshape(self.max_height, self.color_dim+self.shape_dim)
            # get the current height
            height = self.max_height - 1
            for h_i in range(self.max_height):
                color_idx = np.argmax(state[h_i][0:self.color_dim])
                shape_idx = np.argmax(state[h_i][self.color_dim:])
                if color_idx == 0 and shape_idx == 0:
                    height = h_i - 1
                    break

            if height < 0:
                return state[0]
            else:
                return state[height]

        # [state, action, next_state]
        # we should remove the goal infomation from x and label
        state = data[0][:len(data[0])-self.goal_dim]
        action = data[1]
        next_state = data[2][:len(data[0])-self.goal_dim]
        delta_state = next_state - state

        if self.env_name == 'chemistry':
            # check whether the intervention is valid
            action_check = np.argmax(action)
            # only change color of one object one time
            obj_id = action_check // self.num_colors
            color_id = action_check % self.num_colors 
            state_check = state.reshape(self.num_objects, self.num_colors, self.width, self.height)
            state_check = state_check.sum(3)
            state_check = state_check.sum(2)
            if state_check[obj_id][color_id] == 1: # the intervention will not have influence
                return 

        #state_next_state = np.concatenate([state, next_state])
        state_next_state = np.concatenate([state, delta_state], axis=0)

        # build the nodes of action
        start_ = 0
        for a_i in range(len(self.action_dim_list)):
            end_ = self.action_dim_list[a_i] + start_
            node_a = action[start_:end_]
            self.dataset_dict[a_i].append(node_a)
            start_ = end_

        # build the nodes of state
        start_ = 0
        for s_i in range(len(self.state_dim_list)):
            end_ = self.state_dim_list[s_i] + start_
            node_s = state_next_state[start_:end_] 

            if self.env_name == 'chemistry':
                # remove position
                node_s = node_s.reshape(self.num_colors, self.width, self.height)
                node_s = np.sum(node_s, axis=2)
                node_s = np.sum(node_s, axis=1)

            self.dataset_dict[s_i+len(self.action_dim_list)].append(node_s)
            start_ = end_

    def _two_variable_test(self, i, j, cond_list):
        # get x variable
        x = copy.deepcopy(np.array(self.dataset_dict[i]))
        x = np.delete(x, self.remove_list[i], axis=1)
        name_x = self.node_name_mapping[i]

        # get y variable
        y = copy.deepcopy(np.array(self.dataset_dict[j]))
        y = np.delete(y, self.remove_list[j], axis=1)
        name_y = self.node_name_mapping[j]

        # independency test
        if len(cond_list) == 0:
            pvalue = fcit.test(x, y, z=None, num_perm=self.num_perm, prop_test=self.prop_test, discrete=(self.discrete_var[i], self.discrete_var[j]))
        # conditional independency test
        else:
            z = []
            for z_idx in cond_list:
                z_i = copy.deepcopy(np.array(self.dataset_dict[z_idx]))
                z_i = np.delete(z_i, self.remove_list[z_idx], axis=1)
                z.append(z_i)
            z = np.concatenate(z, axis=1)
            pvalue = fcit.test(x, y, z=z, num_perm=self.num_perm, prop_test=self.prop_test, discrete=(self.discrete_var[i], self.discrete_var[j]))

        name_z = ''
        for k in cond_list:
            name_z += self.node_name_mapping[k] 
            name_z += ' '

        #print(name_x, 'and', name_y, 'condition on [', name_z, '] , pvalue is {:.5f}'.format(pvalue))
        return pvalue

    def _two_variable_test_chisquare(self, i, j):
        name_x = self.node_name_mapping[i]
        name_y = self.node_name_mapping[j]
        contingency_table = pd.crosstab(self.dataframe[name_y], self.dataframe[name_x])

        # a table summarization of two categorical variables in this form is called a contingency table.
        _, pvalue, _, _ = stats.chi2_contingency(contingency_table)
        #print(name_x, 'and', name_y, 'pvalue is {:.5f}'.format(pvalue))
        return pvalue

    def _is_action(self, name):
        return True if name.split('_')[0] == 'A' else False

    def _is_state(self, name):
        return True if name.split('_')[0] == 'S' else False

    def _is_next(self, name):
        return True if name.split('_')[0] == 'NS' else False

    def select_action(env, state):
        ''' For interventional discovery, actively select action. For random discovery, randomly select actions '''
        action = env.random_action()
        return action

    def update_causal_graph(self):
        # convert the dataset dict to dataframe for discrete variables
        if self.env_name in ['chemistry']:
            data_dict = {}
            for n_i in self.dataset_dict.keys():
                x = copy.deepcopy(np.array(self.dataset_dict[n_i]))
                x = np.delete(x, self.remove_list[n_i], axis=1)
                name_x = self.node_name_mapping[n_i]
                x_str = list(map(np.array2string, list(x)))
                data_dict[name_x] = x_str
            self.dataframe = pd.DataFrame(data_dict)

        # start the test
        for i in range(len(self.node_dim_list)):
            for j in range(len(self.node_dim_list)):
                name_i = self.node_name_mapping[i]
                name_j = self.node_name_mapping[j]

                # directly add edges from S_xx to NS_xx
                if self._is_state(name_i) and self._is_next(name_j) and name_i.split('_')[1] == name_j.split('_')[1]:
                    self.causal_graph.add_edge(name_i, name_j)

                # for chemistry env, the causal direction will be lower triangular matrix
                if self.env_name in ['chemistry']:
                    if self._is_state(name_i) and self._is_next(name_j) and name_i.split('_')[1] > name_j.split('_')[1]:
                        continue

                action_state = self._is_action(name_i) and self._is_next(name_j)
                state_state = self._is_state(name_i) and self._is_next(name_j) and name_i.split('_')[1] != name_j.split('_')[1]
                if not action_state and not state_state:
                    continue

                # do independent test
                p_value = self._two_variable_test_chisquare(i, j)
                if p_value < self.pvalue_threshold:
                    self.causal_graph.add_edge(name_i, name_j)

        # visualize graph
        #self.visualize_graph(self.causal_graph, './log/causal_graph.png', directed=True)

    def get_true_causal_graph(self):
        if self.env_name == 'chemistry':
            truth_graph = nx.DiGraph()
            # add action edges
            for i in range(0, self.ground_truth.shape[0]):
                truth_graph.add_edge(self.node_name_mapping[0], self.node_name_mapping[i+self.next_state_offset])
            # add state edges
            for i in range(self.ground_truth.shape[0]):
                for j in range(self.ground_truth.shape[1]):
                    if self.ground_truth[j, i] == 1: # need to transpose
                        truth_graph.add_edge(self.node_name_mapping[i+1], self.node_name_mapping[j+self.next_state_offset])
        else:
            raise ValueError('Unknown Environment Name')

        return truth_graph

    def _retrieve_adjacency_matrix(self, graph, order_nodes=None, weight=False):
        """Retrieve the adjacency matrix from the nx.DiGraph or numpy array."""
        if isinstance(graph, np.ndarray):
            return graph
        elif isinstance(graph, nx.DiGraph):
            if order_nodes is None:
                order_nodes = graph.nodes()
            if not weight:
                return np.array(nx.adjacency_matrix(graph, order_nodes, weight=None).todense())
            else:
                return np.array(nx.adjacency_matrix(graph, order_nodes).todense())
        else:
            raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")
        
    def SHD(self, target, pred, double_for_anticausal=True):
        ''' Reference: https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/metrics.py '''
        true_labels = self._retrieve_adjacency_matrix(target)
        predictions = self._retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

        diff = np.abs(true_labels - predictions)
        if double_for_anticausal:
            return np.sum(diff)
        else:
            diff = diff + diff.transpose()
            diff[diff > 1] = 1  # Ignoring the double edges.
            return np.sum(diff)/2

    def visualize_graph(self, causal_graph, save_path=None, directed=True):
        plt.figure(figsize=(4, 7))

        left_node = []
        node_color = []
        for n_i in causal_graph.nodes:
            if self._is_action(n_i) or self._is_state(n_i):
                left_node.append(n_i)
            
            if self._is_action(n_i):
                node_color.append('#DA87B3')
            elif self._is_state(n_i):
                node_color.append('#86A8E7')
            else:
                node_color.append('#56D1C9')

        pos = nx.bipartite_layout(causal_graph, left_node)
        nx.draw_networkx(causal_graph, pos, node_color=node_color, arrows=directed, with_labels=True, node_size=1400, arrowsize=20)

        plt.axis('off')
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)
            plt.close('all')

    def get_adj_matrix_graph(self):
        # NOTE: discovered graph contains 2*N_S+N_A nodes, we need to convert it to N_S+N_A nodes
        node_mapping = {  
            'A_i': 0,                  
            'S_0': 1, 'S_1': 2, 'S_2': 3, 'S_3': 4, 'S_4': 5,
            'NS_0': 1, 'NS_1': 2, 'NS_2': 3, 'NS_3': 4, 'NS_4': 5,
        }
        adj_matrix = np.zeros((self.adj_node_num, self.adj_node_num))
        adj_matrix[0, 0] = 1
        edges = self.causal_graph.edges
        for e_i in edges:
            src_idx = node_mapping[e_i[0]]
            tar_idx = node_mapping[e_i[1]]
            adj_matrix[tar_idx, src_idx] = 1
        return adj_matrix

    def save_model(self, model_path, model_id):
        states = {'graph': self.causal_graph}
        filepath = os.path.join(model_path, 'graph.'+str(model_id)+'.pkl')
        with open(filepath, 'wb') as f:
            pkl.dump(states, f)

    def load_model(self, model_path, model_id):
        filepath = os.path.join(model_path, 'graph.'+str(model_id)+'.pkl')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = pkl.load(f)
            self.causal_graph = checkpoint['graph']
        else:
            raise Exception('No graph found!')
