'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2022-12-28 13:34:34
Description: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_input=7, n_output=6, n_h=1, size_h=128, dropout_p=0.0):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_list = nn.ModuleList()
        for _ in range(n_h):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)

    def forward(self, x):
        x = x.view(-1, self.n_input)
        out = self.fc_in(x)
        out = self.dropout(out)
        out = self.relu(out)
        for layer in self.fc_list:
            out = layer(out)
            out = self.dropout(out)
            out = self.relu(out)
        out = self.fc_out(out)
        return out


class IndividualEmbedding(nn.Module):
    def __init__(self, in_features, out_features, node_num, bias=True):
        super(IndividualEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(node_num, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(node_num, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        # x - [B, node_num, in_features]
        # W - [node_num, in_features, out_features]
        # output - [B,, node_num, out_features]
        output = []
        for n_i in range(x.shape[1]):
            o_i = torch.matmul(x[:, n_i, :], self.weight[n_i])
            if self.bias is not None:
                o_i += self.bias[n_i]
            o_i = o_i[:, None, :]
            output.append(o_i)

        output = torch.cat(output, dim=1)
        return output


class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer. 
    """
    def __init__(self, in_features, out_features, edge_dim, aggregate='mean', dropout=0., use_relu=False, bias=False):
        """
        Args:
            in_features: scalar of channels for node embedding
            out_features: scalar of channels for node embedding
            edge_dim: dim of edge type, virtual type not included
        """
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = None

        self.weight = nn.Parameter(torch.FloatTensor(self.edge_dim, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.edge_dim, 1, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, adj):
        """
        Args:
            x: (B, max_node_num, node_dim): 
            adj: (B, edge_dim, max_node_num, max_node_num): 

        Returns:
            node_embedding: (B, max_node_num, embed_size): updated embedding for nodes x
        """
        x = F.dropout(x, p=self.dropout, training=self.training)  # (B, max_node_num, node_dim)

        # transform
        support = torch.einsum('bid, edh-> beih', x, self.weight) # (B, edge_dim, max_node_num, embed_size)

        # works as a GCN with sum aggregation
        output = torch.einsum('beij, bejh-> beih', adj, support)  # (B, edge_dim, max_node_num, embed_size)

        if self.bias is not None:
            output += self.bias
        if self.act is not None:
            output = self.act(output)  # (B, E, N, d)

        if self.aggregate == 'sum':
            # sum pooling #(b, N, d)
            node_embedding = torch.sum(output, dim=1, keepdim=False)
        elif self.aggregate == 'max':
            # max pooling  #(b, N, d)
            node_embedding = torch.max(output, dim=1, keepdim=False)
        elif self.aggregate == 'mean':
            # mean pooling #(b, N, d)
            node_embedding = torch.mean(output, dim=1, keepdim=False)
        else:
            print('GCN aggregate error!')
        return node_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGCN(nn.Module):
    def __init__(self, node_dim, node_num, aggregate, hidden_dim, output_dim, edge_dim, hidden_num, dropout=0.0, bias=True):
        """
        Args:
            node_dim:
            hidden_dim:
            output_dim:
            edge_dim:
            num_layars: the number of layers in each R-GCN
            dropout:
        """
        super(RGCN, self).__init__()
        self.hidden_num = hidden_num

        self.emb = nn.Linear(node_dim, hidden_dim, bias=bias) 
        self.ind_emb = IndividualEmbedding(node_dim, hidden_dim, node_num=node_num, bias=bias) 

        self.gc1 = RelationGraphConvolution(hidden_dim, hidden_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=True, dropout=dropout, bias=bias)
        self.gc2 = nn.ModuleList([RelationGraphConvolution(hidden_dim, hidden_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=True, dropout=dropout, bias=bias) for i in range(hidden_num)])
        self.gc3 = RelationGraphConvolution(hidden_dim, output_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=False, dropout=dropout, bias=bias)

    def forward(self, x, adj):
        # embedding layer (individual for each node)
        x = self.ind_emb(x)

        # first GCN layer
        x = self.gc1(x, adj)

        # hidden GCN layer(s)
        for i in range(self.hidden_num):
            x = self.gc2[i](x, adj)  # (#node, #class)

        # last GCN layer
        x = self.gc3(x, adj)  # (batch, N, d)

        # return node embedding
        return x


class MPNN(nn.Module):
    def __init__(self, action_dim_list, state_dim_list, node_num, aggregate, hidden_dim, edge_dim, hidden_num, dropout=0.0, bias=True):
        """
        Args:
            node_dim:
            hidden_dim:
            output_dim:
            edge_dim:
            num_layars: 
            dropout:
        """
        super(MPNN, self).__init__()
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.node_num = node_num

        # NOTE: assume we dont input the padding zero, just use seperate embedding layers with different size of input dimensions
        self.node_list = action_dim_list + state_dim_list
        self.input_embs = []
        for i in self.node_list:
            self.input_embs.append(nn.Linear(i, hidden_dim))
        self.input_embs = nn.ModuleList(self.input_embs)

        self.output_embs = []
        for i in self.node_list:
            self.output_embs.append(nn.Linear(hidden_dim, i))
        self.output_embs = nn.ModuleList(self.output_embs)

        self.gc_layer = nn.ModuleList([RelationGraphConvolution(hidden_dim, hidden_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=True, dropout=dropout, bias=bias) for i in range(hidden_num)])
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru.reset_parameters()

    def forward(self, x, adj):
        # x - [B, N, d] - [B, A+S, d]
        # adj - [B, E, N, N]

        # extract the true nodes then pass them throught the emdeddings
        x_list = []
        for e_i, embd_i in enumerate(self.input_embs):
            x_i = embd_i(x[:, e_i, 0:self.node_list[e_i]])[:, None, :]
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)

        # hidden GCN layer(s)
        hidden = x.reshape(1, -1, self.hidden_dim)
        for i in range(self.hidden_num):
            x = self.gc_layer[i](x, adj)  # (#node, #class)
            x = x.reshape(-1,  1, self.hidden_dim)
            x, hidden = self.gru(x, hidden)
            x = x.reshape(-1, self.node_num, self.hidden_dim)

        # convert the embedding back to the true node for loss calculation
        x_padded = torch.zeros_like(x) # [B, N, d]
        for e_i, embd_i in enumerate(self.output_embs):
            x_i = embd_i(x[:, e_i, :])
            print(x_i.shape)
            x_padded[:, e_i, 0:self.node_list[e_i]] = x_i
        return x_padded


class GRU_SCM(nn.Module):
    def __init__(self, action_dim_list, state_dim_list, node_num, aggregate, hidden_dim, edge_dim, hidden_num, dropout=0.0, bias=True, random=False):
        super(GRU_SCM, self).__init__()
        self.random = random
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.node_num = node_num

        # NOTE: assume we dont input the padding zero, just use seperate embedding layers with different size of input dimensions
        self.node_list = action_dim_list + state_dim_list
        self.relu = nn.ReLU()
        self.input_embs = []
        for i in self.node_list:
            self.input_embs.append(nn.Linear(i, hidden_dim))
        self.input_embs = nn.ModuleList(self.input_embs)

        self.output_embs = []
        for i in self.node_list:
            self.output_embs.append(nn.Linear(hidden_dim, i))
        self.output_embs = nn.ModuleList(self.output_embs)

        self.gru_list = []
        for i in range(len(self.node_list)):
            self.gru_list.append(nn.GRU(hidden_dim, hidden_dim, num_layers=self.hidden_num, batch_first=True))
        self.gru_list = nn.ModuleList(self.gru_list)

    def forward(self, x_in, adj):
        # x - [B, N, d] - [B, A+S, d]
        # adj - [B, E, N, N]

        # diagnal is not necessary for aggregation
        adj = adj[0, 0] - torch.eye(x_in.shape[1], device=adj.device)

        # extract the true nodes then pass them throught the emdeddings
        x_list = []
        for e_i, embd_i in enumerate(self.input_embs):
            x_i = embd_i(x_in[:, e_i, 0:self.node_list[e_i]])[:, None, :]
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        x = self.relu(x)

        agg_list = []
        for n_i in range(x.shape[1]):
            hidden = x[:, n_i:n_i+1, :].transpose(0, 1).contiguous()

            # add noise U to the hidden variable
            if self.random:
                noise = (torch.rand(hidden.shape, device=hidden.device) - 0.5) * 0.1
                hidden = hidden + noise

            # NOTE: GRU requires the order to be fixed, while attention is permuational
            neighbors_idx = torch.nonzero(adj[n_i], as_tuple=False)[:, 0]
            if len(neighbors_idx) == 0:
                # if there is no neighbor, use the embedding itself
                agg_list.append(hidden.transpose(0, 1))
            else:
                neighbors = x[:, neighbors_idx, :].contiguous()
                aggregation, _ = self.gru_list[n_i](neighbors, hidden)
                agg_list.append(aggregation[:, -1:, :])
        x = torch.cat(agg_list, dim=1)
        x = self.relu(x)

        # convert the embedding back to the true node for loss calculation
        x_padded = torch.zeros_like(x_in) # [B, N, d]
        for e_i, embd_i in enumerate(self.output_embs):
            x_i = embd_i(x[:, e_i, :])
            x_padded[:, e_i, 0:self.node_list[e_i]] = x_i
        return x_padded
