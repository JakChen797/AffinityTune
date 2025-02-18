import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GINConv
import math
from utils import idx_shuffle, row_normalization


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation, layers, dropout):
        super(MLP, self).__init__()
        
        # Input layer
        encoder = [nn.Linear(in_dim, out_dim), activation, nn.Dropout(dropout)]
        
        # Hidden layers
        for i in range(layers - 1):
            encoder.append(nn.Linear(out_dim, out_dim))
            encoder.append(nn.Dropout(dropout))
            encoder.append(activation)
            
        # Sequential model
        self.mlp = nn.ModuleList(encoder)

    def forward(self, feats):
        h = feats
        for layer in self.mlp:
            h = layer(h)
        h = F.normalize(h, p=2, dim=1)  # 行归一化
        return h
         

class GNN(nn.Module):
    def __init__(self, g, in_dim, hid_dim, activation, layers, dropout):
        super(GNN, self).__init__()
        self.g = g
        self.gcn = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        # # GCN
        # # input layer
        # self.gcn.append(GraphConv(in_dim, hid_dim, activation=activation))
        
        # # hid layer
        # for i in range(layers - 1):
        #     self.gcn.append(GraphConv(hid_dim, hid_dim, activation=activation))

        # GIN
        self.gcn.append(GINConv(nn.Linear(in_dim, hid_dim), learn_eps=True))
        for _ in range(layers - 1):
            self.gcn.append(GINConv(nn.Linear(hid_dim, hid_dim), learn_eps=True))

    def forward(self, feats):    
        h = feats
        for layer in self.gcn:
            h = layer(self.g, h)
            h = self.dropout(h)
        # h = F.normalize(h, p=2, dim=1)  # 行归一化
        return h


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def extract_H_diff(self, graph, h, cluster_ids, mode = 'local'):
        if mode == 'local':
        # 与邻居均值的差值
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                neigh_means = graph.ndata['neigh']
                diff = h - neigh_means
                return diff
        elif mode == 'cluster':
            # 与聚类均值的差值
            if cluster_ids is None:
                raise ValueError("cluster_ids is required when mode='cluster'")
            
            cluster_ids = torch.tensor(cluster_ids)
            # 计算各聚类的均值
            unique_clusters = torch.unique(cluster_ids)
            cluster_means = []
            for c in unique_clusters:
                indices = torch.where(cluster_ids == c)[0]
                cluster_h = h[indices]
                mean_h = torch.mean(cluster_h, dim=0)
                cluster_means.append(mean_h)
            cluster_means = torch.stack(cluster_means)
            
            # 扩展聚类均值以匹配节点特征
            expanded_cluster_means = cluster_means[cluster_ids]
            
            # 计算节点与聚类均值的差值
            diff = h - expanded_cluster_means
            return diff
        else:
            raise ValueError("Invalid mode. Supported modes are 'local' and 'cluster'.")



class Discriminator(nn.Module):
    def __init__(self, hid_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)
    
    def forward(self, features, centers, mode):
        assert mode=='local' or 'global', "mode must be local or global"
        if mode == 'local':
            tmp = torch.matmul(features, self.weight) # tmp = xW^T
            res = torch.sum(tmp * centers, dim=1) # res = <tmp, s>
        else:
            res = torch.matmul(features, torch.matmul(self.weight, centers))  # xW^Tg
        # sigmoid在BCEWithLogitloss中内置
        return res 