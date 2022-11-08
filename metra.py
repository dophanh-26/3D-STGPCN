import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
import json
from utils import *

class ZScaler:
    def scale(self, data, mean, std):
        return (data - mean) / std 

    def inverse_scale(self, data, mean, std):
        return (data * std) + mean


class MetrLA(DGLDataset):
    def __init__(self, history_length, future_length, train_ratio = 0.8):
        super().__init__(name = "metr-la")
        self.history_length = history_length
        self.future_length = future_length
        self.train_ratio = train_ratio
        self.process()

    def _load(self):
        pass

    def process(self):
        A = np.load('./data/METR-LA/adj_mat.npy') 
        X = np.load('./data/METR-LA/node_values.npy').transpose(1,2,0)
        
        # X.shape = (num_node, feat_dim, num_timestep)

        # Calculate the mean of each feature in X
        '''
        mean_X = np.mean(X, axis = (0,2))
        std_X = np.std(X, axis = (0,2))
        X = X - mean_X.reshape(1, -1, 1)
        X = X / std_X.reshape(1,-1, 1)
        '''
        
        self.signals = X
        self.num_nodes = self.signals.shape[0]
        #self.node_stats = {'mean' : mean_X[0], 'std' : std_X[0]}

        tmp_g = nx.DiGraph(A)
        edge_list = list(nx.to_edgelist(tmp_g))

        src = [edge[0] for edge in edge_list]
        dst = [edge[1] for edge in edge_list]
        edge_weight = [edge[2]['weight'] for edge in edge_list]

        self.graph = dgl.graph((src, dst), num_nodes = self.signals.shape[0])
        self.graph.edata['weight'] = torch.tensor(edge_weight).float()

        self.node_features, self.targets = [], []
        self.num_sequences = self.signals.shape[2] - (self.history_length + self.future_length) + 1

        for i in range(self.num_sequences):
            self.node_features.append((self.signals[:, :, i : i + self.history_length]))
            self.targets.append((self.signals[:, 0, i + self.history_length : i + self.history_length + self.future_length]))
        
        self.node_features = np.stack(self.node_features, axis = 0)
        self.targets = np.stack(self.targets, axis = 0)

        self._normalize()

    def _normalize(self):
        train_size = int(self.train_ratio * self.num_sequences)
        _mean = np.mean(self.node_features[: train_size, :, :, :], axis = (0,1,3))
        _std = np.std(self.node_features, axis = (0,1,3))
        self.node_stats = {'mean' : _mean[0], 'std' : _std[0]}
        
        self.node_features = self.node_features - _mean.reshape(1, 1, -1, 1)
        self.node_features = self.node_features / _std.reshape(1,1, -1, 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        node_feature = self.node_features[idx]
        target = self.targets[idx]

        graph = deepcopy(self.graph)

        graph.ndata['feat'] = torch.tensor(node_feature).float()
        graph.ndata['target'] = torch.tensor(target).float()

        return graph


class BuffaloGrove(DGLDataset):
    def __init__(self, history_length, future_length, train_ratio = 0.8):
        super().__init__(name = 'bufferlo_grove')
        self.history_length = history_length
        self.future_length = future_length
        self.train_ratio = train_ratio
        self.process()

    def _load(self):
        pass
    
    def _read_data(self):
        url = './data/buffalogrove-traffic-data.json'
        with open(url, 'r') as file_:
            self.data = json.load(file_)

    def process(self):
        self._read_data()
        
        edge_list = np.array(self.data['edges'])

        # (num_node, num_timestep)
        X1 = np.array(self.data['count']['2018']).T
        X1 = X1.reshape(X1.shape[0], 1, X1.shape[1])

        X2 = np.array(self.data['state']['2018']).T
        X2 = X2.reshape(X2.shape[0], 1, X2.shape[1])

        X =np.concatenate([X1, X2], axis = 1)
        ''' 
        mean_X = np.mean(X, axis = (0,2))
        print(mean_X)
        std_X = np.std(X, axis = (0,2))
       # X = X - mean_X.reshape(1, -1, 1)
       # X = X / std_X.reshape(1,-1, 1)
       '''
        self.signals = X
        self.num_nodes = self.signals.shape[0]
        #self.node_stats = {'mean' : mean_X[0], 'std' : std_X[0]}

        src = [edge[0] for edge in edge_list]
        dst = [edge[1] for edge in edge_list]

        self.graph = dgl.graph((src, dst), num_nodes = self.signals.shape[0])
        self.graph = self.graph.add_self_loop()
        self.graph.edata['weight'] = torch.ones(self.graph.num_edges()).float() 

        self.node_features, self.targets = [], []
        self.num_sequences = self.signals.shape[2] - (self.history_length + self.future_length) + 1

        for i in range(self.num_sequences):
            self.node_features.append(self.signals[:, :, i : i + self.history_length])
            self.targets.append(self.signals[:, 0, i + self.history_length : i + self.history_length + self.future_length])

        self.node_features = np.stack(self.node_features, axis = 0)
        self.targets = np.stack(self.targets, axis = 0)
        self._normalize() 
    
    def _normalize(self):
        train_size = int(self.train_ratio * self.num_sequences)
        _mean = np.mean(self.node_features[: train_size, :, :, :], axis = (0,1,3))
        _std = np.std(self.node_features, axis = (0,1,3))
        self.node_stats = {'mean' : _mean[0], 'std' : _std[0]}
        
        self.node_features = self.node_features - _mean.reshape(1, 1, -1, 1)
        self.node_features = self.node_features / _std.reshape(1,1, -1, 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        node_feature = self.node_features[idx]
        target = self.targets[idx]

        graph = deepcopy(self.graph)

        graph.ndata['feat'] = torch.tensor(node_feature).float()
        graph.ndata['target'] = torch.tensor(target).float()

        return graph


class PEMS04(DGLDataset):
    def __init__(self, history_length, future_length, train_ratio = 0.8):
        super().__init__(name = 'PEMS04')
        self.history_length = history_length
        self.future_length = future_length
        self.train_ratio = train_ratio
        self.process()

    def _load(self):
        pass

    def process(self):
        '''
        order of features: 0: flow, 1: occupy, 2: speed 
        target: flow
        
        '''
        edge_list = pd.read_csv('./data/PEMS04/PEMS04.csv')
        X = np.load('./data/PEMS04/PEMS04.npz')['data']
        X = np.transpose(X, (1, 2, 0))
        
        mean_X = np.mean(X, (0,2))
        std_X = np.std(X, (0, 2))
        X = (X - mean_X.reshape(1, -1, 1)) / std_X.reshape(1,-1,1)

        self.signals = X
        self.num_nodes = self.signals.shape[0]
        #self.node_stats = {'mean' : mean_X[0], 'std' : std_X[0]}

        src = edge_list['from'].tolist() 
        dst = edge_list['from'].tolist()
        edge_weight = edge_list['cost'].tolist() 

        self.graph = dgl.graph((src, dst), num_nodes = self.signals.shape[0])
        self.graph.edata['weight'] = torch.tensor(edge_weight).float()
        self.node_features, self.targets = [], []
        self.num_sequences = self.signals.shape[2] - (self.history_length + self.future_length) + 1

        for i in range(self.num_sequences):
            self.node_features.append(self.signals[:, : , i : i + self.history_length])
            self.targets.append(self.signals[:, 0, i + self.history_length : i + self.history_length + self.future_length])

        self.node_features = np.stack(self.node_features, axis = 0)
        self.targets = np.stack(self.targets, axis = 0)
        self._normalize()
    
    def _normalize(self):
        train_size = int(self.train_ratio * self.num_sequences)
        _mean = np.mean(self.node_features[: train_size, :, :, :], axis = (0,1,3))
        _std = np.std(self.node_features, axis = (0,1,3))
        self.node_stats = {'mean' : _mean[0], 'std' : _std[0]}
        
        self.node_features = self.node_features - _mean.reshape(1, 1, -1, 1)
        self.node_features = self.node_features / _std.reshape(1,1, -1, 1)
    
    def __len__(self):
        return self.num_sequences

    
    def __getitem__(self, idx):
        node_feature = self.node_features[idx]
        target = self.targets[idx]

        graph = deepcopy(self.graph)
        graph.ndata['feat'] = torch.tensor(node_feature).float()
        graph.ndata['target'] = torch.tensor(target).float()
        return graph

if __name__ == '__main__':
    data = MetrLA(12, 12, train_ratio = 1.0)
    print("Num samples: ", len(data))
    print("Num node: ", data.num_nodes)
    
    g1 = data[0]
    g2 = data[1]
    print(id(g1))
    print(id(g2))
    print(id(g1) == id(g2))
    print(g1.ndata['feat'].shape)
    print(g1.ndata['target'].shape)

    loader = GraphDataLoader(data, batch_size = 3)

    for batch in loader:
        h = batch.ndata['feat']
        print(h.shape)
        print(batch.ndata['target'].shape)
        break   