import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import os
from torch_geometric.data import Data, DataLoader

class MultiDataHandler:
    def __init__(self, dataset_list, args):
        self.handlers = []
        net_cas_dict = {}
        for data_name in dataset_list:
            network, cascade_data_suffix = data_name.split('@')
            if network not in net_cas_dict.keys():
                net_cas_dict[network] = []
            net_cas_dict[network].append(cascade_data_suffix)
        self.nets = []
        for network,cascade_data_suffix_list in net_cas_dict.items():
            handler = DataHandler(network, cascade_data_suffix_list, args)
            self.handlers.append(handler)
            self.nets.append(network)
        
        self.dataset = []
        for handler in self.handlers:
            for i in range(len(handler.edge_index_list)):
                data = Data(cas=handler.cascade_data[i][:args.Tstep+1], edge_index=handler.edge_index_list[i], idx = i)
                data.num_nodes = handler.cascade_data[i].shape[-2]
                self.dataset.append(data)
            
class DataHandler:
    def __init__(self, network, cascade_data_suffix_list, args):
        self.network = network
        self.cascade_data_suffix_list = cascade_data_suffix_list
        self.pwd = args.pwd
        self.load_data(args)
        
    def load_data(self, args):
        predir = self.pwd + f'/data/{self.network}/'

        print(f'load data for {self.network}!')

        self.network_edges_path = predir+f'{self.network}.txt'
        direct = False
        cascade_data_list = []

        for cascade_data_suffix in self.cascade_data_suffix_list:
            data_mode = cascade_data_suffix.split('_')[0]
            suffix = cascade_data_suffix.split('_')[-1]
            if suffix == 'direct':
                direct = True
                cascade_data_suffix = cascade_data_suffix[:-7]
            cascade_data_path = predir+f'{data_mode}/{self.network}_{cascade_data_suffix}.npy'
            cascade_data = np.load(cascade_data_path)

            print(f"{cascade_data_path}文件存在")
            cascade_data_list.append(cascade_data[:,:args.Tstep+1])

            
            cascade_data = np.concatenate(cascade_data_list, axis=0)

            indices = np.random.permutation(len(cascade_data))
            cascade_data = cascade_data[indices]

            cas_num = cascade_data.shape[0]
            self.cascade_data = cascade_data[:args.sample_num]
          
        if direct:
            with open(self.network_edges_path, 'rb')as edges_f:
                G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph(), encoding='latin1',
                                            data=(('weight', float),))
        else:
            with open(self.network_edges_path, 'rb')as edges_f:
                G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                            data=(('weight', float),))
        
        self.edge_index = []

        if direct:
            # 添加所有边
            for edge in G.edges():
                self.edge_index.append([edge[0], edge[1]])
        else:
            # 添加所有边
            for edge in G.edges():
                self.edge_index.append([edge[0], edge[1]])
                self.edge_index.append([edge[1], edge[0]])  # 如果是无向图，添加反向边

        self.edge_index = torch.tensor(self.edge_index).t().contiguous()
        self.edge_index_list = [self.edge_index]*args.sample_num
                