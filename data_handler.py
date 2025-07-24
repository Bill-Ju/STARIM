import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import os
from torch_geometric.data import Data, DataLoader

class MultiDataHandler:
    def __init__(self, dataset_list, empirical_dataset_list, args):
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
        
        for data_name in empirical_dataset_list:
            network, cascade_data_suffix = data_name.split('@')
            handler = DataHandler(network, [cascade_data_suffix], args, empirical= True)
            self.handlers.append(handler)
            self.nets.append(network)
        
        cascade_data_list = []
        prompts_list = []
        edge_index_list = []
        # node_num_list = []
        # edge_num_list = []
        self.dataset = []
        for handler in self.handlers:
            
            for i in range(len(handler.edge_index_list)):
                data = Data(cas=handler.cascade_data[i][:args.Tstep+1], edge_index=handler.edge_index_list[i], prompt = handler.prompts[i], idx = i)
                data.num_nodes = handler.cascade_data[i].shape[-2]
                self.dataset.append(data)
            
class DataHandler:
    def __init__(self, network, cascade_data_suffix_list, args, empirical = False):
        self.network = network
        self.cascade_data_suffix_list = cascade_data_suffix_list
        self.pwd = args.pwd
        self.load_data(args, empirical)
        
    def load_data(self, args, empirical):
        predir = self.pwd + f'/data/{self.network}/'
        target_length = args.Tstep+1
        print(f'load data for {self.network}!')
        if empirical:
            cascade_data_suffix = self.cascade_data_suffix_list[0]
            network_edges_path = predir + f'cas/{self.network}_edge_index_list.pkl'
            cascade_data_path = predir+f'cas/{self.network}_{cascade_data_suffix}.pkl'
            with open(network_edges_path, 'rb') as f:
                loaded_edge_index = pickle.load(f)
            with open(cascade_data_path, 'rb') as f:
                loaded_cascade_data= pickle.load(f)
            
            cas = loaded_cascade_data['cas']
            prompts = [f'{self.network}_{cascade_data_suffix}']*len(cas)         
            self.cascade_data = loaded_cascade_data['cas']
            self.prompts = prompts
            self.edge_index_list = loaded_edge_index
        else:
            self.network_edges_path = predir+f'{self.network}.txt'
            direct = False
            cascade_data_list = []
            feat_for_prompt_list = []
            for cascade_data_suffix in self.cascade_data_suffix_list:
                data_mode = cascade_data_suffix.split('_')[0]
                suffix = cascade_data_suffix.split('_')[-1]
                if suffix == 'direct':
                    direct = True
                    cascade_data_suffix = cascade_data_suffix[:-7]
                cascade_data_path = predir+f'{data_mode}/{self.network}_{cascade_data_suffix}.npy'
                cascade_data = np.load(cascade_data_path)
                prompts = [cascade_data_suffix]*len(cascade_data)
                print(f"{cascade_data_path}文件存在")
                cascade_data_list.append(cascade_data[:,:args.Tstep+1])
                feat_for_prompt_list.append(prompts)
                
                cascade_data = np.concatenate(cascade_data_list, axis=0)
                prompts = np.concatenate(feat_for_prompt_list, axis=0)
                indices = np.random.permutation(len(cascade_data))
                cascade_data = cascade_data[indices]
                prompts = prompts[indices]
                cas_num = cascade_data.shape[0]
                self.cascade_data = cascade_data[:args.sample_num]
                self.prompts = prompts[:args.sample_num]            
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
            
            # self.edge_index = process_edge_index_numpy(self.edge_index)
            if self.edge_index is None:
                print("graph is too big!!!")
                return
            
            
            self.edge_index = torch.tensor(self.edge_index).t().contiguous()
            
            

            self.edge_index_list = [self.edge_index]*args.sample_num

# class CascadeDataset(Dataset):
#     def __init__(self, cascade_data, edge_index_list, prompts):
#         self.dataset = []
#         for i in range(len(edge_index_list)):
#             data = Data(cas=cascade_data[i], edge_index=edge_index_list[i], prompt = prompts[i])
#             self.dataset.append(data)
#         # self.cascade_data = cascade_data
#         # self.edge_index_list = edge_index_list
#         # self.prompts = prompts
#         # self.node_num = node_num
#         # self.edge_num = edge_num

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         data = self.cascade_data[idx], self.edge_index_list[idx], self.prompts[idx], idx, self.node_num[idx], self.edge_num[idx]
#         return data
    
# def process_edge_index_numpy(edge_index, max_size=500000):
#     """
#     检查 edge_index 的形状，如果其 shape[1] 超过 max_size，则舍弃，
#     否则填充到 max_size，使用 -1 填充。
    
#     参数:
#     edge_index (ndarray): 形状为 (2, N) 的边的索引。
#     max_size (int): 填充的最大大小，默认 50 万。

#     返回:
#     ndarray: 处理后的 edge_index。
#     """
#     if edge_index.shape[1] > max_size:
#         # 如果 shape[1] 超过 max_size，舍弃该边
#         return None
#     else:
#         # 否则填充到 max_size，使用 -1 填充
#         padding_size = max_size - edge_index.shape[1]
#         padding = np.full((2, padding_size), -1, dtype=np.int64)
#         padded_edge_index = np.hstack([edge_index, padding])
#         return padded_edge_index
                