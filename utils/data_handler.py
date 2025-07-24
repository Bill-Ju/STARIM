import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
class MultiDataHandler:
    def __init__(self, dataset_list, args):
        self.handlers = []
        for data_name in dataset_list:
            network, cascade_data_suffix = data_name.split('@')
            handler = DataHandler(network, cascade_data_suffix, args)
            self.handlers.append(handler)
class DataHandler:
    def __init__(self, network, cascade_data_suffix, args):
        self.network = network
        self.cascade_data_suffix = cascade_data_suffix
        self.pwd = args.pwd
        self.load_data(args)
        
    def load_data(self, args):
        predir = self.pwd + f'/data/{self.network}/'
        self.network_edges_path = predir+f'{self.network}.txt'
        
        data_mode = self.cascade_data_suffix.split('_')[0]
        
        if not args.direct:
            with open(self.network_edges_path, 'rb')as edges_f:
                G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                        data=(('weight', float),))
            self.edge_index = []
            # 添加所有边
            for edge in G.edges():
                self.edge_index.append([edge[0], edge[1]])
                self.edge_index.append([edge[1], edge[0]])  # 如果是无向图，添加反向边
        else:
            with open(self.network_edges_path, 'rb')as edges_f:
                G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph(), encoding='latin1',
                                        data=(('weight', float),))
            self.edge_index = []
            for edge in G.edges():
                self.edge_index.append([edge[0], edge[1]])
        self.edge_index = torch.tensor(self.edge_index)
        if args.data_guass:
            cascade_data_path = predir+f'{data_mode}/{self.network}_{self.cascade_data_suffix}_gauss.npy'
        else:
            cascade_data_path = predir+f'{data_mode}/{self.network}_{self.cascade_data_suffix}.npy'
        if args.feat_for_prompt:
            feat_for_prompts_path = predir+f'{data_mode}/{self.network}_{self.cascade_data_suffix}_feat_for_prompts.npy'
            if os.path.exists(feat_for_prompts_path):
                feat_for_prompts = np.load(feat_for_prompts_path)
                print("feat_for_prompts文件存在")
                
            else:
                feat_for_prompts = None
                print("feat_for_prompts文件不存在")
                
        else:
            feat_for_prompts = None
        if not os.path.exists(cascade_data_path):
            return
        cascade_data = np.load(cascade_data_path)
        print(f"{cascade_data_path}文件存在")
        if args.mode == 'pretrain':
            cascade_dataset = CascadeDataset(cascade_data,feat_for_prompts, 'pretrain')
            self.pretrain_data_loader = DataLoader(cascade_dataset, batch_size=args.pretrain_batch, shuffle=True, drop_last=False)
        else:
            cascade_train_dataset = CascadeDataset(cascade_data,feat_for_prompts, 'train', args.train_size_for_few_shot)
            cascade_validate_dataset = CascadeDataset(cascade_data,feat_for_prompts, 'validate')
            cascade_test_dataset = CascadeDataset(cascade_data,feat_for_prompts, 'test')
            self.train_data_loader = DataLoader(cascade_train_dataset, batch_size=args.tuning_batch, shuffle=True, drop_last=False)
            self.validate_data_loader = DataLoader(cascade_validate_dataset, batch_size=args.test_batch, shuffle=False, drop_last=False)
            self.test_data_loader = DataLoader(cascade_test_dataset, batch_size=args.test_batch, shuffle=False, drop_last=False)

class CascadeDataset(Dataset):
    def __init__(self, cascade_data,feat_for_prompts, mode, train_size_for_few_shot = -1):
        cas_num = cascade_data.shape[0]
        cas_num = 2000
        if feat_for_prompts is None:
            feat_for_prompts = cascade_data
        if mode == 'pretrain':
            self.cascade_data = torch.tensor(cascade_data[:int(cas_num)])
            self.feat_for_prompts = torch.tensor(feat_for_prompts[:int(cas_num)])
        elif mode == 'train':
            self.cascade_data = torch.tensor(cascade_data[:int(train_size_for_few_shot*cas_num)])
            self.feat_for_prompts = torch.tensor(feat_for_prompts[:int(train_size_for_few_shot*cas_num)])
        elif mode == 'validate':
            self.cascade_data = torch.tensor(cascade_data[int(0.6*cas_num):int(0.8*cas_num)])
            self.feat_for_prompts = torch.tensor(feat_for_prompts[int(0.6*cas_num):int(0.8*cas_num)])
        else:
            self.cascade_data = torch.tensor(cascade_data[int(0.8*cas_num):cas_num])
            self.feat_for_prompts = torch.tensor(feat_for_prompts[int(0.8*cas_num):cas_num])
        
        
    def __len__(self):
        return len(self.cascade_data)
    def __getitem__(self, idx):
        data = self.cascade_data[idx], self.feat_for_prompts[idx]
        return data