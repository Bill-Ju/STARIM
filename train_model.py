import os
import torch 
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from params import args
from datetime import datetime
from torch_geometric.loader import DataLoader

from utils.data_handler import MultiDataHandler
from model.nn_model import Diffuion
from torch import optim

class Exp:
    def __init__(self, multi_data_handler):
        self.multi_data_handler = multi_data_handler
    
    def run(self, rank, world_size, args):
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['MASTER_ADDR'] = 'localhost'  
        os.environ['MASTER_PORT'] = '12366'     
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        self.prepare_model(args)
        if args.load_model:
            self.load_model()
        self.train(rank, world_size, args)
        dist.barrier()
        if rank == 0:
            self.save_history()
        dist.destroy_process_group()
        
    def prepare_model(self, args):
        self.model = Diffuion(args)
        if args.load_model:
            self.load_model()
        
    def train(self, rank, world_size, args):
        self.model.train()

        print('==========================train==============================')
            
        print(
            "Train_batch: {}".format(args.train_batch))
        print(f"===epoch num :{args.train_epoch}===")
        
        self.loss = torch.zeros(args.train_epoch)
        dataset = self.multi_data_handler.dataset
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        self.train_loader = DataLoader(dataset, batch_size=args.train_batch, sampler=train_sampler)
        device = torch.device(f'cuda:{rank}')
        print(f"device:{device}")
        self.model = self.model.to(device)
        self.optim = optim.Adam(list(self.model.backbone.parameters()), lr=args.lr)
        # train
        for epoch_index in range(args.train_epoch):
            train_sampler.set_epoch(epoch_index) 
            batch_index = -1
            loss_epoch = 0
            for batch in self.train_loader:
                batch_index += 1
                # cascade_data, edge_index_list, idxs, node_num, edge_num = torch.tensor(cascade_data).to(torch.float32).to(device),torch.tensor(edge_index_list).to(device),torch.tensor(idxs).to(device),torch.tensor(node_num).to(device),torch.tensor(edge_num).to(device)
                batch = batch.to(device)
                self.optim.zero_grad()
                loss_sum  = self.model(batch, args)
                loss_t = loss_sum.item()
                if torch.isnan(loss_sum):
                    print("loss is nan")
                    continue
                loss_sum.backward()
                self.optim.step()

                print("Epoch: {}, train, batch_data_index:{}".format(epoch_index+1, batch_index), 
                    "\tTrain_loss:{}".format(loss_t),
                )
                
                loss_epoch += loss_t
            loss_epoch /= (batch_index+1)
            self.loss[epoch_index] = loss_epoch
            print("Epoch: {}, train".format(epoch_index+1), 
                "\tTrain_epoch_loss: [{}]".format(loss_epoch),
                )
        print('=========================train finish===============================')
        

    def save_history(self):  
        print('=========================save medel===============================')
        
        # 提取月和日
        self.model.eval()
        content = {
            'model': self.model.cpu(),
        }
        save_path = args.pwd+f'/result/{args.data_mode}/models/train_net.mod' 
        dir_name = os.path.dirname(save_path)
        # 如果目录不存在，则创建目录
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        torch.save(content, save_path)
        print(f'Model Saved, Save path:{save_path}')
        
    def load_model(self):
      
        load_path = args.pwd+f'/result/{args.data_mode}/models/train_net.mod'
        # 检查路径是否存在
        if os.path.exists(load_path):
            print(f"文件存在: {load_path}")
        else:
            print(f"文件不存在: {load_path}")
            return
        ckp = torch.load(load_path)
        self.model = ckp['model']
        print(f'Model Loaded, Load Path:{load_path}')

def train(rank, world_size, multiDataHandler, args):
    exp = Exp(multiDataHandler)
    exp.run(rank, world_size, args)
    

if __name__ == '__main__':  
    datasets = dict()
    datasets['networks'] = ['ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food','Lastfm_asia','Router','Soc-hamsterster', 'USA airports','Yeast','wiki-Vote','p2p-Gnutella05','p2p-Gnutella06','soc-Slashdot0811']
    datasets['undirect_net'] = [
        'ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food','Lastfm_asia','Router','Soc-hamsterster', 'USA airports','Yeast'
    ]
    datasets['direct_net'] = [
        'wiki-Vote','congress-twitter', 'soc-dolphins', 'bitcoin-alpha','bitcoin-otc'
    ]

    train_network_list = ['ca-GrQc','Celegans', 'cora','ego-Facebook','fb-pages-food'][0:2]
    dataset_list = []
    suffix = args.propagation_data
    
    for net in train_network_list:
        if net in datasets['undirect_net']:
            dataset = net + '@' + suffix
            dataset_list.append(dataset)
        elif net in datasets['direct_net']:
            dataset = net + '@' + 'direct' + '@'+ suffix
            dataset_list.append(dataset)
    
    handler = MultiDataHandler(dataset_list, args)
    
    world_size = 3

    mp.spawn(train, args=(world_size,handler, args), nprocs=world_size, join=True)
    
    
