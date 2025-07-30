import os
import torch 
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from params import args

from utils.data_handler import MultiDataHandler,DataHandler
from utils import graph_utils
import networkx as nx

class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Sigmoid + Threshold 0.5
        return (x > 0.95).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计：忽略非连续性，直接传梯度
        return grad_output

class Exp:
    def __init__(self):
        self.load_model()
    
    def run(self, directed = False):
        self.im(args, directed)
        
    def im(self, args, directed): 
        
        predir = args.pwd + f'/data/{self.network}/'
        self.network_edges_path = predir+f'{self.network}.txt'
        

        if not directed:
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
        edge_index = torch.tensor(self.edge_index).t().contiguous().to(args.device)
        
        graph, N = graph_utils.read_graph(self.network_edges_path, ind=0, directed=directed)
        

            
        seed_num = int(N*seed_rate)


        x_init = torch.ones(N).to(args.device)

        x_init.requires_grad = True

        initial_momentum = 0.95
        z_optimizer = SGD([x_init], lr=args.lr_z, momentum=initial_momentum)
        max_x = None
        max_influence_num = 0
        now_influence_num = 0
        self.model.backbone.requires_grad_(False) 
        args.burn_in_steps = 0
        
        self.model.eval()
        
        lam = 10

        for i in range(2000):
            z_optimizer.zero_grad()    
            x = BinaryActivationSTE.apply(x_init)
            seeds_t = x.unsqueeze(-1).unsqueeze(0).unsqueeze(0)

            y_hat = self.model.backbone(seeds_t, edge_index, args)
                
            loss = -y_hat[0,-1,:,:].sum() + lam*((seeds_t.sum() - seed_num)**2)
            
            loss.backward()
            z_optimizer.step()
            
            with torch.no_grad():
                _, top_seeds_predict = torch.topk(x_init, seed_num, sorted=False)
                x0 = torch.zeros(N).to(args.device)
                x0[top_seeds_predict] = 1
                seeds_t_ = x0.unsqueeze(-1).unsqueeze(0).unsqueeze(0)

                y_ = self.model.backbone(seeds_t_, edge_index, args)
                    
            now_influence_num = y_[0,-1,:,:].sum()                
            print(now_influence_num)
            if now_influence_num >= max_influence_num:
                max_influence_num = now_influence_num
                max_x = x_init.clone().detach()
                
                if i % 1000 == 0 and i >0:            
                    
                    _, top_seeds_predict = torch.topk(max_x, seed_num, sorted=False)
                    list_pre = top_seeds_predict.tolist()
                    
                    time = 1000
                    
                    if mode=='sir':
                        beta = 0.1
                        gamma = 0.5
                        inf = graph_utils.workerMC_SIR([graph, list_pre, time, beta,gamma]).sum()
                    elif mode=='ic':
                        inf = graph_utils.computeMC_IC(graph,list_pre,time).sum()
                print('Iteration: {}'.format(i+1),
                    '\t Loss:{:.5f}'.format(loss.item()),
                    "\t Now_influence_num: {:.4f}".format(now_influence_num),
                    "\t Infection_mk: {:.9f}".format(inf/N),
                    "\t X_init_sum: {:.4f}".format(seeds_t.sum()),
                    "\t seed_num: {:.4f}".format(seed_num),
                    )

    def load_model(self):
        load_path = args.pwd+f'/result/{args.data_mode}/models/' + args.backbone + f'/{args.backbone}_train_net.mod'

        # 检查路径是否存在
        if os.path.exists(load_path):
            print(f"文件存在: {load_path}")
        else:
            print(f"文件不存在: {load_path}")
            return
        ckp = torch.load(load_path,  map_location='cpu', weights_only=False)
        self.model = ckp['model'].to(args.device)
        print(f'Model Loaded, Load Path:{load_path}')

if __name__ == '__main__':  
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    index = 11
    network_list = ['Celegans','deezer_europe','ca-GrQc','cora', 'ego-Facebook','fb-pages-food','wiki-Vote','soc-Slashdot0811','congress-twitter', 'soc-dolphins', 'soc-douban','cit-HepPh'][index:index+1]
    cascade_data_suffix_list = ['sir_beta0.1_gamma0.5','ic'][1:2]
    mode = args.propagation_data
    seed_rate = 0.05
    directed = True

    dataset_list = []
    
    for net in network_list:
        for suffix in cascade_data_suffix_list:
            dataset = net + '@' + suffix
            dataset_list.append(dataset)
    exp = Exp()
    exp.run(directed)