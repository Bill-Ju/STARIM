import os
import torch 
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from params import args

from utils.data_handler import MultiDataHandler,DataHandler

from torch_geometric.nn import MessagePassing

from torch.optim.optimizer import Optimizer
import numpy as np
from torch.distributions import Normal
from utils import graph_utils
import time
class Exp:
    def __init__(self, multi_data_handler):
        self.multi_data_handler = multi_data_handler
    
    def run(self, directed = False):
        
        self.load_model()
        self.im(args, directed)
        
    def im(self, args, directed): 
        handlers = self.multi_data_handler.handlers   
        for handler in handlers:
            graph, N = graph_utils.read_graph(handler.network_edges_path, ind=0, directed=directed)
            edge_index = handler.edge_index.t().contiguous().to(args.device)
            # train_loader = handler.test_data_loader
            N = edge_index.max()+1
            
            seed_num = int(N*seed_rate)
            y_true = torch.ones(N,1).to(args.device)
            # y_true[:,2] = 1
            x_init = torch.ones(N).to(args.device)
            # # x_init[:,1] = 1
            # x_norm = l1_projection(x_init, seed_num)
            # # x_norm = lp_projection_(x_init,1.2,seed_num)
            # x_init.copy_(x_norm)
            x_init.requires_grad = True

            initial_momentum = 0.95
            z_optimizer = SGD([x_init], lr=0.02, momentum=initial_momentum)
            # z_optimizer = SGLDWithMomentum([x_init],lr=0.01,noise=0.005, momentum=initial_momentum, addnoise=True)
            max_x = None
            max_influence_num = 0
            now_influence_num = 0
            self.model.backbone.requires_grad_(False) 
            # if args.meta == 'Any':
            #     self.model.backbone.requires_grad_(False)
            #     if args.backbone == 'AdapterRNN':
            #         self.model.prompts.requires_grad_(False)
            #     else:
            #         self.model.memory.requires_grad_(False)
            #         self.model.prompt_t.requires_grad_(False)
            # self.model.backbone.requires_grad_(False) 
            args.burn_in_steps = 0
            class BinaryActivationSTE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # Sigmoid + Threshold 0.5
                    return (x > 0.95).float()
                
                @staticmethod
                def backward(ctx, grad_output):
                    # 直通估计：忽略非连续性，直接传梯度
                    return grad_output
            self.model.eval()
            
            inf1 = 0
            begin = time.time()
            for i in range(2000):
                z_optimizer.zero_grad()    
                # seeds_t = x_init.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
                # x = BinaryActivationSTE.apply(torch.sigmoid(x_init))
                x = BinaryActivationSTE.apply(x_init)
                # # x = torch.zeros_like(x)
                seeds_t = x.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
                # _, _, y_hat = sir(edge_index, x0=seeds_t)
                
                # infection_i = torch.sum(inflection * seeds_t, dim = 1)
                if args.meta == 'Any':
                    if args.backbone == 'AdapterRNN':
                        feat_for_prompt = self.model.prompt[0].unsqueeze(0)
                        context = F.leaky_relu(self.model.prompt_fc[0](feat_for_prompt))
                        W_0, B_0, W_1,  B_1= self.model.memory(context)
                        # W_0, W_1, B_0, B_1 = self.model.prompts(-1)
                        y_hat = self.model.backbone(seeds_t, edge_index,W_0, B_0, W_1,  B_1, args)
                    else:
                        feat_for_prompt = self.model.prompt_t.unsqueeze(0)

                        attention_weights = feat_for_prompt
                        W_beta, B_beta, W_gamma, B_gamma= self.model.memory(attention_weights)
              
                        y_hat = self.model.backbone(seeds_t, edge_index,W_beta, B_beta, W_gamma, B_gamma, args)
                        # with torch.no_grad():
                        #     y_ = self.model.backbone(seeds_t_, edge_index,W_beta, B_beta, W_gamma, B_gamma, args)
                        # print(seeds_t.sum())
                    # y_hat = self.model.backbone(alpha_optimized.squeeze(0), beta_optimized.squeeze(0), input, edge_index, args)
                    # y_hat = self.model.backbone(self.model.alpha_t, self.model.beta_t, input, edge_index, args)
                elif args.meta == 'Maml':
                    y_hat = self.model.backbone(input, edge_index, args)
                else:
                    y_hat = self.model.backbone(seeds_t, edge_index, args)
                    
                # loss = F.mse_loss(y_hat[0,-1,:,:], y_true) + 10*torch.abs(seeds_t.mean() - seed_rate)
                loss = -y_hat[0,-1,:,:].mean() + 20*torch.abs(seeds_t.mean() - seed_rate)
                
                loss.backward()
                z_optimizer.step()
                
                # with torch.no_grad():
                #     _, top_seeds_predict = torch.topk(x_init, seed_num, sorted=False)
                #     x0 = torch.zeros(N).to(args.device)
                #     x0[top_seeds_predict] = 1
                #     seeds_t_ = x0.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
                #     # print(seeds_t_.sum())
                #     if args.meta == 'Any':
                #         if args.backbone == 'AdapterRNN':
                #             feat_for_prompt = self.model.prompt[0].unsqueeze(0)
                #             context = F.leaky_relu(self.model.prompt_fc[0](feat_for_prompt))
                #             W_0, B_0, W_1,  B_1= self.model.memory(context)
                #             # W_0, W_1, B_0, B_1 = self.model.prompts(-1)
                #             y_hat = self.model.backbone(seeds_t_, edge_index,W_0, B_0, W_1,  B_1, args)
                #         else:
                #             feat_for_prompt = self.model.prompt_t.unsqueeze(0)

                #             attention_weights = feat_for_prompt
                #             W_beta, B_beta, W_gamma, B_gamma= self.model.memory(attention_weights)
                #             y_ = self.model.backbone(seeds_t_, edge_index,W_beta, B_beta, W_gamma, B_gamma, args)
                #             # print(seeds_t.sum())
                #         # y_hat = self.model.backbone(alpha_optimized.squeeze(0), beta_optimized.squeeze(0), input, edge_index, args)
                #         # y_hat = self.model.backbone(self.model.alpha_t, self.model.beta_t, input, edge_index, args)
                #     elif args.meta == 'Maml':
                #         y_hat = self.model.backbone(seeds_t_, edge_index, args)
                #     else:
                #         y_ = self.model.backbone(seeds_t_, edge_index, args)
                        
                # now_influence_num = y_[0,-1,:,:].sum()
                # with torch.no_grad():
                #     x_init.clamp_(min=0.0, max=1.0)
                #     x_init.copy_(l1_projection(x_init, seed_num))
                
                # print(now_influence_num)
                # if now_influence_num >= max_influence_num:
                #     max_influence_num = now_influence_num
                #     max_x = x_init.clone().detach()
                
                # if i % 1000 == 0 and i >0:            
                    
                #     _, top_seeds_predict = torch.topk(max_x, seed_num, sorted=False)
                #     list_pre = top_seeds_predict.tolist()
                    
                #     time = 1000
                    
                #     if mode=='sir':
                #         beta = 0.1
                #         gamma = 0.5
                #         inf1 = graph_utils_.workerMC_SIR([graph, list_pre, time, beta,gamma]).sum()
                #     elif mode=='ic':
                #         inf1 = graph_utils_.computeMC_IC(graph,list_pre,time).sum()
                #     print(inf1/N)
                #     print(list_pre)
                # print(inf1/N)
                # print('Iteration: {}'.format(i+1),
                #     '\t Loss:{:.5f}'.format(loss.item()),
                #     "\t Now_influence_num: {:.4f}".format(now_influence_num),
                #     "\t Infection_mk: {:.9f}".format(inf1/N),
                #     "\t X_init_sum: {:.4f}".format(seeds_t.sum()),
                #     "\t seed_num: {:.4f}".format(seed_num),
                #     )
            end = time.time()
            all_time = end - begin
            print(f"all_time:{all_time}")
            
    def load_model(self):
        # network_list = ['cora', 'ego-Facebook', 'wiki-Vote', 'congress-twitter','soc-dolphins','bitcoin-alpha','bitcoin-otc']
        network_list = ['ca-GrQc','Celegans','cora','fb-pages-food','Router','USA airports','Yeast'][:2]
        # network_list = ['ca-GrQc','Celegans','cora','fb-pages-food','Router','USA airports','Yeast','wiki-Vote','congress-twitter', 'soc-dolphins']
        # cascade_data_suffix_list = ['sir_beta0.06256902022729498_gamma0.5','sir_beta0.10821022969857398_gamma0.5','sir_beta0.009562987075975317_gamma0.5','sir_beta0.055542343770643404_gamma0.5', 'sir_beta0.006973839310519237_gamma0.5']

        # cascade_data_suffix_list_ = cascade_data_suffix_list[:1]
        cass = '_'.join(cascade_data_suffix_list)
        network_list_ = network_list
        nets = '_'.join(network_list_)

        # load_path = args.pwd+f'/result/{args.meta}/{args.data_mode}/models/' + args.backbone + f'/{args.backbone}_train_net_{nets}_2_19.mod'
        # if args.feat_for_prompt:
        #     load_path = args.pwd+f'/result/{args.meta}/{args.data_mode}/models/' + args.backbone + f'/{args.backbone}_tuning_net_{nets}_{cass}_ffp{args.feat_for_prompt}_feat_rand{args.feat_rand}_train_size{args.train_size_for_few_shot}.mod'
        # else:
        #     load_path = args.pwd+f'/result/{args.meta}/{args.data_mode}/models/' + args.backbone + f'/{args.backbone}_tuning_net_{nets}_{cass}_train_size{args.train_size_for_few_shot}.mod'

        # load_path = args.pwd+f'/result/{args.meta}/{args.data_mode}/models/' + args.backbone + f'/{args.backbone}_train_net_{nets}_{cass}.mod'
        load_path = args.pwd+f'/result/{args.meta}/{args.data_mode}/models/' + args.backbone + f'/{args.backbone}_train_net_{nets}_{cass}_7_24.mod'

        
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
    datasets = dict()
    datasets['networks'] = ['ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food','Lastfm_asia','Router','Soc-hamsterster', 'USA airports','Yeast','wiki-Vote','p2p-Gnutella05','p2p-Gnutella06','soc-Slashdot0811']
    datasets['undirect_net'] = [
        'ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food','Lastfm_asia','Router','Soc-hamsterster', 'USA airports','Yeast'
    ]
    datasets['direct_net'] = [
        'wiki-Vote','p2p-Gnutella05','p2p-Gnutella06','soc-Slashdot0811'
    ]
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    args.mode = 'tuning'
    
    # network_list = ['ca-GrQc']
    # # cascade_data_suffix_list = ['sir_beta0.01_gamma0.5', 'sir_beta0.02_gamma0.5', 'sir_beta0.05_gamma0.5', 'sir_beta0.08_gamma0.5', 'sir_beta0.12_gamma0.5', 'sir_beta0.16_gamma0.5', 'sir_beta0.2_gamma0.5']
    # cascade_data_suffix_list = ['ic']
    index = 11
    network_list = ['Celegans','deezer_europe','ca-GrQc','cora', 'ego-Facebook','fb-pages-food','wiki-Vote','soc-Slashdot0811','congress-twitter', 'soc-dolphins', 'soc-douban','cit-HepPh'][index:index+1]
    # cascade_data_suffix_list = ['sir_beta0.06256902022729498_gamma0.5','sir_beta0.10821022969857398_gamma0.5','sir_beta0.009562987075975317_gamma0.5','sir_beta0.055542343770643404_gamma0.5', 'sir_beta0.006973839310519237_gamma0.5'][index:index+1]
    cascade_data_suffix_list = ['sir_beta0.1_gamma0.5','ic'][1:2]
    mode = 'ic'
    seed_rate = 0.05
    directed = True
    # cascade_data_suffix_list = ['lt_threshold0.1', 'lt_threshold0.2', 'lt_threshold0.3', 'lt_threshold0.4', 'lt_threshold0.5', 'lt_threshold0.6', 'lt_threshold0.7']
    args.direct = directed
    dataset_list = []
    
    for net in network_list:
        for suffix in cascade_data_suffix_list:
            dataset = net + '@' + suffix
            dataset_list.append(dataset)
    handler = MultiDataHandler(dataset_list, args)

    exp = Exp(handler)
    exp.run(directed)