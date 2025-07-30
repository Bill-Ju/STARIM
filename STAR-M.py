import torch
from torch.optim import SGD
import torch.nn.init as init
import numpy as np
import networkx as nx
from model.mec_model import IC, SIR
from utils import graph_utils

class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Sigmoid + Threshold 0.5
        return (x > 0.9).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计：忽略非连续性，直接传梯度
        return grad_output
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = 'ic'
network_index = 4
seed_index = 0
seed_rates = [0.01,0.05,0.10,0.20,0.30]
seed_rate = seed_rates[seed_index]
NetWorks = ['deezer_europe','email-Enron','ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food','wiki-Vote','soc-dolphins','cit-HepPh','soc-douban']
direct = True
network = NetWorks[network_index]
print('Reading {} edgelist'.format(network))
network_edges_dir = './data/{}/{}.txt'.format(network,network)
G = None
if direct:
    with open(network_edges_dir, 'rb')as edges_f:
        G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph(), encoding='latin1',
                                    data=(('weight', float),))
else:
    with open(network_edges_dir, 'rb')as edges_f:
        G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                    data=(('weight', float),))
        
def infected_beta(graph):
    # SIR模型的感染率设置
    degree = nx.degree(graph)
    degree_list = []
    degree_sq_list = []
    for i in degree:
        degree_list.append(i[1])
        degree_sq_list.append(i[1] * i[1])
    degree_avg = np.mean(degree_list)
    degree_avg_sq = np.mean(degree_sq_list)
    infected = degree_avg / (degree_avg_sq -  2 *degree_avg)
    return infected
N = G.number_of_nodes()

gamma = 0.5 # 免疫率
beta = 0.1
steps = 10
seed_num = int(N* seed_rate)

if model == 'sir':
    sir = SIR(N,beta,gamma,50,steps,device).to(device)
else:
    ic = IC(N, steps,device).to(device)
    
edge_index = []
edge_weight = []
if direct:
    # 添加所有边
    for edge in G.edges():
        edge_index.append([edge[0], edge[1]])
        edge_weight.append(1/G.in_degree[edge[1]])
else:
    # 添加所有边
    for edge in G.edges():
        edge_index.append([edge[0], edge[1]])
        edge_weight.append(1/G.degree[edge[1]])
        
        edge_index.append([edge[1], edge[0]])  # 如果是无向图，添加反向边
        edge_weight.append(1/G.degree[edge[0]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
edge_weight = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(-1).to(device)


def get_influence(x_hat, seed_num, edge_index):
    _, top_seeds_predict = torch.topk(x_hat, seed_num, sorted=False)
    list_pre = top_seeds_predict.tolist()
    x0 = torch.zeros(N,1).to(device)
    x0[list_pre] = 1
    if model == 'sir':
        _,_,r = sir(edge_index, x0=x0)
    else:
        _,_,r = ic(edge_index,edge_weight, x0=x0)
    infection = r.squeeze().cpu()
    Influence_num = infection.sum()
    return Influence_num



x_init = torch.rand(N).to(device)

x_init.requires_grad = True
initial_momentum = 0.95
z_optimizer = SGD([x_init], lr=0.01, momentum=initial_momentum)
num_iteraion = 2000
max_x_ste = None
max_influence_num = 0
now_influence_num = 0
influence_num_list_ste = []
lam = 5

for i in range(num_iteraion):
    z_optimizer.zero_grad()

    x = BinaryActivationSTE.apply(x_init)

    seeds_t = x.unsqueeze(-1)

    if model == 'sir':
        _, _, y_hat = sir(edge_index, x0=seeds_t)
    else:
        _, _, y_hat = ic(edge_index,edge_weight, x0=seeds_t)

    loss = - y_hat.squeeze(-1).sum() + lam*((seeds_t.sum() - seed_num)**2)
    
    now_influence_num = get_influence(x_init, seed_num, edge_index)
    influence_num_list_ste.append(now_influence_num)
    if now_influence_num > max_influence_num:
        max_influence_num = now_influence_num
        max_x_ste = x_init.clone().detach()
    if (i+1) % 500 == 0:
        print('Iteration: {}'.format(i+1),
            '\t Loss:{:.5f}'.format(loss.item()),
            "\t Max_influence_num: {:.4f}".format(max_influence_num),
            "\t X_init_sum: {:.4f}".format(seeds_t.sum()),
            )
        
_, top_seeds_predict = torch.topk(max_x_ste, seed_num, sorted=False)
list_pre = top_seeds_predict.tolist()
graph, N = graph_utils.read_graph(network_edges_dir, ind=0, directed=direct)
time = 10000
if model == 'sir':
    num_ste = graph_utils.workerMC_SIR([graph, list_pre, time, beta, gamma]).sum()
else: 
    num_ste = graph_utils.computeMC_IC(graph,list_pre,time).sum()

print(f"Avg infected nodes: {num_ste/N}")
