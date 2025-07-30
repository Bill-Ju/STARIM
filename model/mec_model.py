from torch_geometric.nn import MessagePassing
from torch import nn 
import torch.nn.functional as F
import torch

class IC(nn.Module):
    def __init__(self, N, steps, device,  C=50):
        super(IC, self).__init__()
        self.steps = steps

        self.N = N
        self.C = C
        self.device = device
        self.gamma = 1.0
        
    def init_state(self,x0=None):
        if x0 is None:
            x = torch.zeros(self.N,1).to(self.device)
            ind = torch.argsort(self.w.view(-1),descending=True)
            x[ind[:self.C]] = 1    
        else:
            x = x0    
        r = torch.zeros(self.N,1).to(self.device)
        s = 1 - x
        return s, x ,r
    
    def single_step_forward(self, edge_index, edge_weight, s, x, r):
        # 进行加权聚合
        row, col = edge_index
        # ww = (torch.ones(edge_weight.shape)*self.beta).to(self.device)
        epsilon = 1e-15
        agg_messages = torch.log(1- (edge_weight * x[row])+ epsilon)  # 加权消息传递 (num_edges, hidden_dim)
        agg_messages = torch.zeros_like(x).scatter_add_(0, col.unsqueeze(-1), agg_messages)
        q = torch.exp(agg_messages)
        s, x, r = s*q, s*(1-q) , r + x
        return s, x, r
    
    def forward(self, edge_index, edge_weight, x0=None):
        s, x, r = self.init_state(x0)
        for i in range(self.steps): 
           s, x ,r = self.single_step_forward(edge_index,edge_weight,s,x,r)
        return s, x, r
    def message(self, x_j):
        return x_j

class SIR(MessagePassing):
    def __init__(self, N, beta, gamma, C ,steps, device):
        super().__init__(aggr="add")
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.C = C
        self.steps = steps
        self.device = device

    def init_state(self,x0=None):
        if x0 is None:
            x = torch.zeros(self.N,1).to(self.device)
            ind = torch.argsort(self.w.view(-1),descending=True)
            x[ind[:self.C]] = 1    
        else:
            x = x0    
        r = torch.zeros(self.N,1).to(self.device)
        s = 1 - x
        return s, x ,r
    
    def single_step_forward(self, edge_index, s, x, r):
        q = self.propagate(edge_index, x=torch.log(1-self.beta*x))
        q = torch.exp(q)
        # q = self.propagate(edge_index, x=self.beta*x)
        # q = 1-q
        s, x, r = s*q, (1-self.gamma)*x + s*(1-q) , r + self.gamma*x
        return s, x, r

    def forward(self, edge_index, x0=None):
        s, x, r = self.init_state(x0)
        for i in range(self.steps): 
           s, x ,r = self.single_step_forward(edge_index,s,x,r)
        return s, x, r

    def message(self, x_j):
        return x_j
    


