import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from model.NRI import RNN_decoder

class Diffuion(nn.Module):
    def __init__(self, args):
        super(Diffuion, self).__init__()
        self.device = args.device
        self.in_channels = args.in_channels
        if args.backbone == 'NRI':
            self.backbone = RNN_decoder.RNNDecoder(args)
        else:
            print("no such backbone")

    def forward(self, batch, args):
        loss_sum = 0
        for index in range(batch.edge_index.shape[0]):
            mask = batch.batch == index
            cas = torch.tensor(batch.cas[index]).to(torch.float32).to(batch.edge_index.device)
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            edge_index_local = batch.edge_index[:, edge_mask]  # 获取有效边
            first_true_idx = mask.nonzero(as_tuple=True)[0][0].item()  # 获取第一个有效节点的全局索引
            node_offset = first_true_idx  # 当前图的局部节点索引从第一个有效节点的全局索引开始
            # 重新调整边的索引
            edge = edge_index_local - node_offset 
            input = cas[0:-1].unsqueeze(0)
            target = cas[1:].unsqueeze(0)
            preds = self.backbone(input, edge, args)
            loss = self.compute_loss(preds, target, args,'mse')
            loss_sum += loss
        return loss_sum
    
    def compute_loss(self, preds, target, args,  loss_type='mse', if_reg=False):

        if loss_type=='bce':    
            loss =  F.binary_cross_entropy(preds,target) 
        elif loss_type=='l1':
            loss = F.l1_loss(preds,target) 
        else:
            loss = F.mse_loss(preds,target)
        return loss 