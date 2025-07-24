import torch
import torch.nn.functional as F
import torch.nn as nn

class RNNDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args):
        super(RNNDecoder, self).__init__()


        in_channels = args.in_channels
        hidden_channels = args.hidden_channels
        
        # self.skip_first_edge_type = args.skip_first_edge_type
        self.dropout_prob = args.dropout
        self.pred_steps = args.Tstep

        self.msg_out_shape = hidden_channels
 
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * hidden_channels, hidden_channels) for _ in range(3)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(3)]
        )

        self.hidden_r = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.hidden_i = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.hidden_h = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.input_r = nn.Linear(in_channels, hidden_channels, bias=True)
        self.input_i = nn.Linear(in_channels, hidden_channels, bias=True)
        self.input_n = nn.Linear(in_channels, hidden_channels, bias=True)

        self.out_fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc3 = nn.Linear(hidden_channels, in_channels)


    def mul(self, x, w):
        # (batch, edges, channels), (edges) -> (batch, edges, channels)
        return torch.einsum("bec,e->bec", x, w)

    def single_step_forward(self, x, edge_index, hidden):

        # node2edge
        row, col = edge_index

        # data has shape [batch, nodes, variables]
        pre_msg = torch.cat((hidden[:,row,:],hidden[:,col,:]),dim=-1)

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        
        for i in range(0, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = torch.tanh(self.msg_fc2[i](msg))
            all_msgs = msg if i==0 else all_msgs + msg     
            
        # 聚合邻居信息
        agg_msgs = torch.zeros(x.shape[0], x.shape[1], all_msgs.shape[-1]).to(x.device)
        index = col.unsqueeze(0).unsqueeze(-1).expand(agg_msgs.shape[0], -1, all_msgs.shape[-1])
        agg_msgs = agg_msgs.scatter_add_(1, index, all_msgs)
        
        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(x) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(x) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(x) + r * self.hidden_h(agg_msgs))
        
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob,training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = F.relu(self.out_fc3(pred))
        pred = x + pred
        return pred, hidden

    def forward(self, inputs, edge_index, args):

        # input data has shape [batch, nodes, variables, time]
        # inputs = inputs.permute(0,3,1,2).contiguous()
        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)

        hidden = hidden.to(inputs.device)

        pred_all = []
        if args.burn_in_steps == 0:
            steps = args.Tstep
        else:
            steps = inputs.shape[1]

        for step in range(0, steps):

            if step < args.burn_in_steps or step==0:
                ins = inputs[:, step, :, :]
            else:
                ins = pred_all[step - 1]

            pred, hidden = self.single_step_forward(
                ins, edge_index, hidden
            )
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.contiguous()