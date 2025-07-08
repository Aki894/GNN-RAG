import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv,GraphConv,SAGEConv   
import math

class PSNRModule(nn.Module):
    def __init__(self, nodes_num, n_hid, args):
        super().__init__()
        self.nodes_num =  nodes_num
        self.coef_encoder_type = args.coef_encoder

        if args.coef_encoder == 'gat':
            self.coef_encoder = GATConv(n_hid, 2, num_heads = 1)
        if args.coef_encoder == 'sage':
            self.coef_encoder = SAGEConv(n_hid, 2, 'gcn')
        if args.coef_encoder == 'gcn':
            self.coef_encoder = GraphConv(n_hid, 2)
        if args.coef_encoder == 'mlp':
            self.coef_encoder = nn.Sequential(
                nn.Linear(n_hid, n_hid),
                nn.ReLU(),
                nn.Linear(n_hid, 2)
            ) 

        self.d_model = n_hid
        self.max_seq_len = args.n_layers +  3

        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float()
                             * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe
        self.pe_coff = nn.Parameter(torch.tensor(0.1))
        self.layer_emb = args.layer_emb

    def forward(self, graph, input, t):

        # for inductive 
        self.nodes_num = input.shape[0]

        x  = torch.randn(self.nodes_num,1)
        y = torch.ones(self.nodes_num,1)
        x = x.to(graph.device)
        y = y.to(graph.device)
        
        if self.layer_emb:
            input = input + self.pe_coff*self.pe[t+1].to(input.device)

        if self.coef_encoder_type == 'mlp':
            coef = self.coef_encoder(input)
        else:
            coef = self.coef_encoder(graph, input)
            if self.coef_encoder_type == 'gat':
                coef = torch.mean(coef, dim=1)

        std = F.relu(coef[:,0])
        mean = F.relu(coef[:,1])
        std = std.view(-1,1)
        mean = mean.view(-1,1)

        return input * (F.sigmoid(x * std + y*mean)) 