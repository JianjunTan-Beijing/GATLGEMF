#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import torch_geometric.nn as gnn
from torch.autograd import Variable


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32,32,32],num_class=2):
        super(Net, self).__init__()
        conv = gnn.GATConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv1 = GCNConv((input_dim),latent_dim[0] )
        self.conv_params.append(conv(latent_dim[0],latent_dim[1],heads=8,dropout=0.2))
        self.conv_params.append(conv(latent_dim[1]*8, latent_dim[2],heads=8,dropout=0.2))
        # self.conv_params.append(conv(latent_dim[2]*8,latent_dim[3] , heads=8, dropout=0.2))
        # self.conv_params.append(conv(latent_dim[3] * 1, latent_dim[4], heads=1, dropout=0.2))

        # self.linear1 = nn.Linear((latent_dim[1] + latent_dim[2] )*4 , hidden_size) #+ latent_dim[3]*4
        self.linear1 = nn.Linear(latent_dim[0]+ (latent_dim[1]+ latent_dim[2])*8, hidden_size)
        # self.linear1 = nn.Linear(latent_dim[1], hidden_size)
        # self.linear1 = nn.Linear((latent_dim[1] + latent_dim[2]+ latent_dim[3]) * 1 + latent_dim[4], hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_class)



    def forward(self, data):
        data.to(torch.device("cuda:0"))
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        cat_message_layers = []
        x = F.elu(self.conv1(x, edge_index))
        cat_message_layers.append(x)
        for conv in self.conv_params:
            x = F.elu(conv(x, edge_index))
            cat_message_layers.append(x)

        cur_message_layer = torch.cat(cat_message_layers, 1)


        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch==i).nonzero()[0].cpu().numpy()[0])

        cur_message_layer = cur_message_layer[idx,:]

        hidden = self.linear1(cur_message_layer)
        self.feature = hidden
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=0.5,training=self.training)

        logits = self.linear2(hidden)
        prob = F.softmax(logits, dim=1).detach()
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y.long())

            pred = logits.data.max(1, keepdim=True)[1]
            return logits, loss, pred,prob, self.feature
        else:
            return logits



