import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import math


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, hid_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hid_features = hid_features
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features))
        self.lr_att = nn.Linear(in_features+hid_features, 1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.lr_att.reset_parameters()

    def forward(self, input, adj):
        input = torch.spmm(adj, input)
        input = torch.mm(input, self.weight)
        return input


class gcn_air(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hops, adj):
        super(gcn_air, self).__init__()
        self.fc = nn.Linear(nhid, nclass)
        self.lr_att = nn.Linear(nfeat+nhid, 1)

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid, nhid))
        for _ in range(num_hops-1):
            self.gcs.append(GraphConvolution(nhid, nhid, nhid))

        self.dropout = dropout
        self.adj = adj
        self.num_hops = num_hops

    def forward(self, features):
        h_list = []
        h_list.append(features)
        for i in range(self.num_hops):
            features = self.gcs[i](features, self.adj)
            features = F.relu(features)
            features = F.dropout(features, self.dropout,
                                 training=self.training)
            h_list.append(features)

        num_node = features.shape[0]
        attention_scores = [torch.sigmoid(self.lr_att(torch.cat(
            (x, h_list[0]), dim=1)).view(num_node, 1)) for x in h_list[1:]]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        output = torch.mul(h_list[1], W[:, 0].view(num_node, 1))
        for i in range(1, self.num_hops):
            output += torch.mul(h_list[i], W[:, i].view(num_node, 1))

        output = self.fc(output)
        return F.log_softmax(output, dim=1)


class appnp_air(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hops, alpha, adj):
        super(appnp_air, self).__init__()
        self.fc_1 = nn.Linear(nfeat, nhid)
        self.fc_2 = nn.Linear(nhid, nclass)
        self.lr_att = nn.Linear(nclass+nclass, 1)

        self.dropout = dropout
        self.adj = adj
        self.alpha = alpha
        self.num_hops = num_hops

    def forward(self, features):
        input = F.dropout(
            features, self.dropout, training=self.training)
        input = self.fc_1(input)
        input = F.relu(input)
        input = F.dropout(
            input, self.dropout, training=self.training)
        input = self.fc_2(input)

        h0 = input
        h_list = []
        h_list.append(input)
        for i in range(self.num_hops):
            input = self.alpha*torch.spmm(self.adj, input)+(1-self.alpha)*h0
            h_list.append(input)

        num_node = features.shape[0]
        attention_scores = []
        for i in range(len(h_list)):
            attention_scores.append(torch.sigmoid(self.lr_att(
                torch.cat([h_list[0], h_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        W = F.softmax(attention_scores, 1)
        output = torch.mul(h_list[0], W[:, 0].view(num_node, 1))
        for i in range(1, self.num_hops):
            output = output + torch.mul(h_list[i], W[:, i].view(num_node, 1))

        return F.log_softmax(output, dim=1)


class sgc_air(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hops):
        super(sgc_air, self).__init__()
        self.num_hops = num_hops

        self.fc_1 = nn.Linear((num_hops+1)*nfeat, nhid)
        self.fc_2 = nn.Linear(nhid, nclass)
        self.lr_att = nn.Linear(nfeat+nfeat, 1)

        self.fc_out = nn.Linear(nfeat, nclass)
        self.dropout = dropout

    def forward(self, feature_list):
        drop_features = [F.dropout(
            feature, self.dropout, training=self.training) for feature in feature_list]

        num_node = feature_list[0].shape[0]
        attention_scores = []
        for i in range(len(drop_features)):
            attention_scores.append(torch.sigmoid(self.lr_att(
                torch.cat([drop_features[0], drop_features[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        W = F.softmax(attention_scores, 1)
        output = torch.mul(drop_features[0], W[:, 0].view(num_node, 1))
        for i in range(1, self.num_hops):
            output = output + \
                torch.mul(drop_features[i], W[:, i].view(num_node, 1))

        output = self.fc_out(output)
        return F.log_softmax(output, dim=1)
