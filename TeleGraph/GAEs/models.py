import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import SGConv
from torch_geometric.nn import GATConv


class GCN(torch.nn.Module):

    def __init__(self, input_feat, output_feat):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_feat, 2 * output_feat)
        self.conv2 = GCNConv(2 * output_feat, output_feat)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GAT(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SGC(torch.nn.Module):

    def __init__(self, input_feat, output_feat):
        super(SGC, self).__init__()
        self.conv = SGConv(input_feat, output_feat, K=2)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
