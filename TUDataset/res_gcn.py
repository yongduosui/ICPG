from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import pdb

class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(ResGCN, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = dataset.num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, dataset.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, data_mask=None):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index, edge_weight=data_mask))
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

        
        
class GCNmasker(torch.nn.Module):
    """GCN masker: a dynamic trainable masker"""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 score_function='inner_product', 
                 mask_type='GCN',
                 edge_norm=True):
        super(GCNmasker, self).__init__()
        
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = dataset.num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.scale = 1
        self.mask_type = mask_type
        if mask_type == "GCN":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GConv(hidden, hidden))
        elif mask_type == "GIN":
            for i in range(num_conv_layers):
                self.convs.append(GINConv(
                Sequential(Linear(hidden, hidden), 
                        BatchNorm1d(hidden), 
                        ReLU(),
                        Linear(hidden, hidden), 
                        ReLU())))
        elif mask_type == "GAT":
            head = 4
            dropout = 0.2
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        elif mask_type == "MLP":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(Linear(hidden, hidden))
        else:
            assert False
        
        self.sigmoid = torch.nn.Sigmoid()
        self.score_function = score_function
        self.mlp = nn.Linear(hidden * 2, 1)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, score_type='inner_product'):
        
        x, edge_index = data.x, data.edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        if self.mask_type in ["GCN", "GAT"]:
            for i, conv in enumerate(self.convs):
                x = self.bns_conv[i](x)
                x = conv(x, edge_index)
                if i == 2: break
                x = F.relu(x)
        elif self.mask_type == "GIN":
            for i, conv in enumerate(self.convs):
                x= conv(x, edge_index)
                if i == 2: break
                x = F.relu(x)
        else:
            for i, conv in enumerate(self.convs):
                x = self.bns_conv[i](x)
                x = conv(x)
                if i == 2: break
                x = F.relu(x)
        
        if self.score_function == 'inner_product':
            link_score = self.inner_product_score(x, edge_index)
        elif self.score_function == 'concat_mlp':
            link_score = self.concat_mlp_score(x, edge_index)
        else:
            assert False

        return link_score
    
    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        link_score = self.sigmoid(link_score)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score).view(-1)
        
        return link_score


class GINNet(torch.nn.Module):
    def __init__(self, dataset, 
                       hidden, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = dataset.num_features
        hidden_out = dataset.num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x= conv(x, edge_index)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)




class GATNet(torch.nn.Module):
    def __init__(self, dataset, 
                       hidden,
                       head=4,
                       num_fc_layers=1, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        
        hidden_in = dataset.num_features
        hidden_out = dataset.num_classes

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, data_mask=None):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

        
# class GATNet(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GATNet, self).__init__()

#         self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
#         self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
#         self.conv3 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
#         self.conv4 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)


#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=-1)