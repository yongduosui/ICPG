from gcn_conv import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GCNNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNNet, self).__init__()
        
        self.conv1 = GCNConv(dataset.num_features, args.dim, cached=False)
        self.conv2 = GCNConv(args.dim, dataset.num_classes, cached=False)

    def forward(self, data, data_mask=None):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=data_mask))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=data_mask)
        return F.log_softmax(x, dim=1)



class GCNmasker(torch.nn.Module):
    
    def __init__(self, dataset, args):
        super(GCNmasker, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, args.masker_dim, cached=False)
        self.conv2 = GCNConv(args.masker_dim, args.masker_dim, cached=False)
        self.mlp = nn.Linear(args.masker_dim * 2, 1)
        self.score_function = args.score_function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        
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
        #print("max:{:.2f} min:{:.2f} mean:{:.2f}".format(link_score.max(), link_score.min(), link_score.mean()))
        link_score = self.sigmoid(link_score).view(-1)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        # weight = self.mlp.weight
        # print("max:{:.2f} min:{:.2f} mean:{:.2f}".format(link_score.max(), link_score.min(), link_score.mean()))
        link_score = self.sigmoid(link_score).view(-1)
        return link_score