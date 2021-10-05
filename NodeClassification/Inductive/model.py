import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from gat_conv import GATConv
import pdb

class GATNet(torch.nn.Module):
    def __init__(self, train_dataset):
        super(GATNet, self).__init__()
        
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index, data_mask=None):
        
        x = F.elu(self.conv1(x, edge_index, edge_weight=data_mask) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index, edge_weight=data_mask) + self.lin2(x))
        x = self.conv3(x, edge_index, edge_weight=data_mask) + self.lin3(x)
        return x


class Masker(torch.nn.Module):
    def __init__(self, train_dataset, hidden=128):
        super(Masker, self).__init__()
        
        self.conv1 = GATConv(train_dataset.num_features, hidden, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * hidden)
        self.conv2 = GATConv(4 * hidden, hidden, heads=4)
        self.lin2 = torch.nn.Linear(4 * hidden, 4 * hidden)
        self.conv3 = GATConv(4 * hidden, hidden, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * hidden, hidden)
        self.mlp = torch.nn.Linear(hidden * 2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        link_score = self.concat_mlp_score(x, edge_index)
        return link_score

    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        # print("max:{:.2f} min:{:.2f} mean:{:.2f}".format(link_score.max(), link_score.min(), link_score.mean()))
        link_score = self.sigmoid(link_score).view(-1)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        # print("mean:{:.2f} min:{:.2f} max:{:.2f}".format(link_score.mean(), link_score.min(), link_score.max()))
        link_score = self.sigmoid(link_score).view(-1)
        return link_score


class MaskerGIN(torch.nn.Module):
    def __init__(self, train_dataset, hidden=128):
        super(MaskerGIN, self).__init__()
        hidden = 512
        self.conv1 = GINConv(
            Sequential(Linear(train_dataset.num_features, hidden), 
                       BatchNorm1d(hidden), 
                    #    ReLU(), 
                    #    Linear(hidden, hidden), 
                       ReLU()))
        self.lin1 = torch.nn.Linear(train_dataset.num_features, hidden)
        self.conv2 = GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                    #    ReLU(), 
                    #    Linear(hidden, hidden), 
                       ReLU()))
        self.lin2 = torch.nn.Linear(hidden, hidden)
        self.conv3 = GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                    #    ReLU(),
                    #    Linear(hidden, hidden),
                       ReLU()))
        self.lin3 = torch.nn.Linear(hidden, hidden)
        self.mlp = torch.nn.Linear(hidden * 2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        link_score = self.concat_mlp_score(x, edge_index)
        return link_score

    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        link_score = self.sigmoid(link_score).view(-1)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score).view(-1)
        return link_score


class MaskerGCN(torch.nn.Module):
    def __init__(self, train_dataset, hidden=128):
        super(MaskerGCN, self).__init__()
        hidden = 1024
        self.conv1 = GCNConv(train_dataset.num_features, hidden)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin3 = torch.nn.Linear(hidden, hidden)
        self.mlp = torch.nn.Linear(hidden * 2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        link_score = self.concat_mlp_score(x, edge_index)
        return link_score

    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        link_score = self.sigmoid(link_score).view(-1)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score).view(-1)
        return link_score


class MaskerMLP(torch.nn.Module):
    def __init__(self, train_dataset, hidden=128):
        super(MaskerMLP, self).__init__()
        hidden = 1024
        self.conv1 = Linear(train_dataset.num_features, hidden)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, hidden)
        self.conv2 = Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden)
        self.conv3 = Linear(hidden, hidden)
        self.lin3 = torch.nn.Linear(hidden, hidden)
        self.mlp = torch.nn.Linear(hidden * 2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        
        x = F.elu(self.conv1(x) + self.lin1(x))
        x = F.elu(self.conv2(x) + self.lin2(x))
        x = self.conv3(x) + self.lin3(x)
        link_score = self.concat_mlp_score(x, edge_index)
        return link_score

    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        link_score = self.sigmoid(link_score).view(-1)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score).view(-1)
        return link_score