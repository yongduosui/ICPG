import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
import pdb
import math



### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr, data_mask=None):
        # x: torch.Size([15117, 300])
        # edge_index: torch.Size([41910, 2])
        # edge_embedding: torch.Size([41910, 300])
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)
        row, col = edge_index
        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if data_mask is not None:
            norm = deg_inv_sqrt[row] * data_mask * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, node_encoder, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, data_mask=None):
        # Batch(batch=[15117], 
                # edge_attr=[41910, 2], 
                # edge_index=[2, 41910], 
                # node_depth=[15117, 1], 
                # node_dfs_order=[15117, 1], 
                # node_is_attributed=[15117, 1], 
                # ptr=[129], x=[15117, 2], y=[128], y_arr=[128, 5])
        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth, batched_data.batch
        ### computing input node embedding
        
        h_list = [self.node_encoder(x, node_depth.view(-1,))] # [torch.Size([15117, 300])]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr, data_mask=data_mask)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation




if __name__ == "__main__":
    pass
