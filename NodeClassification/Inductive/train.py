import torch
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score
import pdb
import pruning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_op = torch.nn.BCEWithLogitsLoss()

def train_model_and_masker(model, masker, optimizer, train_loader):

    model.train()
    masker.train()
    total_loss = 0
    mask_distribution = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        data_mask = masker(data.x, data.edge_index)
        mask_distribution.append(pruning.plot_mask(data_mask))
        out = model(data.x, data.edge_index, data_mask)
        loss = loss_op(out, data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    mask_distribution = torch.tensor(mask_distribution).mean(dim=0)
    return total_loss / len(train_loader.dataset), mask_distribution


@torch.no_grad()
def eval_acc_with_mask(model, masker, loader):
    model.eval()
    masker.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        data_mask = masker(data.x.to(device), data.edge_index.to(device))
        out = model(data.x.to(device), data.edge_index.to(device), data_mask)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def train(model, optimizer, train_loader):

    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0