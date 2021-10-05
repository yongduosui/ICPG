import torch.nn.functional as F
import torch
import pruning
import pdb
import copy
from torch_geometric.data import DataLoader

def eval_acc_with_pruned_dataset(model, masker, dataset, device, args):

    if args.with_eval_mode:
        model.eval()
        masker.eval()
    dataset_pru = copy.deepcopy(dataset)
    dataset_pru = pruning.masker_pruning_dataset(dataset_pru, masker, args)
    loader_ori = DataLoader(dataset, args.batch_size, shuffle=False)
    loader_pru = DataLoader(dataset_pru, args.batch_size, shuffle=False)
    sp = pruning.print_pruning_percent(loader_ori, loader_pru)

    correct = 0
    for data in loader_pru:
        data = data.to(device)
        with torch.no_grad():
            
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader_pru.dataset), sp


def eval_acc_with_mask(model, masker, dataset, device, args, binary=False):
    if args.with_eval_mode:
        model.eval()
        masker.eval()

    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            data_mask = masker(data)
            if binary:
                data_mask = pruning.binary_mask(data_mask, args.pruning_percent)
            
            pred = model(data, data_mask).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def train_model_and_masker(model, optimizer, loader, device, args, masker):
    
    model.train()
    masker.train()
    total_loss = 0
    correct = 0
    mask_distribution = []
    for data in loader: 
        
        optimizer.zero_grad()
        data = data.to(device) # 128 graphs 
        data_mask = masker(data)
        
        mask_distribution.append(pruning.plot_mask(data_mask))
        
        out = model(data, data_mask)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    
    mask_distribution = torch.tensor(mask_distribution).mean(dim=0)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), mask_distribution



def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
        
def train(model, optimizer, loader, device, show=False):
    
    model.train()
    total_loss = 0
    correct = 0
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        if show:
            if it % 20 == 0:
                print("Iter:[{}/{}] Loss:[{:.4f}]".format(it, len(loader), loss))
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):

    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)