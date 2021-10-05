import random
import torch
import torch.nn.utils.prune as prune
import copy
import gcn_conv
import pdb

def see_zero_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, gcn_conv.GCNConv):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))     
    print('INFO: Remain Weight [{:.4f}%]  '.format(100 * (1 - zero_sum / sum_list)))
    return zero_sum / float(sum_list)


def random_pruning_data(data, args):

    if args.pruning_adj == 0:
        return data
    
    num_edges = data.num_edges
    drop_edge_num = int(num_edges * args.pruning_adj)
    remain_index = random.sample([i for i in range(num_edges)], num_edges - drop_edge_num)
    data.edge_index = data.edge_index[:, remain_index]
    
    return data

def plot_mask(data_mask):

    a = (data_mask <= 0.2).sum()
    b = (data_mask <= 0.4).sum()
    c = (data_mask <= 0.6).sum()
    d = (data_mask <= 0.8).sum()
    e = (data_mask <= 1.0).sum()
    a, b, c, d, e = float(a), float(b), float(c), float(d), float(e)

    a1 = a / e         # (0.0 - 0.2)
    b1 = (b - a) / e   # (0.2 - 0.4)
    c1 = (c - b) / e   # (0.4 - 0.6)
    d1 = (d - c) / e   # (0.6 - 0.8)
    e1 = (e - d) / e   # (0.8 - 1.0)

    return [a1, b1, c1, d1, e1]

def pruning_data(data_orig, data_mask, args):

    data = copy.deepcopy(data_orig)
    num_edges = data.num_edges
    prune_num_edges = int(num_edges * args.pruning_adj)
    _, index = torch.sort(data_mask)
    remain_index = index[prune_num_edges:]
    data.edge_index = data.edge_index[:, remain_index]
    return data

def pruning_data_by_masker(masker, data, args):

    device = torch.device('cuda')
    data = data.to(device)
    data_mask = masker(data)
    num_edges = data.num_edges
    prune_num_edges = int(num_edges * args.pruning_adj)
    _, index = torch.sort(data_mask)
    remain_index = index[prune_num_edges:]
    data.edge_index = data.edge_index[:, remain_index]
    return data



def extract_mask(model):

    model_dict = model.state_dict()
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]
    return new_dict



def pruning_model(model, px, random=False):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, gcn_conv.GCNConv):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    if random:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )
    else:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )


def pruning_model_by_mask(model, mask_dict):

    module_to_prune = []
    mask_to_prune = []
    
    module_to_prune.append(model.conv1)
    mask_to_prune.append(mask_dict['conv1.weight_mask'])
    module_to_prune.append(model.conv2)
    mask_to_prune.append(mask_dict['conv2.weight_mask'])
    
    
    for ii in range(len(module_to_prune)):
        prune.CustomFromMask.apply(module_to_prune[ii], 'weight', mask=mask_to_prune[ii])