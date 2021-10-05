import torch
import torch.nn as nn
import numpy as np
import random
import pdb
import os
from tqdm import tqdm
from torch_geometric.data import Batch
import torch.nn.utils.prune as prune
import gcn_conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prRed(skk): print("\033[91m{}\033[00m".format(skk))
def prGreen(skk): print("\033[92m{}\033[00m".format(skk))
def prYellow(skk): print("\033[93m{}\033[00m".format(skk))


def see_zero_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, gcn_conv.GCNConv) or isinstance(m, nn.Linear):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))     
    print('INFO: Weight Sparsity [{:.4f}%] '.format(100 * (zero_sum / sum_list)))
    return zero_sum / sum_list

def extract_mask(model):

    model_dict = model.state_dict()
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]
    return new_dict

def pruning_model_by_mask(model, mask_dict):

    module_to_prune = []
    mask_to_prune = []
    
    module_to_prune.append(model.conv_feat)
    mask_to_prune.append(mask_dict['conv_feat.weight_mask'])

    module_to_prune.append(model.convs[0])
    mask_to_prune.append(mask_dict['convs.0.weight_mask'])

    module_to_prune.append(model.convs[1])
    mask_to_prune.append(mask_dict['convs.1.weight_mask'])

    module_to_prune.append(model.convs[2])
    mask_to_prune.append(mask_dict['convs.2.weight_mask'])

    module_to_prune.append(model.lins[0])
    mask_to_prune.append(mask_dict['lins.0.weight_mask'])

    module_to_prune.append(model.lin_class)
    mask_to_prune.append(mask_dict['lin_class.weight_mask'])

    for ii in range(len(module_to_prune)):
        prune.CustomFromMask.apply(module_to_prune[ii], 'weight', mask=mask_to_prune[ii])


def pruning_model(model, px, random=False):

    if px == 0:
        pass
    else:
        parameters_to_prune =[]
        for m in model.modules():
            if isinstance(m, gcn_conv.GCNConv) or isinstance(m, nn.Linear):
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


def grad_model(model, fix=True):
    
    for name, param in model.named_parameters():
        param.requires_grad = fix
        #print("NAME:{} GRAD:{}".format(name, param.requires_grad))

def pruning_batch_data_from_mask(data_list, data_mask, args):

    offset = 0
    for data in data_list:
        
        num_edges = data.num_edges
        edge_score = data_mask[offset:offset + num_edges]
        prune_num_edges = int(num_edges * args.pruning_percent)
        _, index = torch.sort(edge_score)
        remain_index = index[prune_num_edges:]
        data.edge_index = data.edge_index[:, remain_index]
        offset += num_edges


def masker_pruning_dataset(dataset, masker, args):

    data_list = []
    offset = 0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data_list.append(data)
            
            if (i + 1) % args.batch_size == 0:
                batch_data = Batch.from_data_list(data_list[offset:offset + args.batch_size]).to(device)
                data_mask = masker(batch_data)
                pruning_batch_data_from_mask(data_list[offset:offset + args.batch_size], data_mask, args)
                offset += args.batch_size

        batch_data = Batch.from_data_list(data_list[offset:]).to(device)
        data_mask = masker(batch_data)
        pruning_batch_data_from_mask(data_list[offset:], data_mask, args)
    return data_list


def random_pruning_dataset(dataset, args):
    if args.pruning_percent == 0:
        return dataset

    data_list = []
    for i, data in enumerate(dataset):

        num_edges = data.num_edges
        drop_edge_num = int(num_edges * args.pruning_percent)
        remain_index = random.sample([i for i in range(num_edges)], num_edges - drop_edge_num)
        data.edge_index = data.edge_index[:, remain_index]
        data_list.append(data)

    return data_list


def print_pruning_percent(dataset_ori, dataset_pru):

    ori_all = 0.0
    pru_all = 0.0
    
    for data_ori, data_pru in zip(dataset_ori, dataset_pru):
        ori = data_ori.num_edges
        pru = data_pru.num_edges
        ori_all += ori
        pru_all += pru
    
    sp = 1 - pru_all / ori_all
    # print('INFO: Dataset Sparsity [{:.4f}%] '.format(100 * sp))
    return sp


def binary_mask(data_mask, percent):

    edge_total = data_mask.shape[0]
    edge_y, edge_i = torch.sort(data_mask.abs())
    edge_thre_index = int(edge_total * percent)
    edge_thre = edge_y[edge_thre_index]
    binary_mask = get_each_mask(data_mask, edge_thre)
    
    return binary_mask



def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask


def setup_all(args):

    setup_seed(args.seed)
    # args.save_dir = os.path.join(args.save_dir, args.dataset)
    # if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    prGreen("Exp Information:")
    prGreen("-" * 150)
    prYellow("Dataset:[{}] | Save Dir:[{}] | Seed:[{}]"
            .format(args.dataset, 
                    args.save_dir, 
                    args.seed))
    print_args(args)
    prGreen("-" * 100)
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



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


def save_imp_things(fold_list, save_dir, file_name):
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    name = os.path.join(save_dir, file_name)
    fold_dict = {}
    for i, things in enumerate(fold_list):
        fold_dict[str(i + 1)] = things

    torch.save(fold_dict, name)
    print("save in {}".format(name))


def save_model(model, save_dir, file_name):

    if 'pretrain' in save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        name = os.path.join(save_dir, file_name)
        torch.save(model.state_dict(), name)
        print("save in {}".format(name))


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

