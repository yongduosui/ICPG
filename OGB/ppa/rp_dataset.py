from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from train import train, eval, add_zeros
import torch
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
import time
import numpy as np
import pruning
import pdb
import os

def random_dataset(rp_num, args):
    
    t0 = time.time()
    print("process...")

    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    dataset_pru = pruning.random_pruning_dataset(dataset, args)
    spa = pruning.print_pruning_percent(dataset, dataset_pru)

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    
    train_dataset = [dataset_pru[i] for i in split_idx["train"]]
    valid_dataset = [dataset_pru[i] for i in split_idx["valid"]]
    test_dataset =  [dataset_pru[i] for i in split_idx["test"]]

    dataset_dict = {}
    dataset_dict['train'] = train_dataset
    dataset_dict['valid'] = valid_dataset
    dataset_dict['test'] = test_dataset

    save_dir = "random_dataset"
    file_name = "rp{}".format(rp_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    name = os.path.join(save_dir, file_name)
    torch.save(dataset_dict, name)
    

if __name__ == "__main__":

    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)
    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1)) for i in range(20)]
    for rp_num, (pa, pw) in enumerate(percent_list):
        args.pa = pa
        args.pw = pw
        random_dataset(rp_num, args)