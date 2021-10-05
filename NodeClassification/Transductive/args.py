import torch
import random
import numpy as np
import argparse
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def parser_loader():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--mask_lr', type=float, default=0.01)
    parser.add_argument('--mask_wd', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=300)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--pruning_adj', type=float, default=0.05)
    parser.add_argument('--pruning_wei', type=float, default=0.2)
    parser.add_argument('--score_function', type=str, default='concat_mlp')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--masker_dim', type=int, default=128)
    
    return parser


def setup_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def get_dataset(args):

    dataset = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    return dataset