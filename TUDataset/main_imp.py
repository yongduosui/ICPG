from functools import partial
from itertools import product
import pruning
import torch
import re
import sys
import argparse
from utils import logger
from datasets import get_dataset
from train_eval_imp import train_get_mask, eval_tickets
from res_gcn import ResGCN, GCNmasker
import random
import copy
import pdb

DATA_SOCIAL = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI']
DATA_SOCIAL += ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY']
DATA_BIO = ['MUTAG', 'NCI1', 'PROTEINS', 'DD', 'ENZYMES', 'PTC_MR']
DATA_REDDIT = [data for data in DATA_BIO + DATA_SOCIAL if "REDDIT" in data]
DATA_NOREDDIT = [data for data in DATA_BIO + DATA_SOCIAL if "REDDIT" not in data]
DATA_SUBSET_STUDY = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'PROTEINS', 'DD']
DATA_SUBSET_STUDY_SUP = [d for d in DATA_SOCIAL + DATA_BIO if d not in DATA_SUBSET_STUDY]
DATA_SUBSET_FAST = ['IMDB-BINARY', 'PROTEINS', 'IMDB-MULTI', 'ENZYMES']
DATA_IMAGES = ['MNIST', 'MNIST_SUPERPIXEL', 'CIFAR10']

str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()

################## GCN Auto Mask Training Settings ##################
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--get_mask_epochs', type=int, default=100)
parser.add_argument('--score_function', type=str, default="concat_mlp", help="concat_mlp, inner_product")
parser.add_argument('--mask_lr', type=float, default=1e-4)
parser.add_argument('--mask_dim', type=int, default=64)
parser.add_argument('--mask_type', type=str, default="GCN", help="GCN, GIN, GAT, MLP etc..")
parser.add_argument('--save_masker_ckpt', type=str2bool, default=False)

parser.add_argument('--pruning_percent', type=float, default=0.05)
parser.add_argument('--pruning_percent_w', type=float, default=0.2)
parser.add_argument('--binary', type=str2bool, default=False)
################## GCN Training Settings ##################
parser.add_argument('--dataset', type=str, default="NCI1")
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--data_root', type=str, default="data")
parser.add_argument('--save_dir',  type=str, default="debug")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--semi_split', type=int, default=10)
args = parser.parse_args()


def create_n_filter_triples(datasets, 
                            feat_strs, 
                            nets, 
                            gfn_add_ak3=False,
                            gfn_reall=True, 
                            reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]
    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered


def get_model_and_masker():
    def model_func(dataset):
        return ResGCN(dataset, hidden=128)  

    def masker_func(dataset):
        return GCNmasker(dataset, hidden=args.mask_dim, 
                                  score_function=args.score_function,
                                  mask_type=args.mask_type) 

    return model_func, masker_func


def run_all(dataset_feat_net_triples):
    
    dataset_name, feat_str, net = dataset_feat_net_triples[0]
    dataset_ori = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root, pruning_percent=0)
    
    model_func, masker_func = get_model_and_masker()
    fold_things_list = None
    # args.pruning_percent_w = 0.0
    save_dir = "impckpt/{}-lr-{}-dim-{}".format(args.dataset,
                                                          args.mask_lr,
                                                          args.mask_dim)
    for imp_num in range(1, 21):
        
        fold_things_list = train_get_mask(dataset_ori=dataset_ori,  
                                          model_func=model_func, 
                                          masker_func=masker_func, 
                                          fold_things_list=fold_things_list,
                                          imp_num=imp_num, 
                                          args=args)
        file_name = "imp-train{}.pt".format(imp_num)
        
        pruning.save_imp_things(fold_things_list, save_dir, file_name)
        fold_things_list = eval_tickets(dataset_ori=dataset_ori,
                                        model_func=model_func, 
                                        masker_func=masker_func, 
                                        fold_things_list=fold_things_list,
                                        imp_num=imp_num,
                                        args=args)
        file_name = "imp-eval{}.pt".format(imp_num)
        pruning.save_imp_things(fold_things_list, save_dir, file_name)


def main():

    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    pruning.setup_all(args)
    
    run_all(create_n_filter_triples(datasets, 
                                    feat_strs, 
                                    nets,
                                    gfn_add_ak3=True,
                                    reddit_odeg10=True,
                                    dd_odeg10_ak1=True))

if __name__ == '__main__':
    main()