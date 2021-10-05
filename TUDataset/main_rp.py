from functools import partial
from itertools import product
import pruning
import re
import sys
import argparse
from utils import logger
from datasets import get_dataset
from train_eval_rp import eval_random, train_eval_model_imp
from res_gcn import ResGCN, GCNmasker, GINNet, GATNet
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
parser.add_argument('--idx', type=int, default=100)
parser.add_argument('--pruning_percent', type=float, default=0.1)
parser.add_argument('--pruning_percent_wei', type=float, default=0)
parser.add_argument('--random_type', type=str, default="rprp", help="rprp, rpimp, rpnp")
parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN")

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)

parser.add_argument('--reg', type=float, default=0.05)
parser.add_argument('--get_mask_epochs', type=int, default=50)
parser.add_argument('--score_function', type=str, default="concat_mlp", help="concat_mlp, inner_product")
parser.add_argument('--pruning_type', type=str, default="masker", help="masker, random")
parser.add_argument('--mask_lr', type=float, default=1e-4)
################## GCN Training Settings ##################
parser.add_argument('--dataset', type=str, default="NCI1")
parser.add_argument('--epochs', type=int, default=100)
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
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
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
    def model_func1(dataset):
        return ResGCN(dataset, hidden=128)  

    def model_func2(dataset):
        return GINNet(dataset, hidden=128) 

    def model_func3(dataset):
        return GATNet(dataset, hidden=args.hidden, 
                               num_fc_layers=args.n_layers_fc, 
                               num_conv_layers=args.n_layers_conv,
                               head=args.head,
                               dropout=args.dropout) 


    if args.model == "GCN":
        model_func = model_func1
    elif args.model == "GIN":
        model_func = model_func2
    elif args.model == "GAT":
        model_func = model_func3
    else:
        assert False

    def masker_func(dataset):
        return GCNmasker(dataset, hidden=64, score_function=args.score_function) 
    
    return model_func, masker_func


def run_rprp(dataset_feat_net_triples):
    
    dataset_name, feat_str, net = dataset_feat_net_triples[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root, pruning_percent=0)

    model_func, masker_func = get_model_and_masker()
    adj_pruning = 0.05
    wei_pruning = 0.2
    percent_list = [(1 - (1 - adj_pruning) ** (i + 1), 1 - (1 - wei_pruning) ** (i + 1)) for i in range(20)]
    # percent_list.reverse()
    # percent_list.append((0.0,0.0)) # no pruning baseline
    # percent_list.reverse()
    
    # pa, pw = percent_list[args.idx - 1]
    # args.pruning_percent = pa
    # if args.random_type == "rpnp":
    #     args.pruning_percent_wei = 0
    # else:
    #     args.pruning_percent_wei = pw
    
    # dataset_pru = pruning.random_pruning_dataset(dataset, args)
    # eval_random(dataset=dataset, 
    #             dataset_pru=dataset_pru,
    #             model_func=model_func, 
    #             args=args)

    for idx, (pa, pw) in enumerate(percent_list):
        # if idx % 3 == 1:
        args.pruning_percent = pa
        if args.random_type == "rpnp":
            args.pruning_percent_wei = 0
        else:
            args.pruning_percent_wei = pw
        
        dataset_pru = pruning.random_pruning_dataset(dataset, args)
        eval_random(dataset=dataset, 
                    dataset_pru=dataset_pru,
                    model_func=model_func, 
                    args=args)
        


def run_rpimp(dataset_feat_net_triples):
    
    dataset_name, feat_str, net = dataset_feat_net_triples[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root, pruning_percent=0)
    model_func, masker_func = get_model_and_masker()

    adj_pruning = 0.05
    wei_pruning = 0.2

    percent_list = [(1 - (1 - adj_pruning) ** (i + 1), 1 - (1 - wei_pruning) ** (i + 1)) for i in range(21)]
    fold_things_list = None
    dataset_pru = copy.deepcopy(dataset)
    for imp_num, (pa, pw) in enumerate(percent_list):
        if args.random_type == 'npimp':
            args.pruning_percent = 0
        else:
            args.pruning_percent = pa
        
        fold_things_list = train_eval_model_imp(dataset_ori=dataset, 
                                                dataset_pru=dataset_pru,
                                                model_func=model_func, 
                                                fold_things_list=fold_things_list,
                                                imp_num=imp_num,
                                                args=args)
        
        dataset_pru = pruning.random_pruning_dataset(dataset_pru, args)

def main():

    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    pruning.setup_all(args)

    if args.random_type in ["rprp", "rpnp"]:
        run_rprp(create_n_filter_triples(datasets, 
                                         feat_strs, 
                                         nets,
                                         gfn_add_ak3=True,
                                         reddit_odeg10=True,
                                         dd_odeg10_ak1=True))

    elif args.random_type in ['rpimp', 'npimp']:
        run_rpimp(create_n_filter_triples(datasets, 
                                         feat_strs, 
                                         nets,
                                         gfn_add_ak3=True,
                                         reddit_odeg10=True,
                                         dd_odeg10_ak1=True))
   
    else: assert False

if __name__ == '__main__':
    main()