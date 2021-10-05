import dgl
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from train.train_imp import train_epoch, evaluate_network
from tqdm import tqdm
from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
import pruning
import copy
import pdb

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

"""
    TRAINING CODE
"""

def run_rpnp(dataset, net_params, rp_num, args):

    dataset_pru = LoadData("MNIST", args)
    trainset_pru, valset_pru, testset_pru = dataset_pru.train, dataset_pru.val, dataset_pru.test 
    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    sp_train = pruning.print_pruning_percent(trainset, trainset_pru)
    sp_test = pruning.print_pruning_percent(testset, testset_pru)
    sp_val = pruning.print_pruning_percent(valset, valset_pru)
    spa = (sp_train + sp_test + sp_val) / 3.0

    device = torch.device("cuda")
    model = gnn_model("GCN", net_params)
    model = model.to(device)
    spw = 0.0

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=10,
                                                     verbose=True)

    train_loader = DataLoader(trainset_pru, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset.collate)
    val_loader = DataLoader(valset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset.collate)
    run_time, best_val_acc, best_epoch, update_test_acc  = 0, 0, 0, 0
    for epoch in range(args.eval_epochs):

        t0 = time.time()
        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, args)
        epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
        _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)                
        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            update_test_acc = epoch_test_acc
            best_epoch = epoch

        print('-'*120)
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                'RPNP:[{}] spa[{:.2f}%] spw:[{:.2f}%] | Epoch [{}/{}]: Loss [{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}] | Run Total Time: [{:.2f} min]'
                .format(rp_num,
                        spa * 100,
                        spw * 100,
                        epoch + 1, 
                        args.eval_epochs,
                        epoch_train_loss, 
                        epoch_train_acc * 100,
                        epoch_val_acc * 100, 
                        epoch_test_acc * 100, 
                        update_test_acc * 100,
                        best_epoch,
                        run_time / 60)) 
        print('-'*120)
        
    print("syd: RPNP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Update Test:[{:.2f}] at epoch:[{}]"
            .format(rp_num,
                    spa * 100,
                    spw * 100,
                    update_test_acc * 100,
                    best_epoch))


def main():    
    """
        USER CONTROLS
    """
    args = pruning.parser_loader().parse_args()
    pruning.setup_seed(args.seed)
    pruning.print_args(args)  
   
    with open(args.config) as f:
        config = json.load(f)
    
    DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME, args)

    params = config['params']
    params['seed'] = int(args.seed)
    net_params = config['net_params']
    
    net_params['batch_size'] = params['batch_size']
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes

    things_dict = None
    adj_pruning = 0.05
    wei_pruning = 0.2
    args.pw = wei_pruning
    start = 4
    percent_list = [(1 - (1 - adj_pruning) ** (i + 1), 1 - (1 - wei_pruning) ** (i + 1)) for i in range(20)]
    for rp_num, (pa, pw) in enumerate(percent_list[start:]):
        rp_num = rp_num + start
        args.pa = pa
        run_rpnp(dataset, net_params, rp_num + 1, args)

    
if __name__ == '__main__':
    main()

