import numpy as np
import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from train.train_imp import train_model_and_masker, eval_acc_with_mask, train_epoch, evaluate_network
from nets.superpixels_graph_classification.load_net import gnn_model, mask_model
from data.data import LoadData # import dataset
import pruning
import copy
import pdb

def train_get_mask(dataset_ori, net_params, things_dict, imp_num, args):

    t0 = time.time()
    print("process ...")
    trainset_ori, valset_ori, testset_ori = dataset_ori.train, dataset_ori.val, dataset_ori.test
    
    train_loader_ori = DataLoader(trainset_ori, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset_ori.collate)
    val_loader_ori = DataLoader(valset_ori, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    test_loader_ori = DataLoader(testset_ori, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    device = torch.device("cuda")
    model = gnn_model("GCN", net_params).to(device)
    masker = mask_model(net_params).to(device)
    
    if things_dict is not None:

        trainset_pru, valset_pru, testset_pru = things_dict['trainset_pru'], things_dict['valset_pru'], things_dict['testset_pru']
        train_loader_pru = DataLoader(trainset_pru, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset_ori.collate)
        val_loader_pru = DataLoader(valset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        test_loader_pru = DataLoader(testset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)

        rewind_weight = things_dict['rewind_weight']
        rewind_weight2 = things_dict['rewind_weight2']
        model_mask_dict = things_dict['model_mask_dict']
        model.load_state_dict(rewind_weight)
        pruning.pruning_model_by_mask(model, model_mask_dict)
        masker.load_state_dict(rewind_weight2)
        
    else:
        trainset_pru = copy.deepcopy(trainset_ori)
        valset_pru = copy.deepcopy(valset_ori)
        testset_pru = copy.deepcopy(testset_ori)

        train_loader_pru = DataLoader(trainset_pru, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset_ori.collate)
        val_loader_pru = DataLoader(valset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        test_loader_pru = DataLoader(testset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        rewind_weight = copy.deepcopy(model.state_dict())
        rewind_weight2 = copy.deepcopy(masker.state_dict())

    sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
    sp_val = pruning.print_pruning_percent(val_loader_ori, val_loader_pru)
    sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
    spa = (sp_train + sp_test + sp_val) / 3
    spw = pruning.see_zero_rate(model)
        
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001 }, 
                            {'params': masker.parameters(),'lr': args.masker_lr}], weight_decay=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=10, verbose=True)                                                                        
    run_time, best_val_acc, best_epoch, update_test_acc  = 0, 0, 0, 0

    print("done! cost time:[{:.2f} min]".format((time.time() - t0) / 60))
    for epoch in range(args.mask_epochs):

        t0 = time.time()
        epoch_train_loss, epoch_train_acc, optimizer, mask_distribution = train_model_and_masker(model, masker, optimizer, device, train_loader_pru, epoch, args)
        epoch_val_loss, epoch_val_acc = eval_acc_with_mask(model, masker, device, val_loader_pru, epoch)
        _, epoch_test_acc = eval_acc_with_mask(model, masker, device, test_loader_pru, epoch)     

        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc > best_val_acc:
            update_test_acc = epoch_test_acc
            best_epoch = epoch
            best_masker_state_dict = copy.deepcopy(masker.state_dict())

        print('-'*120)
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                'Train IMP:[{}] spa[{:.2f}%] spw:[{:.2f}%] | Epoch [{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] epoch:[{}] | Time:[{:.2f} min] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]'
                .format(imp_num,
                        spa * 100,
                        spw * 100,
                        epoch + 1, 
                        args.mask_epochs,
                        epoch_train_loss, 
                        epoch_train_acc * 100,
                        epoch_val_acc * 100, 
                        epoch_test_acc * 100, 
                        update_test_acc * 100,
                        best_epoch,
                        run_time / 60,
                        mask_distribution[0] * 100,
                        mask_distribution[1] * 100,
                        mask_distribution[2] * 100,
                        mask_distribution[3] * 100,
                        mask_distribution[4] * 100)) 
        print('-'*120)
        
    things_dict = {}
    
    pruning.pruning_model(model, args.pw, random=False)
    _ = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)
    # masker.load_state_dict(best_masker_state_dict)
    t0 = time.time()
    trainset_pru = pruning.masker_pruning_dataset(train_loader_pru, masker, device, args)
    valset_pru = pruning.masker_pruning_dataset(val_loader_pru, masker, device, args)
    testset_pru = pruning.masker_pruning_dataset(test_loader_pru, masker,device, args)
    t1 = time.time()
    
    sp_train = pruning.print_pruning_percent(trainset_ori, trainset_pru)
    sp_val = pruning.print_pruning_percent(valset_ori, valset_pru)
    sp_test = pruning.print_pruning_percent(testset_ori, testset_pru)
    spa = (sp_train + sp_test + sp_val) / 3

    print("INFO: Data Sparsity:[{:.2f}%] time:[{:.2f} min]".format(spa * 100, (t1 - t0)/60))
    things_dict['trainset_pru'] = trainset_pru 
    things_dict['valset_pru'] = valset_pru 
    things_dict['testset_pru'] = testset_pru 
    things_dict['rewind_weight'] = rewind_weight
    things_dict['rewind_weight2'] = rewind_weight2
    things_dict['model_mask_dict'] = model_mask_dict

    return things_dict


def eval_tickets(dataset_ori, net_params, things_dict, imp_num, args):

    trainset_ori, valset_ori, testset_ori = dataset_ori.train, dataset_ori.val, dataset_ori.test
    train_loader_ori = DataLoader(trainset_ori, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset_ori.collate)
    val_loader_ori = DataLoader(valset_ori, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    test_loader_ori = DataLoader(testset_ori, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    device = torch.device("cuda")
    model = gnn_model("GCN", net_params)
    model = model.to(device)

    trainset_pru, valset_pru, testset_pru = things_dict['trainset_pru'], things_dict['valset_pru'], things_dict['testset_pru']
    train_loader_pru = DataLoader(trainset_pru, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset_ori.collate)
    val_loader_pru = DataLoader(valset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    test_loader_pru = DataLoader(testset_pru, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    rewind_weight = things_dict['rewind_weight']
    model_mask_dict = things_dict['model_mask_dict']
    model.load_state_dict(rewind_weight)
    pruning.pruning_model_by_mask(model, model_mask_dict)
    
    sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
    sp_val = pruning.print_pruning_percent(val_loader_ori, val_loader_pru)
    sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
    spa = (sp_train + sp_test + sp_val) / 3
    spw = pruning.see_zero_rate(model)
        
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,verbose=True)
                                                                                                 
    run_time, best_val_acc, best_epoch, update_test_acc  = 0, 0, 0, 0

    for epoch in range(args.eval_epochs):

        t0 = time.time()
        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader_pru, epoch, args)
        epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader_pru, epoch)
        _, epoch_test_acc = evaluate_network(model, device, test_loader_pru, epoch)     

        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            update_test_acc = epoch_test_acc
            best_epoch = epoch

        print('-'*120)
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                'Test IMP:[{}] spa[{:.2f}%] spw:[{:.2f}%] | Epoch [{}/{}]: Loss [{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}] | Run Total Time: [{:.2f} min]'
                .format(imp_num,
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
        
    print("sydfinal IMP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Update Test:[{:.2f}] at epoch:[{}]"
            .format(imp_num,
                    spa * 100,
                    spw * 100,
                    update_test_acc * 100,
                    best_epoch))
    
def main():  

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
    args.pa = adj_pruning
    args.pw = wei_pruning

    for imp_num in range(1, 21):

        things_dict = train_get_mask(dataset, net_params, things_dict, imp_num, args)
        eval_tickets(dataset, net_params, things_dict, imp_num, args)

    
if __name__ == '__main__':
    main()

