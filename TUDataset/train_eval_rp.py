import sys
import time
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.data import DataLoader
from utils import print_weights, k_fold
import pruning
import copy
import pdb
from train import eval_acc, eval_loss, train_model_and_masker, train, eval_acc_with_mask, eval_acc_with_pruned_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_random(dataset=None, dataset_pru=None, model_func=None, args=None):

    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):
        
        # if fold > 0: break
        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset[train_idx]
        test_dataset_ori = dataset[test_idx]
        
        train_dataset_pru = [dataset_pru[i] for i in train_idx.tolist()]
        test_dataset_pru = [dataset_pru[i] for i in test_idx.tolist()]
        # train_dataset_pru = pruning.random_pruning_dataset(train_dataset_ori, args)
        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)

        # test_dataset_pru = pruning.random_pruning_dataset(test_dataset_ori, args)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)
        
        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
        sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)

        model = model_func(dataset).to(device)
        # print(model)
        pruning.pruning_model(model, args.pruning_percent_wei, random=True)
        spw = pruning.see_zero_rate(model)
        spa = (sp_train + sp_test) / 2

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):

            train_loss, train_acc = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader_pru, device, args.with_eval_mode)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                #pruning.save_model(model, args.save_dir, "fold_{}_best.pt".format(fold + 1))
            
            print("(Random [{}] dataset:[{}] Model:[{}] fold:[{}]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}] | spa:[{:.2f}%] spw:[{:.2f}%]"
                    .format(args.random_type,
                            args.dataset,
                            args.model,
                            fold,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch,
                            spa * 100,
                            spw * 100))

        #pruning.save_model(model, args.save_dir, "fold_{}_last.pt".format(fold + 1))
        pruning.prRed("syd: Random [{}] fold:[{}] | Dataset:[{}] Model:[{}] spa:[{:.2f}%] spw:[{:.2f}%] | Best Test:[{:.2f}] at epoch [{}]"
                .format(args.random_type,
                        fold,
                        args.dataset,
                        args.model,
                        spa * 100,
                        spw * 100,
                        best_test_acc * 100, 
                        best_epoch))
    
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    if args.epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(args.folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()

    pruning.prRed('sydall Final: Random [{}] Dataset:[{}] Model:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Test Acc: {:.2f}±{:.2f} | \nsydal: Selected epoch:{}| acc list:{}'
         .format(args.random_type,
                 args.dataset,
                 args.model,
                 spa * 100,
                 spw * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))


def train_eval_model_imp(dataset_ori, dataset_pru, model_func, fold_things_list, imp_num, args):

    new_fold_things_list = []
    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):

        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset_ori[train_idx]
        test_dataset_ori = dataset_ori[test_idx]
        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        model = model_func(dataset_ori).to(device)

        train_dataset_pru = [dataset_pru[i] for i in train_idx.tolist()]
        test_dataset_pru = [dataset_pru[i] for i in test_idx.tolist()]
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
        test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)

        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
        sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
        spa = (sp_train + sp_test) / 2

        if fold_things_list:
            
            rewind_weight = fold_things_list[fold]['rewind_weight']
            model_mask_dict = fold_things_list[fold]['model_mask_dict']
            model.load_state_dict(rewind_weight)
            pruning.pruning_model_by_mask(model, model_mask_dict)
        
        else:
            rewind_weight = copy.deepcopy(model.state_dict())

        spw = pruning.see_zero_rate(model)
        
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader_ori, device, args.with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_model_state_dict = copy.deepcopy(model.state_dict())

            
            print("(Train RPIMP:[{}] fold:[{}] spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(imp_num,
                            fold,
                            spa * 100,
                            spw * 100,
                            epoch, 
                            args.get_mask_epochs, 
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))

        fold_things_dict = {}
        # using model to generate the mask
        model.load_state_dict(best_model_state_dict)
        pruning.pruning_model(model, 0.2, random=False)
        _ = pruning.see_zero_rate(model)
        
        model_mask_dict = pruning.extract_mask(model)
        fold_things_dict['rewind_weight'] = rewind_weight
        fold_things_dict['model_mask_dict'] = model_mask_dict

        new_fold_things_list.append(fold_things_dict)

    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    if args.epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(args.folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()

    pruning.prRed('sydall Final: RPIMP:[{}] Dataset:[{}] spa:[{:.2f}%] spw:[{:.2f}%] | Test Acc: {:.2f}±{:.2f} | \nsydal: Selected epoch:{}| acc list:{}'
         .format(imp_num,
                 args.dataset,
                 spa * 100,
                 spw * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))

    return new_fold_things_list