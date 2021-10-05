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

device = torch.device('cuda')

def train_get_mask(dataset_ori, model_func, masker_func, fold_things_list, imp_num, args):

    new_fold_things_list = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):
        
        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset_ori[train_idx]
        test_dataset_ori = dataset_ori[test_idx]
        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        model = model_func(dataset_ori).to(device)
        masker = masker_func(dataset_ori).to(device)
        
        if fold_things_list:

            train_dataset_pru = fold_things_list[fold]['train_dataset_pru']
            test_dataset_pru = fold_things_list[fold]['test_dataset_pru']
            train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
            test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)
            sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
            sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
            spa = (sp_train + sp_test) / 2
            rewind_weight = fold_things_list[fold]['rewind_weight']
            rewind_weight2 = fold_things_list[fold]['rewind_weight2']
            model.load_state_dict(rewind_weight)
            masker.load_state_dict(rewind_weight2)
            if args.pruning_percent_w != 0:
                model_mask_dict = fold_things_list[fold]['model_mask_dict']
                pruning.pruning_model_by_mask(model, model_mask_dict)
            spw = pruning.see_zero_rate(model)

        else:
            
            train_dataset_pru = copy.deepcopy(train_dataset_ori)
            test_dataset_pru = copy.deepcopy(test_dataset_ori)
            train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
            test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)
            sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
            sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
            spa = (sp_train + sp_test) / 2
            rewind_weight = copy.deepcopy(model.state_dict())
            rewind_weight2 = copy.deepcopy(masker.state_dict())
            spw = pruning.see_zero_rate(model)
        
        optimizer = Adam([{'params': model.parameters(), 'lr': args.lr}, 
                          {'params': masker.parameters(),'lr': args.mask_lr}], weight_decay=args.weight_decay)
        
        for epoch in range(1, args.get_mask_epochs + 1):
            
            train_loss, train_acc, mask_distribution = train_model_and_masker(model, optimizer, train_loader_pru, device, args, masker=masker)
            test_acc, _ = eval_acc_with_pruned_dataset(model, masker, test_dataset_pru, device, args)
            # eval_acc_with_mask
            # test_acc = eval_acc_with_mask(model, masker, test_dataset_pru, device, args, binary=args.binary)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_masker_state_dict = copy.deepcopy(masker.state_dict())
                
                # best_model_state_dict = copy.deepcopy(model.state_dict())
            print("(Train IMP:[{}] fold:[{}] spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}] | 0-0.2:[{:.2f}%] 0.2-0.4:[{:.2f}%] 0.4-0.6:[{:.2f}%] 0.6-0.8:[{:.2f}%] 0.8-1.0:[{:.2f}%]"
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
                            best_epoch,
                            mask_distribution[0] * 100,
                            mask_distribution[1] * 100,
                            mask_distribution[2] * 100,
                            mask_distribution[3] * 100,
                            mask_distribution[4] * 100))

        fold_things_dict = {}
        # using model to generate the mask
        # model.load_state_dict(best_model_state_dict)
        if args.pruning_percent_w != 0:
            pruning.pruning_model(model, args.pruning_percent_w, random=False)
            _ = pruning.see_zero_rate(model)
            model_mask_dict = pruning.extract_mask(model)

        fold_things_dict['train_dataset_pru'] = train_dataset_pru 
        fold_things_dict['test_dataset_pru'] = test_dataset_pru 
        fold_things_dict['rewind_weight'] = rewind_weight
        fold_things_dict['rewind_weight2'] = rewind_weight2
        
        if args.pruning_percent_w != 0:
            fold_things_dict['model_mask_dict'] = model_mask_dict
        fold_things_dict['best_masker_state_dict'] = best_masker_state_dict
        new_fold_things_list.append(fold_things_dict)

    return new_fold_things_list


def eval_data_transfer(dataset_ori, model_func, masker_func, fold_things_list, imp_num, args):
    
    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):
        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset_ori[train_idx]
        test_dataset_ori = dataset_ori[test_idx]
        
        best_masker_state_dict = fold_things_list[str(fold + 1)]['best_masker_state_dict']
        masker = masker_func(dataset_ori).to(device)
        if not args.rp:
            masker.load_state_dict(best_masker_state_dict)
        pruning.grad_model(masker, False) # fix masker

        train_dataset_pru = pruning.masker_pruning_dataset(train_dataset_ori, masker, args)
        test_dataset_pru = pruning.masker_pruning_dataset(test_dataset_ori, masker, args)
        
        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)

        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
        sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
        spa = (sp_train + sp_test) / 2
        
        model = model_func(dataset_ori).to(device)
        spw = pruning.see_zero_rate(model)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc  = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader_pru, device, args.with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            print("(Test data transfer:[{}] fold:[{}] spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(imp_num,
                            fold,
                            spa * 100,
                            spw * 100,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))
        
        pruning.prRed("syd: data transfer:[{}] fold:[{}] | Dataset:[{}] spa:[{:.2f}%] spw:[{:.2f}%] | Best Test:[{:.2f}] at epoch [{}]"
                .format(imp_num,
                        fold,
                        args.dataset,
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
    pruning.prRed('sydall data transfer Dataset:[{}] IMP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Test Acc:{:.2f}±{:.2f} | \nsydal: Selected epoch:{} | acc list:{}'
         .format(args.dataset,
                 imp_num,
                 spa * 100,
                 spw * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))

    

def eval_tickets(dataset_ori, model_func, masker_func, fold_things_list, imp_num, args):
    
    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):
        
        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset_ori[train_idx]
        test_dataset_ori = dataset_ori[test_idx]

        rewind_weight = fold_things_list[fold]['rewind_weight']
        if args.pruning_percent_w != 0:
            model_mask_dict = fold_things_list[fold]['model_mask_dict']
        best_masker_state_dict = fold_things_list[fold]['best_masker_state_dict']
        masker = masker_func(dataset_ori).to(device)
        masker.load_state_dict(best_masker_state_dict)
        pruning.grad_model(masker, False) # fix masker

        if imp_num == 1:
            train_dataset_pru = pruning.masker_pruning_dataset(train_dataset_ori, masker, args)
            test_dataset_pru = pruning.masker_pruning_dataset(test_dataset_ori, masker, args)
        else:
            train_dataset_pru = fold_things_list[fold]['train_dataset_pru']
            train_dataset_pru = pruning.masker_pruning_dataset(train_dataset_pru, masker, args)
            test_dataset_pru = fold_things_list[fold]['test_dataset_pru']
            test_dataset_pru = pruning.masker_pruning_dataset(test_dataset_pru, masker, args)

        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)
        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
        sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
        spa = (sp_train + sp_test) / 2
        fold_things_list[fold]['train_dataset_pru'] = train_dataset_pru
        fold_things_list[fold]['test_dataset_pru'] = test_dataset_pru
        
        model = model_func(dataset_ori).to(device)
        model.load_state_dict(rewind_weight)
        if args.pruning_percent_w != 0:
            pruning.pruning_model_by_mask(model, model_mask_dict)
        spw = pruning.see_zero_rate(model)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc  = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader_pru, device, args.with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            print("(Test IMP:[{}] fold:[{}] spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(imp_num,
                            fold,
                            spa * 100,
                            spw * 100,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))
        

        pruning.prRed("syd: IMP:[{}] fold:[{}] | Dataset:[{}] spa:[{:.2f}%] spw:[{:.2f}%] | Best Test:[{:.2f}] at epoch [{}]"
                .format(imp_num,
                        fold,
                        args.dataset,
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
    pruning.prRed('sydall Final Dataset:[{}] IMP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Test Acc:{:.2f}±{:.2f} | \nsydal: Selected epoch:{} | acc list:{}'
         .format(args.dataset,
                 imp_num,
                 spa * 100,
                 spw * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))

    return fold_things_list


def pure_eval(dataset_ori, model_func, fold_things_list, imp_num, args):
    
    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):
        
        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset_ori[train_idx]
        test_dataset_ori = dataset_ori[test_idx]
        keys = str(fold + 1)
        rewind_weight = fold_things_list[keys]['rewind_weight']
        model_mask_dict = fold_things_list[keys]['model_mask_dict']
        train_dataset_pru = fold_things_list[keys]['train_dataset_pru']
        test_dataset_pru = fold_things_list[keys]['test_dataset_pru']
            
        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)

        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
        sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
        spa = (sp_train + sp_test) / 2

        model = model_func(dataset_ori).to(device)
        model.load_state_dict(rewind_weight)
        pruning.pruning_model_by_mask(model, model_mask_dict)
        spw = pruning.see_zero_rate(model)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc  = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader_pru, device, args.with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            print("(Test IMP:[{}] fold:[{}] spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(imp_num,
                            fold,
                            spa * 100,
                            spw * 100,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))
        
        pruning.prRed("syd: IMP:[{}] fold:[{}] | Dataset:[{}] spa:[{:.2f}%] spw:[{:.2f}%] | Best Test:[{:.2f}] at epoch [{}]"
                .format(imp_num,
                        fold,
                        args.dataset,
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
    pruning.prRed('sydall Final Dataset:[{}] IMP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Test Acc:{:.2f}±{:.2f} | \nsydal: Selected epoch:{} | acc list:{}'
         .format(args.dataset,
                 imp_num,
                 spa * 100,
                 spw * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))


def train_transfer(dataset_ori, model_func, fold_things_list, imp_num, args):
    
    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):
        
        best_test_acc, best_epoch = 0, 0
        train_dataset_ori = dataset_ori[train_idx]
        test_dataset_ori = dataset_ori[test_idx]
        keys = str(fold + 1)
        train_dataset_pru = fold_things_list[keys]['train_dataset_pru']
        test_dataset_pru = fold_things_list[keys]['test_dataset_pru']
            
        train_loader_ori = DataLoader(train_dataset_ori, args.batch_size, shuffle=True)
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
        test_loader_ori = DataLoader(test_dataset_ori, args.batch_size, shuffle=False)
        test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)

        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
        sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
        spa = (sp_train + sp_test) / 2

        model = model_func(dataset_ori).to(device)
        spw = pruning.see_zero_rate(model)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc  = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader_pru, device, args.with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            print("(Transfer IMP:[{}] fold:[{}] spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(imp_num,
                            fold,
                            spa * 100,
                            spw * 100,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))
        
        pruning.prRed("syd: Transfer IMP:[{}] fold:[{}] | Dataset:[{}] spa:[{:.2f}%] spw:[{:.2f}%] | Best Test:[{:.2f}] at epoch [{}]"
                .format(imp_num,
                        fold,
                        args.dataset,
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
    pruning.prRed('sydall Final Transfer Dataset:[{}] IMP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Test Acc:{:.2f}±{:.2f} | \nsydal: Selected epoch:{} | acc list:{}'
         .format(args.dataset,
                 imp_num,
                 spa * 100,
                 spw * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))