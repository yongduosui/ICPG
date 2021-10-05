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

def train_masker(dataset, model_func, masker_func, args):

    fold_state_dict_list = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):
        # if fold > 0: break
        best_test_acc, best_epoch = 0, 0
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model = model_func(dataset).to(device)
        masker = masker_func(dataset).to(device)

        if args.use_pretrain:
            model_state_dict = torch.load("pretrain/{}/fold_{}_best.pt".format(args.dataset, fold + 1))
            model.load_state_dict(model_state_dict)
            train_acc = eval_acc(model, train_loader, device, args.with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, args.with_eval_mode)
            print("pretrain model train acc:{:.2f} | test acc:{:.2f}".format(train_acc * 100, test_acc * 100))
            pruning.grad_model(model, False)
            optimizer = Adam(masker.parameters(), lr=args.mask_lr, weight_decay=args.weight_decay)
        else:
            optimizer = Adam([{'params': model.parameters(), 'lr': args.lr}, 
                              {'params': masker.parameters(),'lr': args.mask_lr}], weight_decay=args.weight_decay)
        
        for epoch in range(1, args.get_mask_epochs + 1):
            
            train_loss, train_acc, mask_distribution = train_model_and_masker(model, optimizer, train_loader, device, args, masker=masker)
            test_acc, sp_test = eval_acc_with_pruned_dataset(model, masker, test_dataset, device, args)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_masker_state_dict = copy.deepcopy(masker.state_dict())

            print("(Train {} fold:[{}] sp-test:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}] | 0-0.2:[{:.2f}%] 0.2-0.4:[{:.2f}%] 0.4-0.6:[{:.2f}%] 0.6-0.8:[{:.2f}%] 0.8-1.0:[{:.2f}%]"
                    .format(args.pruning_type,
                            fold,
                            sp_test * 100,
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

        fold_state_dict_list.append(best_masker_state_dict)
    return fold_state_dict_list

def eval_masker(dataset_ori=None, model_func=None, masker_func=None, state_dict_list=None, args=None):
    
    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_ori, args.folds, args.epoch_select))):

        # if fold > 0: break
        best_test_acc, best_epoch = 0, 0
        train_dataset = dataset_ori[train_idx]
        test_dataset = dataset_ori[test_idx]

        masker = masker_func(dataset_ori).to(device)
        masker.load_state_dict(state_dict_list[fold])
        pruning.grad_model(masker, False) # fix masker
        
        train_dataset_pru = pruning.masker_pruning_dataset(train_dataset, masker, args)
        train_loader_ori = DataLoader(train_dataset, args.batch_size, shuffle=True)
        train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
        sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)

        test_loader_ori = DataLoader(test_dataset, args.batch_size, shuffle=False)
        if args.eval_type == 'p2p':
            
            test_dataset_pru = pruning.masker_pruning_dataset(test_dataset, masker, args)
            test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)
            sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
            test_loader = test_loader_pru
        elif args.eval_type == 'p2o':

            sp_test = 1.0
            test_loader = test_loader_ori
        else:
            assert False

        model = model_func(dataset_ori).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc  = train(model, optimizer, train_loader_pru, device)
            test_acc = eval_acc(model, test_loader, device, args.with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            
            print("(Test {} fold:[{}]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}] | Sp-train:[{:.2f}%] Sp-test:[{:.2f}%]"
                    .format(args.pruning_type,
                            fold,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch,
                            sp_train * 100,
                            sp_test * 100))
        
        pruning.prRed("syd: fold:[{}] | Dataset:[{}] Type:[{}] Sp-train:[{:.2f}%] Sp-test:[{:.2f}%] | Best Test:[{:.2f}] at epoch [{}]"
                .format(fold,
                        args.dataset,
                        args.pruning_type,
                        sp_train * 100,
                        sp_test * 100,
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

    pruning.prRed('sydall Final: Dataset:[{}] Type:[{}] Sp-train:[{:.2f}%] Sp-test:[{:.2f}%] | Test Acc:{:.2f}±{:.2f} | \nsydal: Selected epoch:{} | acc list:{}'.
          format(args.dataset,
                 args.pruning_type,
                 sp_train * 100,
                 sp_test * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))

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
        pruning.pruning_model(model, args.pruning_percent_wei, random=True)
        sp_wei = pruning.see_zero_rate(model)
        
        if args.eval_type == 'p2p':
            train_loader = train_loader_pru
            test_loader = test_loader_pru
        elif args.eval_type == 'p2o':
            train_loader = train_loader_pru
            test_loader = test_loader_ori
        elif args.eval_type == 'o2p':
            train_loader = train_loader_ori
            test_loader = test_loader_pru
        else: assert False

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):

            train_loss, train_acc = train(model, optimizer, train_loader, device)
            test_acc = eval_acc(model, test_loader, device, args.with_eval_mode)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                pruning.save_model(model, args.save_dir, "fold_{}_best.pt".format(fold + 1))
            
            if epoch % 20 == 0:
                print("(Test [{}] dataset:[{}] fold:[{}]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}] | Sp-train:[{:.2f}%] Sp-test:[{:.2f}%] Sp-Wei:[{:.2f}%]"
                        .format(args.pruning_type,
                                args.dataset,
                                fold,
                                epoch, 
                                args.epochs,
                                train_loss, 
                                train_acc * 100, 
                                test_acc * 100, 
                                best_test_acc * 100, 
                                best_epoch,
                                sp_train * 100,
                                sp_test * 100,
                                sp_wei * 100))

        pruning.save_model(model, args.save_dir, "fold_{}_last.pt".format(fold + 1))
        pruning.prRed("syd: Test fold:[{}] | Dataset:[{}] Type:[{}] Sp-train:[{:.2f}%] Sp-test:[{:.2f}%] Sp-Wei:[{:.2f}%]| Best Test:[{:.2f}] at epoch [{}]"
                .format(fold,
                        args.dataset,
                        args.pruning_type,
                        sp_train * 100,
                        sp_test * 100, 
                        sp_wei * 100,
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

    pruning.prRed('sydall Final:Dataset:[{}] Type:[{}] Sp-train:[{:.2f}%] Sp-test:[{:.2f}%] Sp-Wei:[{:.2f}%] | Test Acc: {:.2f}±{:.2f} | \nsydal: Selected epoch:{}| acc list:{}'.
          format(args.dataset,
                 args.pruning_type,
                 sp_train * 100,
                 sp_test * 100,
                 sp_wei * 100,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))
    

def train_superpixel(dataset, model_func, args):
    
    
    train_dataset, test_dataset = dataset
    model = model_func(train_dataset).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    train_accs, test_accs = [], []
    best_test_acc, best_epoch = 0, 0
    for epoch in range(1, args.epochs + 1):
        
        train_loss, train_acc = train(model, optimizer, train_loader, device, show=True)
        test_acc = eval_acc(model, test_loader, device, args.with_eval_mode)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        print("(Test [{}] dataset:[{}]) Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(args.pruning_type,
                            args.dataset,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))