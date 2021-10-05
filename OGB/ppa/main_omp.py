from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from train import train, eval, add_zeros, train_model_and_masker, eval_acc_with_mask
import torch
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN, Masker
import time
import numpy as np
import pruning
import copy
import pdb

def train_get_mask(omp_num, things_dict, args, resume=False):
    
    device = torch.device("cuda:" + str(args.device))
    evaluator = Evaluator(args.dataset)
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    model = GNN(gnn_type='gcn', 
                num_class=dataset.num_classes, 
                num_layer=args.num_layer, 
                emb_dim=args.emb_dim, 
                drop_ratio=args.drop_ratio, 
                virtual_node=False).to(device)

    masker = Masker(gnn_type='gcn', 
                    num_class=dataset.num_classes, 
                    num_layer=args.num_layer, 
                    emb_dim=args.emb_dim, 
                    drop_ratio=args.drop_ratio, 
                    virtual_node=False).to(device)

    if things_dict is not None:
        rewind_weight = things_dict['rewind_weight']
        rewind_weight2 = things_dict['rewind_weight2']
        model.load_state_dict(rewind_weight)
        masker.load_state_dict(rewind_weight2)
        model_mask_dict = things_dict['model_mask_dict']
        pruning.pruning_model_by_mask(model, model_mask_dict)
        
    else:
        things_dict = {}
        rewind_weight = copy.deepcopy(model.state_dict())
        rewind_weight2 = copy.deepcopy(masker.state_dict())
    
    if resume:
        resume_dict = torch.load("omp_ckpt/train-imp{}-resume.pt".format(omp_num))
        start_epoch = resume_dict['epoch']
        model.load_state_dict(resume_dict['resume_model'])
        masker.load_state_dict(resume_dict['resume_masker'])
        update_masker_dict = resume_dict['update_masker_dict']
        update_epoch = resume_dict['update_epoch']
        update_test = resume_dict['update_test']
        best_val = resume_dict['best_val']
        test_perf = 0
        print("begining resume at epoch:{}".format(start_epoch))

    else:
        resume_dict = {}
        start_epoch = 1
        update_epoch = update_test = best_val = test_perf = 0
    
    spw = pruning.see_zero_rate(model)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001 }, 
                            {'params': masker.parameters(),'lr': args.mask_lr}], weight_decay=0)

    update_masker_dict = copy.deepcopy(masker.state_dict())
    for epoch in range(start_epoch, args.epochs + 1):
        
        t0 = time.time()
        train_loss, mask_distribution = train_model_and_masker(model, masker, device, train_loader, optimizer, epoch, args)
        valid_perf = eval_acc_with_mask(model, masker, device, valid_loader, evaluator, epoch, args)
        test_perf = eval_acc_with_mask(model, masker, device, test_loader, evaluator, epoch, args)
        
        if valid_perf > best_val:
            best_val = valid_perf
            update_epoch = epoch
            update_test = test_perf
            update_masker_dict = copy.deepcopy(masker.state_dict())

        epoch_time = time.time() - t0
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              "(OMP (Train):[{}] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]"
              .format(omp_num,
                      spw * 100,
                      epoch,
                      args.epochs,
                      train_loss,
                      valid_perf * 100,
                      test_perf * 100,
                      update_test * 100,
                      update_epoch,
                      epoch_time / 60,
                      mask_distribution[0] * 100,
                      mask_distribution[1] * 100,
                      mask_distribution[2] * 100,
                      mask_distribution[3] * 100,
                      mask_distribution[4] * 100
                      ))
        resume_dict['resume_model'] = model.state_dict()
        resume_dict['resume_masker'] = masker.state_dict()
        resume_dict['update_masker_dict'] = update_masker_dict
        resume_dict['epoch'] = epoch
        resume_dict['best_val'] = best_val
        resume_dict['update_epoch'] = update_epoch
        resume_dict['update_test'] = test_perf
        torch.save(resume_dict, "omp_ckpt/train-imp{}-resume.pt".format(omp_num))
        

    pruning.pruning_model(model, args.pw, random=False)
    _ = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)
    
    things_dict['rewind_weight'] = rewind_weight
    things_dict['rewind_weight2'] = rewind_weight2
    things_dict['model_mask_dict'] = model_mask_dict
    things_dict['update_masker_dict'] = update_masker_dict
    torch.save(things_dict, "omp_ckpt/train-imp{}-things.pt".format(omp_num))
    return things_dict


def eval_tickets(omp_num, things_dict, args, resume=False):
    
    device = torch.device("cuda:" + str(args.device))
    evaluator = Evaluator(args.dataset)
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    model = GNN(gnn_type='gcn', 
                num_class=dataset.num_classes, 
                num_layer=args.num_layer, 
                emb_dim=args.emb_dim, 
                drop_ratio=args.drop_ratio, 
                virtual_node=False).to(device)

    masker = Masker(gnn_type='gcn', 
                    num_class=dataset.num_classes, 
                    num_layer=args.num_layer, 
                    emb_dim=args.emb_dim, 
                    drop_ratio=args.drop_ratio, 
                    virtual_node=False).to(device)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if resume:

        test_perf = resume_dict['update_test']  
        start_epoch = resume_dict['epoch']
        update_epoch = resume_dict['update_epoch']
        update_test = resume_dict['update_test']
        best_val = resume_dict['best_val']
        model.load_state_dict(resume_dict['resume_model'])
        masker.load_state_dict(resume_dict['update_masker_dict'])
    else:
        update_epoch = update_test = best_val = test_perf = 0
        start_epoch = 1
        rewind_weight = things_dict['rewind_weight']
        model_mask_dict = things_dict['model_mask_dict']
        update_masker_dict = things_dict['update_masker_dict']
        model.load_state_dict(rewind_weight)
        masker.load_state_dict(update_masker_dict)

    pruning.grad_model(masker, False)
    pruning.pruning_model_by_mask(model, model_mask_dict)
    spw = pruning.see_zero_rate(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    resume_dict = {}
    for epoch in range(start_epoch, args.epochs + 1):
        
        t0 = time.time()
        train_loss, mask_distribution = train_model_and_masker(model, masker, device, train_loader, optimizer, epoch, args, binary=True)
        valid_perf = eval_acc_with_mask(model, masker, device, valid_loader, evaluator, epoch, args, binary=True)
        test_perf = eval_acc_with_mask(model, masker, device, test_loader, evaluator, epoch, args, binary=True)

        if valid_perf > best_val:
            best_val = valid_perf
            update_epoch = epoch
            update_test = test_perf

        epoch_time = time.time() - t0
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              "(OMP (Test):[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
              .format(omp_num,
                      mask_distribution[0] * 100,
                      spw * 100,
                      epoch,
                      args.epochs,
                      train_loss,
                      valid_perf * 100,
                      test_perf * 100,
                      update_test * 100,
                      update_epoch,
                      epoch_time / 60
                      ))
        
        resume_dict['update_masker_dict'] = update_masker_dict
        resume_dict['model_mask_dict'] = model_mask_dict
        resume_dict['resume_model'] = model.state_dict()
        resume_dict['epoch'] = epoch
        resume_dict['best_val'] = best_val
        resume_dict['update_epoch'] = update_epoch
        resume_dict['update_test'] = test_perf
        torch.save(things_dict, "omp_ckpt/eval-imp{}-resume.pt".format(omp_num))
        
    print("(syd final: OMP :[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Update Test:[{:.2f}] at Epoch:[{}]"
              .format(omp_num,
                      mask_distribution[0] * 100,
                      spw * 100,
                      update_test * 100,
                      update_epoch
                      ))


def eval_tickets2(omp_num, resume_dict, things_dict, args, resume=False):
    
    device = torch.device("cuda:" + str(args.device))
    evaluator = Evaluator(args.dataset)
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    model = GNN(gnn_type='gcn', 
                num_class=dataset.num_classes, 
                num_layer=args.num_layer, 
                emb_dim=args.emb_dim, 
                drop_ratio=args.drop_ratio, 
                virtual_node=False).to(device)

    masker = Masker(gnn_type='gcn', 
                    num_class=dataset.num_classes, 
                    num_layer=args.num_layer, 
                    emb_dim=args.emb_dim, 
                    drop_ratio=args.drop_ratio, 
                    virtual_node=False).to(device)
    
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if resume:

        test_perf = resume_dict['test_perf']  
        start_epoch = resume_dict['epoch']
        update_epoch = resume_dict['update_epoch']
        update_test = resume_dict['update_test']
        best_val = resume_dict['best_val']
        pruning.pruning_model_by_mask(model, resume_dict['model_mask_dict'])
        model.load_state_dict(resume_dict['resume_model'])
        spw = pruning.see_zero_rate(model)
        update_masker_dict = things_dict['update_masker_dict']
        masker.load_state_dict(update_masker_dict)
        pruning.grad_model(masker, False)
        
    else:
        update_epoch = update_test = best_val = test_perf = 0
        start_epoch = 1
        
        model.load_state_dict(resume_dict['resume_model'])
        pruning.pruning_model(model, args.pw, random=False)
        _ = pruning.see_zero_rate(model)
        model_mask_dict = pruning.extract_mask(model)
        pruning.remove_prune(model)
        model.load_state_dict(things_dict['rewind_weight'])
        pruning.pruning_model_by_mask(model, model_mask_dict)
        spw = pruning.see_zero_rate(model)
        update_masker_dict = things_dict['update_masker_dict']
        masker.load_state_dict(update_masker_dict)
        pruning.grad_model(masker, False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    resume_dict = {}
    for epoch in range(start_epoch, args.epochs + 1):
        
        t0 = time.time()
        train_loss, mask_distribution = train_model_and_masker(model, masker, device, train_loader, optimizer, epoch, args, binary=True)
        valid_perf = eval_acc_with_mask(model, masker, device, valid_loader, evaluator, epoch, args, binary=True)
        test_perf = eval_acc_with_mask(model, masker, device, test_loader, evaluator, epoch, args, binary=True)

        if valid_perf > best_val:
            best_val = valid_perf
            update_epoch = epoch
            update_test = test_perf

        epoch_time = time.time() - t0
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              "(OMP (Test):[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
              .format(omp_num,
                      mask_distribution[0] * 100,
                      spw * 100,
                      epoch,
                      args.epochs,
                      train_loss,
                      valid_perf * 100,
                      test_perf * 100,
                      update_test * 100,
                      update_epoch,
                      epoch_time / 60
                      ))
        
        resume_dict['update_masker_dict'] = update_masker_dict
        resume_dict['model_mask_dict'] = model_mask_dict
        resume_dict['resume_model'] = model.state_dict()
        resume_dict['epoch'] = epoch
        resume_dict['best_val'] = best_val
        resume_dict['update_epoch'] = update_epoch
        resume_dict['update_test'] = update_test
        resume_dict['test_perf'] = test_perf
        torch.save(things_dict, "omp_ckpt/eval-imp{}-resume.pt".format(omp_num))
        
    print("(syd final: OMP :[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Update Test:[{:.2f}] at Epoch:[{}]"
              .format(omp_num,
                      mask_distribution[0] * 100,
                      spw * 100,
                      update_test * 100,
                      update_epoch
                      ))

                       
if __name__ == "__main__":

    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)
    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1), i + 1) for i in range(20)]
    # eval_baseline(args)
    pa, pw, idx = percent_list[args.idx - 1]
    args.pw = pw
    args.pa = pa
    resume_dict = torch.load("omp_ckpt/train-imp5-resume.pt")
    things_dict = torch.load("omp_ckpt/train-imp5-things.pt")
    # things_dict = None
    eval_tickets2(idx, resume_dict, things_dict, args)
    # things_dict = None
    # things_dict = train_get_mask(idx, things_dict, args)
    # eval_tickets(idx, things_dict, args)