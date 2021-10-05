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

def train_get_mask(oip_num, things_dict, args):
    
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
        
    spw = pruning.see_zero_rate(model)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001 }, 
                            {'params': masker.parameters(),'lr': args.mask_lr}], weight_decay=0)

    update_epoch = update_test = best_val = 0
    update_masker_dict = copy.deepcopy(masker.state_dict())
    for epoch in range(1, args.epochs + 1):
        
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
              "(OIP (Train):[{}] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]"
              .format(oip_num,
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
        things_dict['rewind_weight'] = rewind_weight
        things_dict['rewind_weight2'] = rewind_weight2
        things_dict['resume_model'] = model.state_dict()
        things_dict['resume_masker'] = masker.state_dict()
        things_dict['update_masker_dict'] = update_masker_dict
        things_dict['epoch'] = epoch
        torch.save(things_dict, "train-imp-resume.pt")

    pruning.pruning_model(model, 0.2, random=False)
    _ = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)
    
    things_dict['rewind_weight'] = rewind_weight
    things_dict['rewind_weight2'] = rewind_weight2
    things_dict['model_mask_dict'] = model_mask_dict
    things_dict['update_masker_dict'] = update_masker_dict
   
    return things_dict


def eval_tickets(oip_num, things_dict, args):
    
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

    rewind_weight = things_dict['rewind_weight']
    model_mask_dict = things_dict['model_mask_dict']
    update_masker_dict = things_dict['update_masker_dict']
   
    model.load_state_dict(rewind_weight)
    masker.load_state_dict(update_masker_dict)
    pruning.grad_model(masker, False)
    pruning.pruning_model_by_mask(model, model_mask_dict)
    spw = pruning.see_zero_rate(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    update_epoch = update_test = best_val = 0

    for epoch in range(1, args.epochs + 1):
        
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
              "(OIP (Test):[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
              .format(oip_num,
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

        things_dict['resume_model'] = model.state_dict()
        things_dict['epoch'] = epoch
        torch.save(things_dict, "eval-imp-resume.pt")

    print("(syd: OIP final:[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Update Test:[{:.2f}] at Epoch:[{}]"
              .format(oip_num,
                      mask_distribution[0] * 100,
                      spw * 100,
                      update_test * 100,
                      update_epoch
                      ))


if __name__ == "__main__":

    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)
    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1)) for i in range(20)]
    things_dict = None
    for oip_num, (pa, pw) in enumerate(percent_list):
        args.pa = pa
        things_dict = train_get_mask(oip_num, things_dict, args)
        eval_tickets(oip_num, things_dict, args)