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
import pdb

def eval_rp(rp_num, args):
    
    t0 = time.time()
    print("random pruning...")
    
    device = torch.device("cuda:" + str(args.device))
    evaluator = Evaluator(args.dataset)
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GNN(gnn_type='gcn', 
                num_class=dataset.num_classes, 
                num_layer=args.num_layer, 
                emb_dim=args.emb_dim, 
                drop_ratio=args.drop_ratio, 
                virtual_node=False).to(device)
    # pdb.set_trace()
    masker = Masker(gnn_type='gcn', 
                    num_class=dataset.num_classes, 
                    num_layer=args.num_layer, 
                    emb_dim=args.emb_dim, 
                    drop_ratio=args.drop_ratio, 
                    virtual_node=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if args.resume:
        
        all_dict = torch.load(args.save_dir + "/last_rprp{}.pt".format(rp_num))
        start_epoch = all_dict['start_epoch']
        update_epoch = all_dict['update_epoch']
        update_test = all_dict['update_test']
        best_val = all_dict['best_val']
        model_state_dict = all_dict['model_state_dict']
        masker_state_dict = all_dict['masker_state_dict']
        optim_state_dict = all_dict['optim_state_dict']
        print("begin resume at epoch:{}".format(start_epoch))
        print("load model...")
        model_mask = pruning.extract_mask(model_state_dict) 
        pruning.pruning_model_by_mask(model, model_mask)
        model.load_state_dict(model_state_dict)
        masker.load_state_dict(masker_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        
        print("finish!")
    else:
        start_epoch = 1
        update_epoch = update_test = best_val = valid_perf = test_perf = 0
        pruning.pruning_model(model, args.pw, random=True)

    spw = pruning.see_zero_rate(model)
    pruning.grad_model(masker, False)

    t1 = time.time()
    print("done ! time:[{:.2f} min]".format((t1 - t0)/60))
    for epoch in range(start_epoch, args.epochs + 1):
        
        t0 = time.time()
        train_loss, mask_distribution = train_model_and_masker(model, masker, device, train_loader, optimizer, epoch, args, binary=True)
        if epoch > int(args.epochs / 2):
            valid_perf = eval_acc_with_mask(model, masker, device, valid_loader, evaluator, epoch, args, binary=True)
            test_perf = eval_acc_with_mask(model, masker, device, test_loader, evaluator, epoch, args, binary=True)
        
        if valid_perf > best_val:
            best_val = valid_perf
            update_epoch = epoch
            update_test = test_perf

        pruning.save_all({"epoch": epoch, 
                          "model_state_dict": model.state_dict(),
                          "masker_state_dict": masker.state_dict(),
                          "optim_state_dict": optimizer.state_dict(),
                          "best_val":best_val,
                          "update_epoch": update_epoch,
                          "update_test": update_test}, args.save_dir, "{}-last-rprp{}.pt".format(args.model, rp_num))

        epoch_time = time.time() - t0
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              "(RP:[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
              .format(rp_num,
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


# def eval_baseline(args):
    
#     print("baseline...")
#     device = torch.device("cuda:" + str(args.device))
#     evaluator = Evaluator(args.dataset)
#     dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
#     split_idx = dataset.get_idx_split()

#     train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     model = DeeperGCN(args).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     update_epoch = update_test = best_val = 0
#     for epoch in range(1, args.epochs + 1):
        
#         t0 = time.time()
#         total_loss = train(model, device, train_loader, optimizer, epoch, args)
#         valid_perf = eval(model, device, valid_loader, evaluator, epoch, args)
#         test_perf = eval(model, device, test_loader, evaluator, epoch, args)
        
#         if valid_perf > best_val:
#             best_val = valid_perf
#             update_epoch = epoch
#             update_test = test_perf
#             pruning.save_model(model, args.save_dir, "baseline_best_val.pt")
#         pruning.save_model(model, args.save_dir, "baseline_last.pt")

#         epoch_time = time.time() - t0
#         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
#               "(Baseline)  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
#               .format(epoch,
#                       args.epochs,
#                       train_loss,
#                       valid_perf * 100,
#                       test_perf * 100,
#                       update_test * 100,
#                       update_epoch,
#                       epoch_time / 60
#                       ))
if __name__ == "__main__":

    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)
    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1), i + 1) for i in range(20)]
    # eval_baseline(args)
    pa, pw, idx = percent_list[args.idx - 1]
    args.pw = pw
    args.pa = pa
    eval_rp(idx, args)


    # for _, (pa, pw, idx) in enumerate(percent_list[10:]):
    #     args.pw = pw
    #     args.pa = pa
    #     eval_rp(idx, args)