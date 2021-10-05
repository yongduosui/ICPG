import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
from tqdm import tqdm
import argparse
import time
import numpy as np
import time
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from train import train, eval, add_zeros
import pruning
import pdb

def main():
    # Training settings
    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)

    device = torch.device("cuda:" + str(args.device))
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = GNN(gnn_type='gcn', 
                num_class=dataset.num_classes, 
                num_layer=args.num_layer, 
                emb_dim=args.emb_dim, 
                drop_ratio=args.drop_ratio, 
                virtual_node=False).to(device)

    pruning.save_model(model, args.save_dir, "epoch0_weight.pt")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_perf = 0.0
    train_curve, test_curve, valid_curve = [], [], []
    update_epoch = update_test = best_val = 0
    for epoch in range(1, args.epochs + 1):
        
        t0 = time.time()
        train_loss = train(model, device, train_loader, optimizer, epoch, args)
        if epoch % 20 == 0:
            train_perf = eval(model, device, train_loader, evaluator, epoch, args)
        valid_perf = eval(model, device, valid_loader, evaluator, epoch, args)
        test_perf = eval(model, device, test_loader, evaluator, epoch, args)
        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        if valid_perf > best_val:
            best_val = valid_perf
            update_epoch = epoch
            update_test = test_perf
            pruning.save_model(model, args.save_dir, "best_val.pt")
        pruning.save_model(model, args.save_dir, "last.pt")

        epoch_time = time.time() - t0
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              "(Baseline)  Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
              .format(epoch,
                      args.epochs,
                      train_loss,
                      train_perf * 100, 
                      valid_perf * 100,
                      test_perf * 100,
                      update_test * 100,
                      update_epoch,
                      epoch_time / 60
                      ))
        
    print(train_curve, valid_curve, test_curve)
    
if __name__ == "__main__":
    main()