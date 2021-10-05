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


def eval_tickets(args):
    
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

    a = torch.load("resume_ckpt/eval-imp-resume.pt")
    model_mask_dict = a['model_mask_dict']
    update_masker_dict = a["update_masker_dict"]
    resume_model = a["resume_model"]

    masker.load_state_dict(update_masker_dict)
    pruning.grad_model(masker, False)
    pruning.pruning_model_by_mask(model, model_mask_dict)
    model.load_state_dict(resume_model)
    spw = pruning.see_zero_rate(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    update_epoch = update_test = best_val = 0
    test_perf = eval_acc_with_mask(model, masker, device, test_loader, evaluator, 0, args, binary=True)
    print(test_perf)



if __name__ == "__main__":

    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)
    args.pa = 0.05
    eval_tickets(args)