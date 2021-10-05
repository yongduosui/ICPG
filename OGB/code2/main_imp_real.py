import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN, Masker
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from utils import ASTNodeEncoder, get_vocab_mapping
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq
from train import train, eval, train_model_and_masker, eval_acc_with_mask
import copy
import pdb
import pruning

def train_get_mask(imp_num, things_dict, args):

    t0 = time.time()
    print("process ...")
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygGraphPropPredDataset(name=args.dataset)
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    split_idx = dataset.get_idx_split()
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
    evaluator = Evaluator(args.dataset)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    node_encoder = ASTNodeEncoder(args.emb_dim, 
                                  num_nodetypes=len(nodetypes_mapping['type']), 
                                  num_nodeattributes=len(nodeattributes_mapping['attr']),
                                  max_depth=20,
                                  mask=True)
    node_encoder2 = ASTNodeEncoder(args.emb_dim, 
                                  num_nodetypes=len(nodetypes_mapping['type']), 
                                  num_nodeattributes=len(nodeattributes_mapping['attr']),
                                  max_depth=20,
                                  mask=True)

    model = GNN(num_vocab=len(vocab2idx), 
                max_seq_len=args.max_seq_len,
                node_encoder=node_encoder,
                num_layer=args.num_layer,
                gnn_type='gcn',
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio,
                virtual_node=False).to(device)
    
    masker = Masker(node_encoder=node_encoder2,
                   num_layer=args.num_layer,
                   gnn_type='gcn',
                   emb_dim=args.mask_dim,
                   drop_ratio=args.drop_ratio).to(device)

    
    if things_dict:

        train_loader_ori = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader_ori = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader_ori = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_dataset_pru = things_dict['train_dataset_pru']
        valid_dataset_pru = things_dict['valid_dataset_pru']
        test_dataset_pru = things_dict['test_dataset_pru']
        rewind_weight = things_dict['rewind_weight']
        rewind_weight2 = things_dict['rewind_weight2']
        model.load_state_dict(rewind_weight)
        masker.load_state_dict(rewind_weight2)
        model_mask_dict = things_dict['model_mask_dict']
        pruning.pruning_model_by_mask(model, model_mask_dict)

    else:
        things_dict = {}
        train_dataset_ori = [dataset[i][0] for i in split_idx["train"]]
        valid_dataset_ori = [dataset[i][0] for i in split_idx["valid"]]
        test_dataset_ori =  [dataset[i][0] for i in split_idx["test"]]
        train_loader_ori = DataLoader(train_dataset_ori, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader_ori = DataLoader(valid_dataset_ori, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader_ori = DataLoader(test_dataset_ori, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_dataset_pru = copy.deepcopy(train_dataset_ori)
        valid_dataset_pru = copy.deepcopy(valid_dataset_ori)
        test_dataset_pru = copy.deepcopy(test_dataset_ori)
        rewind_weight = copy.deepcopy(model.state_dict())
        rewind_weight2 = copy.deepcopy(masker.state_dict())

    train_loader_pru = DataLoader(train_dataset_pru, args.batch_size, shuffle=True)
    valid_loader_pru = DataLoader(valid_dataset_pru, args.batch_size, shuffle=False)
    test_loader_pru = DataLoader(test_dataset_pru, args.batch_size, shuffle=False)
    sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
    sp_valid = pruning.print_pruning_percent(valid_loader_ori, valid_loader_pru)
    sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
    spa = (sp_train + sp_valid + sp_test) / 3
    spw = pruning.see_zero_rate(model)

    t1 = time.time()
    print("done: time:[{:.2f}min]".format((t1 - t0) / 60))
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001 }, 
                            {'params': masker.parameters(),'lr': args.mask_lr}], weight_decay=0)

    update_epoch = update_test = best_val = 0
    # for epoch in range(1, args.epochs + 1):
    #     t0 = time.time()
    #     train_loss, mask_distribution = train_model_and_masker(model, masker, device, train_loader_pru, optimizer, epoch, args)
    #     # train_perf = eval_acc_with_mask(model, masker, device, train_dataset_pru, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
    #     valid_perf = eval_acc_with_mask(model, masker, device, valid_loader_pru, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
    #     test_perf = eval_acc_with_mask(model, masker, device, test_loader_pru, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
        
    #     if valid_perf > best_val:
    #         best_val = valid_perf
    #         update_epoch = epoch
    #         update_test = test_perf
            
    #     epoch_time = time.time() - t0
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
    #           "(syd: IMP (Train):[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]"
    #           .format(rp_num,
    #                   spa * 100,
    #                   spw * 100,
    #                   epoch,
    #                   args.epochs,
    #                   train_loss,
    #                   valid_perf * 100,
    #                   test_perf * 100,
    #                   update_test * 100,
    #                   update_epoch,
    #                   dis[0] * 100,
    #                   dis[1] * 100,
    #                   dis[2] * 100,
    #                   dis[3] * 100,
    #                   dis[4] * 100))

    pdb.set_trace()
    pruning.pruning_model(model, 0.2, random=False)
    _ = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)

    things_dict['rewind_weight'] = rewind_weight
    things_dict['rewind_weight2'] = rewind_weight2
    things_dict['model_mask_dict'] = model_mask_dict
    print("pruning dataset ...")
    t0 = time.time()
    train_dataset_pru = pruning.masker_pruning_dataset(train_dataset_pru, masker, args)
    valid_dataset_pru = pruning.masker_pruning_dataset(valid_dataset_pru, masker, args)
    test_dataset_pru = pruning.masker_pruning_dataset(test_dataset_pru, masker, args)
    t1 = time.time()
    print("done: time cost:{:.2f}min".format((t1 - t0)/60))
    things_dict['train_dataset_pru'] = train_dataset_pru
    things_dict['valid_dataset_pru'] = valid_dataset_pru
    things_dict['test_dataset_pru'] = test_dataset_pru

    return things_dict

    
def eval_tickets(imp_num, things_dict, args):

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygGraphPropPredDataset(name=args.dataset)
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])

    split_idx = dataset.get_idx_split()
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
    
    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
    evaluator = Evaluator(args.dataset)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
    model = GNN(num_vocab=len(vocab2idx), 
                max_seq_len=args.max_seq_len,
                node_encoder = node_encoder,
                num_layer=args.num_layer,
                gnn_type='gcn',
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio,
                virtual_node = False).to(device)
    
    train_loader_ori = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader_ori = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_ori = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_dataset_pru = things_dict['train_dataset_pru']
    valid_dataset_pru = things_dict['valid_dataset_pru']
    test_dataset_pru = things_dict['test_dataset_pru']
    rewind_weight = things_dict['rewind_weight']
    model_mask_dict = things_dict['model_mask_dict']

    model.load_state_dict(rewind_weight)
    pruning.pruning_model_by_mask(model, model_mask_dict)

    train_loader_pru = DataLoader(train_dataset_pru, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader_pru = DataLoader(valid_dataset_pru, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_pru = DataLoader(test_dataset_pru, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru)
    sp_val = pruning.print_pruning_percent(valid_loader_ori, valid_loader_pru)
    sp_test = pruning.print_pruning_percent(test_loader_ori, test_loader_pru)
    spa = (sp_train + sp_test + sp_val) / 3
    spw = pruning.see_zero_rate(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    update_epoch = update_test = best_val = 0
    for epoch in range(1, args.epochs + 1):

        t0 = time.time()
        train_loss = train(model, device, train_loader_pru, optimizer, epoch, args)
        # train_perf = eval(model, device, train_loader_pru, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
        valid_perf = eval(model, device, valid_loader_pru, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
        test_perf = eval(model, device, test_loader_pru, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
        
        if valid_perf > best_val:
            best_val = valid_perf
            update_epoch = epoch
            update_test = test_perf

        epoch_time = time.time() - t0
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              "(syd: IMP (Test):[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at Epoch:[{}] | Epoch Time:[{:.2f} min]"
              .format(rp_num,
                      spa * 100,
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

    print("(syd: IMP final:[{}] spa:[{:.2f}%] spw:[{:.2f}%])  Update Test:[{:.2f}] at Epoch:[{}]"
              .format(rp_num,
                      spa * 100,
                      spw * 100,
                      update_test * 100,
                      update_epoch
                      ))


if __name__ == "__main__":

    args = pruning.parser_loader().parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)
    things_dict = None
    save_dir = "impckpt/{}-lr-{}-dim-{}".format(args.dataset, args.mask_lr, args.mask_dim)
    for imp_num in range(1, 21):

        things_dict = train_get_mask(imp_num, things_dict, args)
        pruning.save_imp_things(things_dict, save_dir, "imp-train{}.pt".format(imp_num))
        eval_tickets(imp_num, things_dict, args)
        pruning.save_imp_things(things_dict, save_dir, "imp-eval{}.pt".format(imp_num))