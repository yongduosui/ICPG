import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from utils import ASTNodeEncoder, get_vocab_mapping
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq
from train import train, eval
import pdb
import pruning

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code2 data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=666, help='seed')
    parser.add_argument('--drop_ratio', type=float, default=0, help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5, help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000, help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 25)')
    parser.add_argument('--random_split', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code2", help='dataset name (default: ogbg-code2)')
    parser.add_argument('--save_dir', type=str, default="baseline_ckpt", help='dataset name (default: ogbg-code2)')

    args = parser.parse_args()
    pruning.print_args(args)
    pruning.setup_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygGraphPropPredDataset(name=args.dataset)
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)))

    split_idx = dataset.get_idx_split()
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    print(nodeattributes_mapping)

    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
    model = GNN(num_vocab=len(vocab2idx), 
                max_seq_len=args.max_seq_len,
                node_encoder = node_encoder,
                num_layer=args.num_layer,
                gnn_type='gcn',
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio,
                virtual_node = False).to(device)
    pruning.save_model(model, args.save_dir, "epoch0_weight.pt")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    valid_curve = []
    test_curve = []
    train_curve = []
    update_epoch = update_test = best_val = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss = train(model, device, train_loader, optimizer, epoch, args)
        train_perf = eval(model, device, train_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
        valid_perf = eval(model, device, valid_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
        test_perf = eval(model, device, test_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab), epoch=epoch, args=args)
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