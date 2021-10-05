import torch
from model import GCNNet, GCNmasker
from train import train_model_and_masker, test_masker, train, test
from args import parser_loader, get_dataset, print_args, setup_seed
import pruning
import copy
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_masker(args, imp_num, data, rewind_weight=None, model_mask_dict=None):

    model, masker, data = GCNNet(data, args).to(device), GCNmasker(data, args).to(device), data.to(device)
    if rewind_weight is not None:
        model.load_state_dict(rewind_weight)
        pruning.pruning_model_by_mask(model, model_mask_dict)
        _ = pruning.see_zero_rate(model)
    else:
        rewind_weight = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': 5e-4}, 
                                  {'params': masker.parameters(),'lr': args.mask_lr, 'weight_decay': args.mask_wd}])

    best_val_acc = best_test_acc = best_epoch = 0
    for epoch in range(1, args.total_epoch + 1):
        
        train_loss, mask_distribution = train_model_and_masker(model, masker, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test_masker(model, masker, data, args, pruned=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            
            log = 'IMP:[{}] (Train) Epoch:{:03d}/{} | Train:[{:.4f}], Val:[{:.4f}], Test:[{:.4f}] | Best Val:[{:.4f}] Update Test:[{:.4f}] at Epoch:[{}] | 0-0.2:[{:.2f}%] 0.2-0.4:[{:.2f}%] 0.4-0.6:[{:.2f}%] 0.6-0.8:[{:.2f}%] 0.8-1.0:[{:.2f}%]'
            print(log.format(imp_num,
                            epoch,
                            args.total_epoch,
                            train_acc, 
                            val_acc, 
                            tmp_test_acc, 
                            best_val_acc, 
                            best_test_acc,
                            best_epoch,
                            mask_distribution[0] * 100,
                            mask_distribution[1] * 100,
                            mask_distribution[2] * 100,
                            mask_distribution[3] * 100,
                            mask_distribution[4] * 100))

    model.load_state_dict(best_model_state_dict)
    pruning.pruning_model(model, 0.2, random=False)
    _ = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)
    return masker, rewind_weight, model_mask_dict


def eval_masker(args, imp_num, data, rewind_weight, model_mask_dict):

    sp_adj = 1 - float(data.num_edges) / data.num_edges_orig
    model, data = GCNNet(data, args).to(device), data.to(device)
    model.load_state_dict(rewind_weight)
    pruning.pruning_model_by_mask(model, model_mask_dict)
    sp_wei = pruning.see_zero_rate(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  
    best_val_acc = best_test_acc = best_epoch = 0
    for epoch in range(1, args.total_epoch + 1):

        train_loss = train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_epoch = epoch

        if epoch % 20 == 0:
            log = 'IMP:[{}] (Test) Epoch:{:03d}/{} | Train:[{:.4f}], Val:[{:.4f}], Test:[{:.4f}] | Best Val:[{:.4f}] Update Test:[{:.4f}] at Epoch:[{}] | SPA:[{:.2f}%] SPW:[{:.2f}%] | DSA:[{:.2f}%] DSW:[{:.2f}%]'
            print(log.format(
                            imp_num,
                            epoch,
                            args.total_epoch,
                            train_acc, 
                            val_acc, 
                            tmp_test_acc, 
                            best_val_acc, 
                            best_test_acc,
                            best_epoch,
                            sp_adj * 100,
                            sp_wei * 100,
                            (1 - sp_adj) * 100,
                            (1 - sp_wei) * 100))
    print("-" * 150)
    print("sydfinal: IMP:[{}] (Test) dataset[{}] Update Test:[{:.1f}] at Epoch:[{}] | SPA:[{:.2f}%] SPW:[{:.2f}%] | dim:[{}] lr:[{:.6f}] wd:[{:.6f}]"
                        .format( 
                         imp_num,
                         args.dataset,
                         best_test_acc * 100,
                         best_epoch,
                         sp_adj * 100,
                         sp_wei * 100,
                         args.masker_dim,
                         args.mask_lr,
                         args.mask_wd))
    print("-" * 150)

def main():

    args = parser_loader().parse_args()
    print_args(args)
    setup_seed(args)
    rewind_weight = None
    model_mask_dict = None
    dataset = get_dataset(args)
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.num_edges_orig = data.num_edges

    for imp in range(1, 21):
        
        masker, rewind_weight, model_mask_dict = train_masker(args, imp, data, rewind_weight, model_mask_dict)
        data = pruning.pruning_data_by_masker(masker, data, args)
        eval_masker(args, imp, data, rewind_weight, model_mask_dict)
        
if __name__ == '__main__':
    main()