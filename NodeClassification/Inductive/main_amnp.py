import os.path as osp
import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from train import train, test, train_model_and_masker, eval_acc_with_mask
from model import GATNet, Masker
import copy
import pruning
import pdb

def train_get_mask(imp_num, things_dict):

    total_epoch = 100
    masker_lr = 0.001
    hidden = 128

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset_ori = PPI(path, split='train')
    val_dataset_ori = PPI(path, split='val')
    test_dataset_ori = PPI(path, split='test')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet(train_dataset_ori).to(device)
    masker = Masker(train_dataset_ori, hidden=hidden).to(device)

    if things_dict is not None:
        train_dataset_pru = things_dict['train_dataset_pru'] 
        val_dataset_pru = things_dict['val_dataset_pru'] 
        test_dataset_pru = things_dict['test_dataset_pru'] 
        rewind_weight = things_dict['rewind_weight']
        rewind_weight2 = things_dict['rewind_weight2']
        model.load_state_dict(rewind_weight)
        masker.load_state_dict(rewind_weight2)

    else:
        things_dict = {}
        train_dataset_pru = copy.deepcopy(train_dataset_ori)
        val_dataset_pru = copy.deepcopy(val_dataset_ori)
        test_dataset_pru = copy.deepcopy(test_dataset_ori)
        rewind_weight = copy.deepcopy(model.state_dict())
        rewind_weight2 = copy.deepcopy(masker.state_dict())

    sp_train = pruning.print_dataset_sparsity(train_dataset_ori, train_dataset_pru)
    sp_val = pruning.print_dataset_sparsity(val_dataset_ori, val_dataset_pru)
    sp_test = pruning.print_dataset_sparsity(test_dataset_ori, test_dataset_pru)
    spa = (sp_train + sp_val + sp_test) / 3.0
    spw = pruning.see_zero_rate(model)

    train_loader = DataLoader(train_dataset_pru, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset_pru, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset_pru, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.005},
                                  {'params': masker.parameters(), 'lr': masker_lr}])

    best_val = update_test = update_epoch = 0
    for epoch in range(1, total_epoch + 1):
        loss, mask_distribution = train_model_and_masker(model, masker, optimizer, train_loader)
        val_f1 = eval_acc_with_mask(model, masker, val_loader)
        test_f1 = eval_acc_with_mask(model, masker, test_loader)
        if val_f1 > best_val:
            best_val = val_f1
            update_test = test_f1
            update_epoch = epoch
            best_masker_state_dict = copy.deepcopy(masker.state_dict())

        print('AMNP [{}] (Train spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]'
                    .format(
                        imp_num,
                        100 * spa,
                        100 * spw,
                        epoch, 
                        total_epoch,
                        loss, 
                        val_f1 * 100, 
                        test_f1 * 100,
                        update_test * 100,
                        update_epoch,
                        mask_distribution[0] * 100,
                        mask_distribution[1] * 100,
                        mask_distribution[2] * 100,
                        mask_distribution[3] * 100,
                        mask_distribution[4] * 100))

    masker.load_state_dict(best_masker_state_dict)
    pruning.grad_model(masker, False)
    train_dataset_pru = pruning.masker_pruning_dataset(train_dataset_pru, masker, 1, 0.05)
    val_dataset_pru = pruning.masker_pruning_dataset(val_dataset_pru, masker, 2, 0.05)
    test_dataset_pru = pruning.masker_pruning_dataset(test_dataset_pru, masker, 2, 0.05)

    sp_train = pruning.print_dataset_sparsity(train_dataset_ori, train_dataset_pru)
    sp_val = pruning.print_dataset_sparsity(val_dataset_ori, val_dataset_pru)
    sp_test = pruning.print_dataset_sparsity(test_dataset_ori, test_dataset_pru)
    spa = (sp_train + sp_val + sp_test) / 3.0
    spw = pruning.see_zero_rate(model)

    # print("Graph Sparsity:[{:.2f}%] ".format(spa * 100))
    things_dict['train_dataset_pru'] = train_dataset_pru 
    things_dict['val_dataset_pru'] = val_dataset_pru 
    things_dict['test_dataset_pru'] = test_dataset_pru 
    things_dict['rewind_weight'] = rewind_weight
    things_dict['rewind_weight2'] = rewind_weight2

    return things_dict

def eval_tickets(imp_num, things_dict):

    total_epoch = 100
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset_ori = PPI(path, split='train')
    val_dataset_ori = PPI(path, split='val')
    test_dataset_ori = PPI(path, split='test')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet(train_dataset_ori).to(device)
    
    train_dataset_pru = things_dict['train_dataset_pru'] 
    val_dataset_pru = things_dict['val_dataset_pru'] 
    test_dataset_pru = things_dict['test_dataset_pru'] 
    rewind_weight = things_dict['rewind_weight']
    model.load_state_dict(rewind_weight)

    sp_train = pruning.print_dataset_sparsity(train_dataset_ori, train_dataset_pru)
    sp_val = pruning.print_dataset_sparsity(val_dataset_ori, val_dataset_pru)
    sp_test = pruning.print_dataset_sparsity(test_dataset_ori, test_dataset_pru)
    spa = (sp_train + sp_val + sp_test) / 3.0
    spw = pruning.see_zero_rate(model)

    train_loader = DataLoader(train_dataset_pru, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset_pru, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset_pru, batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    best_val = update_test = update_epoch = 0
    for epoch in range(1, total_epoch + 1):
        loss = train(model, optimizer, train_loader)
        val_f1 = test(model, val_loader)
        test_f1 = test(model, test_loader)
        if val_f1 > best_val:
            best_val = val_f1
            update_test = test_f1
            update_epoch = epoch

        print('AMNP:[{}] (Test spa:[{:.2f}%] spw:[{:.2f}%]) Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}]'
                    .format(
                        imp_num,
                        100 * spa,
                        100 * spw,
                        epoch, 
                        total_epoch,
                        loss, 
                        val_f1 * 100, 
                        test_f1 * 100,
                        update_test * 100,
                        update_epoch))

    print("sydfinal: AMNP:[{}] (Test spa:[{:.2f}%] spw:[{:.2f}%]) Update Test:[{:.2f}] at epoch:[{}]"
            .format(imp_num,
                    100 * spa,
                    100 * spw,
                    update_test * 100,
                    update_epoch))

    
if __name__ == '__main__':

    save_dir = 'debug'
    things_dict = None
    for imp_num in range(1, 21):
        file_name = "amnp{}.pt".format(imp_num)
        things_dict = train_get_mask(imp_num, things_dict)
        # pruning.save_model(things_dict, save_dir, file_name)
        eval_tickets(imp_num, things_dict)