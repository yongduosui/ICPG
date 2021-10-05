import os.path as osp
import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from train import train, test
from model import GATNet
import copy
import pruning
import pdb

def main(rp_imp, pa, pw, things_dict):

    total_epoch = 100
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset_ori = PPI(path, split='train')
    val_dataset_ori = PPI(path, split='val')
    test_dataset_ori = PPI(path, split='test')
    
    train_dataset = pruning.random_pruning_dataset(train_dataset_ori, pa)
    val_dataset = pruning.random_pruning_dataset(val_dataset_ori, pa)
    test_dataset = pruning.random_pruning_dataset(test_dataset_ori, pa)

    sp_train = pruning.print_dataset_sparsity(train_dataset_ori, train_dataset)
    sp_val = pruning.print_dataset_sparsity(val_dataset_ori, val_dataset)
    sp_test = pruning.print_dataset_sparsity(test_dataset_ori, test_dataset)
    spa = (sp_train + sp_val + sp_test) / 3.0

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet(train_dataset_ori).to(device)
    
    if things_dict is not None:

        rewind_weight = things_dict['rewind_weight']
        model_mask_dict = things_dict['model_mask_dict']
        model.load_state_dict(rewind_weight)
        pruning.pruning_model_by_mask(model, model_mask_dict)
        spw = pruning.see_zero_rate(model)
    else:
        things_dict = {}
        rewind_weight = copy.deepcopy(model.state_dict())
        spw = 0.0

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

        print('RPIMP:[{}] Epoch:[{}/{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}] | spa:[{:.2f}%] spw:[{:.2f}%]'
                    .format(
                        rp_imp,
                        epoch, 
                        total_epoch,
                        loss, 
                        val_f1 * 100, 
                        test_f1 * 100,
                        update_test * 100,
                        update_epoch,
                        100 * spa,
                        100 * spw))

    print("syd final: RPIMP:[{}] Update Test:[{:.2f}] at epoch:[{}] | spa:[{:.2f}%] spw:[{:.2f}%]"
            .format(rp_imp,
                    update_test * 100,
                    update_epoch,
                    100 * spa,
                    100 * spw))

    pruning.pruning_model(model, 0.2, random=False)
    _ = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)
    things_dict['rewind_weight'] = rewind_weight
    things_dict['model_mask_dict'] = model_mask_dict
    return things_dict


if __name__ == '__main__':

    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1)) for i in range(20)]
    things_dict = main(0, 0, None, None)
    for rp_imp, (pa, pw) in enumerate(percent_list):
        things_dict = main(rp_imp + 1, pa, None, things_dict)