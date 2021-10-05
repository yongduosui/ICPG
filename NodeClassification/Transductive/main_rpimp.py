import torch
from model import GCNNet
from train import train, test
from args import parser_loader, get_dataset, print_args, setup_seed
import pruning
import copy
import pdb


def train_imp(args, imp_num, data, rewind_weight=None, model_mask_dict=None):

    spa = 1 - float(data.num_edges) / data.num_edges_orig
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = GCNNet(data, args).to(device),  data.to(device)
    
    if rewind_weight is not None:
        model.load_state_dict(rewind_weight)
        pruning.pruning_model_by_mask(model, model_mask_dict)
        spw = pruning.see_zero_rate(model)
    else:
        rewind_weight = copy.deepcopy(model.state_dict())
        spw = 0.0

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
    best_val_acc = best_test_acc = best_epoch = 0
    for epoch in range(1, args.total_epoch + 1):
        
        train_loss = train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0:
            
            log = 'RPIMP:[{}] | spa[{:.2f}%] spw[{:.2f}%] |Epoch:{:03d}/{} | Loss:[{:.4f}] Train:[{:.4f}], Val:[{:.4f}], Test:[{:.4f}] | Best Val:[{:.4f}] Update Test:[{:.4f}] at Epoch:[{}]'
            print(log.format(imp_num,
                            spa * 100,
                            spw * 100,
                            epoch,
                            args.total_epoch,
                            train_loss,
                            train_acc, 
                            val_acc, 
                            tmp_test_acc, 
                            best_val_acc, 
                            best_test_acc,
                            best_epoch))

    model.load_state_dict(best_model_state_dict)
    pruning.pruning_model(model, 0.2, random=False)
    spw = pruning.see_zero_rate(model)
    model_mask_dict = pruning.extract_mask(model)

    print("syd: final dataset[{}] RPIMP [{}] spa[{:.2f}%] spw[{:.2f}%]  Update Test:[{:.1f}] at Epoch:[{}]"
                        .format( 
                         args.dataset,
                         imp_num,
                         spa * 100,
                         spw * 100,
                         best_test_acc * 100,
                         best_epoch))

    return rewind_weight, model_mask_dict


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

    rewind_weight = model_mask_dict = None
    args.pruning_adj = 0.05
    for imp_num in range(1, 21):
        
        rewind_weight, model_mask_dict = train_imp(args, imp_num, data, rewind_weight, model_mask_dict)
        data = pruning.random_pruning_data(data, args)


if __name__ == '__main__':
    main()