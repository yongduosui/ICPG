import torch
from model import GCNNet
from train import train, test
from args import parser_loader, get_dataset, print_args, setup_seed
import pruning
import pdb

def eval_random(args):

    dataset = get_dataset(args)
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = GCNNet(dataset, args).to(device), data.to(device)

    data = pruning.random_pruning_data(data, args)
    pruning.pruning_model(model, args.pruning_wei, random=True)

    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=5e-4)
    ], lr=args.lr)  

    best_val_acc = best_test_acc = best_epoch = 0
    for epoch in range(1, args.total_epoch + 1):

        train_loss = train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_epoch = epoch
        if epoch % 50 == 0:
            log = 'Epoch:{:03d}/{} | Train:[{:.4f}], Val:[{:.4f}], Test:[{:.4f}] | Best Val:[{:.4f}] Update Test:[{:.4f}] at Epoch:[{}]'
            print(log.format(epoch,
                            args.total_epoch,
                            train_acc, 
                            val_acc, 
                            tmp_test_acc, 
                            best_val_acc, 
                            best_test_acc,
                            best_epoch))

    print("syd: final dataset[{}]  random spa[{:.2f}%] spw[{:.2f}%] Update Test:[{:.1f}] at Epoch:[{}]".format( 
                         args.dataset,
                         args.pruning_adj * 100,
                         args.pruning_wei * 100,
                         best_test_acc * 100,
                         best_epoch))

def main():

    args = parser_loader().parse_args()
    print_args(args)
    setup_seed(args)
    
    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1)) for i in range(20)]
    for pa, pw in percent_list:
        args.pruning_adj = pa
        args.pruning_wei = pw
        eval_random(args)


if __name__ == '__main__':
    main()