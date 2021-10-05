import os.path as osp
import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from train import train, test
from model import GATNet
import pdb

def main():

    total_epoch = 100
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet(train_dataset).to(device)
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

        print('Epoch:[{}/{}], Loss:[{:.4f}], Val:[{:.2f}], Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}]'
                    .format(
                        epoch, 
                        total_epoch,
                        loss, 
                        val_f1 * 100, 
                        test_f1 * 100,
                        update_test * 100,
                        update_epoch))

if __name__ == '__main__':
    main()