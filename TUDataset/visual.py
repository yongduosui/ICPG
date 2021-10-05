import pruning
import torch

import argparse
from utils import logger
from datasets import get_dataset
import random
import copy
import pdb

def main():
    path = "./impckpt/COLLAB-lr-0.0005-dim-64"
    file1 = path + "/imp-train1.pt"
    file2 = path + "/imp-train2.pt"
    file3 = path + "/imp-train3.pt"

    train_dataset_pru1 = torch.load(file1)['1']['train_dataset_pru']
    train_dataset_pru2 = torch.load(file2)['1']['train_dataset_pru']
    train_dataset_pru3 = torch.load(file3)['1']['train_dataset_pru']
    
    pdb.set_trace()
    sp1 = pruning.print_pruning_percent(train_dataset_pru1, train_dataset_pru2)
    sp2 = pruning.print_pruning_percent(train_dataset_pru1, train_dataset_pru3)
    








if __name__ == '__main__':
    main()