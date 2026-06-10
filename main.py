import os
import torch
import torch.optim as optim
import numpy as np
import random
import argparse

from train import pretrain,  train
from data_loader import load_data, prepare_aligned_views
from configure import get_default_config
from model import MyPVC


parser = argparse.ArgumentParser(description='MyPVC')
parser.add_argument('--dataset', default='WebKB',
                                 choices=['WebKB', 'Caltech101-7', 'HandWritten', 'BDGP', 'Reuters'])
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
args = parser.parse_args()


def main():
    args = parser.parse_args()

    # Set Environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Get Configure
    config = get_default_config(args.dataset)
    config['dataset'] = args.dataset

    manual_seed = config['seed']
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    print(args.dataset)
    print('current seed {}'.format(manual_seed))
    print('aligned ratio {}'.format(config['aligned_ratio']))
    print('lambda_cons: ', config['lambda_cons'])
    print('k: ', config['k'])

    # Load and process data
    data, labels, A = load_data(args.dataset, config)
    X1, X2, A1, A2, aligned_idx, unaligned_idx, labels_0, labels_1 = prepare_aligned_views(data, labels, A, config['aligned_ratio'])

    config['num_class'] = len(np.unique(labels))
    config['num_sample'] = len(labels)

    # Initialize model and optimizer
    model = MyPVC(config)
    model.to(device)
    main_params = [p for p in model.parameters() if p not in set(model.got.parameters())]
    optimizer = optim.Adam([
        {'params': main_params, 'lr': config['lr']},  # 主参数组
        {'params': model.got.parameters(), 'lr': config['got']['lr']}  # got 参数组
    ])

    # Pre-training
    print('Starting pre-training...')
    pretrain(model, optimizer, config, X1, X2, A1, A2, labels_0, aligned_idx, unaligned_idx, device)

    # Training
    print('Starting training...')
    train(model, optimizer, X1, X2, A1, A2, labels_0, labels_1, aligned_idx, unaligned_idx, manual_seed, config, device)


if __name__ == '__main__':
    main()

