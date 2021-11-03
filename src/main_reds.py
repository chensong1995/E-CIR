import _init_paths
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import lib
from lib.datasets import REDSDataset
from lib.models import UNet, RefineNet
from lib.utils import *
from trainers.trainer import Trainer

import pdb

cuda = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--iters_per_epoch', type=int, default=10000)
    parser.add_argument('--reds_data_dir', type=str, default='data/REDS')
    parser.add_argument('--save_dir', type=str, default='saved_weights/debug')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_kpts', type=int, default=10)
    parser.add_argument('--lambda_der', type=float, default=1)
    parser.add_argument('--lambda_pri', type=float, default=10)
    parser.add_argument('--lambda_ref', type=float, default=10)
    parser.add_argument('--lambda_res', type=float, default=0.5)
    parser.add_argument('--refine_iters', type=int, default=3)
    parser.add_argument('--use_events', type=int, default=1)
    parser.add_argument('--use_blurry', type=int, default=1)
    args = parser.parse_args()
    return args

def initialize(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_loaders(args):
    train_set, test_set = [], []
    for i in range(16):
        hdf5_name = os.path.join(args.reds_data_dir, 'train_{}.hdf5'.format(i))
        dataset = REDSDataset(hdf5_name=hdf5_name)
        train_set.append(dataset)
    for i in range(2):
        hdf5_name = os.path.join(args.reds_data_dir, 'val_{}.hdf5'.format(i))
        dataset = REDSDataset(hdf5_name=hdf5_name)
        test_set.append(dataset)
    train_set = torch.utils.data.ConcatDataset(train_set)
    test_set = torch.utils.data.ConcatDataset(test_set)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               shuffle=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              num_workers=8,
                                              shuffle=False)
    return train_loader, test_loader

def setup_model(args):
    model, optimizer = {}, {}
    # build generator
    model['g'] = UNet(use_blurry=bool(args.use_blurry),
                      use_events=bool(args.use_events),
                      out_channels=args.n_kpts)
    if args.lambda_ref > 0:
        model['r'] = RefineNet(refine_iters=args.refine_iters)
    # move models to cuda devices
    if cuda:
        for key in model.keys():
            model[key] = nn.DataParallel(model[key]).cuda()
    # build optimizators
    if args.lambda_ref > 0:
        g_parameters = list(model['g'].parameters()) + \
                       list(model['r'].parameters())
    else:
        g_parameters = model['g'].parameters()
    optimizer['g'] = optim.Adam(g_parameters, lr=args.lr)
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_session(model, optimizer, args)
    else:
        start_epoch = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer['g'],
                                               milestones=[20, 40],
                                               gamma=0.5)
    return model, optimizer, start_epoch, scheduler

# main function
if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    train_loader, test_loader = setup_loaders(args)
    model, optimizer, start_epoch, scheduler = setup_model(args)
    trainer = Trainer(model, optimizer, train_loader, test_loader, args)
    #trainer.test()
    for epoch in range(start_epoch, args.n_epochs):
        trainer.train(epoch)
        if (epoch + 1) % args.save_every == 0:
            trainer.save_model(epoch)
        scheduler.step()
    trainer.test()
