import _init_paths
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lib.datasets import EDIDataset
from lib.models import UNet, RefineNet
from lib.modules import FrameConstructor, CurveIntegrator
import pdb

cuda = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--edi_data_dir', type=str, default='data/EDI/hdf5')
    parser.add_argument('--save_dir', type=str, default='saved_weights/debug')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--n_kpts', type=int, default=10)
    parser.add_argument('--refine_iters', type=int, default=3)
    parser.add_argument('--use_events', type=int, default=1)
    parser.add_argument('--use_blurry', type=int, default=1)
    args = parser.parse_args()
    return args

def setup_model(args):
    model, optimizer = {}, {}
    # build generator
    model['g'] = UNet(use_blurry=bool(args.use_blurry),
                      use_events=bool(args.use_events),
                      out_channels=args.n_kpts)
    if args.refine_iters > 0:
        model['r'] = RefineNet(refine_iters=args.refine_iters)
    constructor = FrameConstructor()
    integrator = CurveIntegrator()
    # move models to cuda devices
    if cuda:
        for key in model.keys():
            model[key] = nn.DataParallel(model[key]).cuda()
        constructor = nn.DataParallel(constructor).cuda()
        integrator = nn.DataParallel(integrator).cuda()
        constructor.eval()
        integrator.eval()
    if args.load_dir is not None:
        for key in model.keys():
            load_name = os.path.join(args.load_dir, 'model_{}.pth'.format(key))
            model[key].load_state_dict(torch.load(load_name))
            model[key].eval()
    return model, constructor, integrator

def setup_loader(args):
    loader = {}
    for name in ['camerashake1', 'cvprtext', 'indoordrop', 'indoorrenjump',
                 'ren_more2', 'rotatevideonew2_6', 'textlowlight']:
        hdf5_name = os.path.join(args.edi_data_dir, name + '.hdf5')
        dataset = EDIDataset(hdf5_name=hdf5_name)
        loader[name] = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
    return loader

def main():
    args = parse_args()
    model, constructor, integrator = setup_model(args)
    loader = setup_loader(args)
    with torch.no_grad():
        for name in loader.keys():
            print('testing on', name)
            output_dir = os.path.join(args.save_dir, 'edi_output', name)
            os.makedirs(output_dir, exist_ok=True)
            for batch in tqdm(loader[name]):
                pred = model['g'](batch['blurry_frame'],
                                  batch['event_map_small'],
                                  batch['keypoints'])
                pred['coeffs'], _ = integrator(pred['derivative'],
                                               batch['blurry_frame'],
                                               batch['keypoints'])
                pred['frame_init'] = constructor(pred['coeffs'],
                                                 batch['timestamps'])
                frame = pred['frame_init']
                if args.refine_iters > 0:
                    pred['frame_refine'], _ = model['r'](pred['frame_init'],
                                                         batch['event_map_big'])
                    frame = pred['frame_refine']
                frame = np.clip(frame.detach().cpu().numpy(), 0, 1)
                frame_idx = batch['frame_idx'].detach().cpu().numpy()
                batch_size = frame.shape[0]
                for i in range(batch_size):
                    for t_idx in range(len(batch['timestamps'][i])):
                        save_name = os.path.join(output_dir,
                                                 '{:06d}_{}.png'.format(frame_idx[i] + 1,
                                                                        t_idx))
                        cv2.imwrite(save_name, frame[i][t_idx] * 255)

if __name__ == '__main__':
    main()
