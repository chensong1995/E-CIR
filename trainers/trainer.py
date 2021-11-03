import os
import time

import cv2
import numpy as np
import skimage.metrics
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.losses import WeightedL1Loss
from lib.modules import FrameConstructor, CurveIntegrator
from lib.utils import save_session, AverageMeter

import pdb

cuda = torch.cuda.is_available()

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, test_loader, args):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.criteria = self.setup_criteria()
        self.constructor = FrameConstructor()
        self.integrator = CurveIntegrator()
        if cuda:
            for name in self.criteria:
                self.criteria[name] = nn.DataParallel(self.criteria[name]).cuda()
            self.constructor = nn.DataParallel(self.constructor).cuda()
            self.integrator = nn.DataParallel(self.integrator).cuda()
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'log'))
        self.step = {
                'train': 0,
                'validate': 0
                }

    def setup_criteria(self):
        criteria = {}
        criteria['der'] = torch.nn.L1Loss()
        criteria['pri'] = torch.nn.L1Loss()
        criteria['ref'] = torch.nn.L1Loss()
        criteria['res'] = WeightedL1Loss()
        return criteria

    def setup_records(self):
        records = {}
        records['time'] = AverageMeter()
        records['total'] = AverageMeter()
        return records

    def compute_der_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_der = self.criteria['der'](pred['derivative'],
                                        batch['derivative']).mean()
        self.writer.add_scalar(stage + '/derivative', loss_der, step)
        loss = self.args.lambda_der * loss_der
        return loss

    def compute_pri_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_pri = self.criteria['pri'](pred['primitive'],
                                        batch['primitive']).mean()
        self.writer.add_scalar(stage + '/primitive', loss_pri, step)
        loss = self.args.lambda_pri * loss_pri
        return loss

    def compute_ref_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_ref = self.criteria['ref'](pred['frame_refine'],
                                        batch['sharp_frame']).mean()
        self.writer.add_scalar(stage + '/refine', loss_ref, step)
        loss = self.args.lambda_ref * loss_ref
        return loss

    def compute_res_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_res = 0.
        for i in range(batch['sharp_frame'].shape[1] - 1):
            res_pred = pred['residual'][:, :, i]
            res_gt = batch['sharp_frame'][:, i + 1] - batch['sharp_frame'][:, i]
            res_gt = torch.stack([res_gt] * self.args.refine_iters, dim=1)
            loss_res = loss_res + self.criteria['res'](res_pred, res_gt).mean()
        self.writer.add_scalar(stage + '/residual', loss_res, step)
        loss = self.args.lambda_res * loss_res
        return loss

    def compute_losses(self, pred, batch, records, stage='train'):
        loss = 0.
        batch_size = batch['blurry_frame'].shape[0]
        info = []
        if self.args.lambda_der > 0:
            loss = loss + self.compute_der_loss(pred, batch, stage)
        if self.args.lambda_pri > 0:
            loss = loss + self.compute_pri_loss(pred, batch, stage)
        if self.args.lambda_ref > 0:
            loss = loss + self.compute_ref_loss(pred, batch, stage)
        if self.args.lambda_res > 0:
            loss = loss + self.compute_res_loss(pred, batch, stage)
        records['total'].update(loss.detach().cpu().numpy(), batch_size)
        info.append('Total: {:.3f} ({:.3f})'.format(records['total'].val,
                                                    records['total'].avg))
        info = '\t'.join(info)
        step = self.step[stage]
        self.writer.add_scalar(stage + '/total', loss, step)
        self.writer.flush()
        self.step[stage] += 1
        return loss, info

    def predict(self, batch):
        pred = self.model['g'](batch['blurry_frame'],
                               batch['event_map'],
                               batch['keypoints'])
        use_integrator = self.args.lambda_pri > 0 or \
                         self.args.lambda_ref > 0 or \
                         self.args.lambda_res > 0
        if use_integrator:
            # take the indefinite integral and reconstruct the primitive signal
            # under the standard bases
            pred['coeffs'], pred['integrator_cache'] = \
                    self.integrator(pred['derivative'],
                                    batch['blurry_frame'],
                                    batch['keypoints'])
            if self.args.lambda_pri > 0:
                # evaluate the primitive signal value at keypoints
                pred['primitive'] = self.constructor(pred['coeffs'],
                                                     batch['keypoints'])
            if self.args.lambda_ref > 0 or self.args.lambda_res > 0:
                # extract frames and spatial derivatives (edges) at timestamps
                # corresponding to ground-truth sharp frames
                pred['frame_init'] = self.constructor(pred['coeffs'],
                                                      batch['timestamps'])
                pred['frame_refine'], pred['residual'] = \
                        self.model['r'](pred['frame_init'], batch['event_map'])
                batch['coeffs'], _ = self.integrator(batch['derivative'],
                                                     batch['blurry_frame'],
                                                     batch['keypoints'],
                                                     pred['integrator_cache'])
                batch['frame'] = self.constructor(batch['coeffs'],
                                                  batch['timestamps'])
        return pred

    def train(self, epoch):
        for key in self.model.keys():
            self.model[key].train()
        records = self.setup_records()
        num_iters = min(len(self.train_loader), self.args.iters_per_epoch)
        for i_batch, batch in enumerate(self.train_loader):
            start_time = time.time()
            if i_batch >= self.args.iters_per_epoch:
                break
            pred = self.predict(batch)
            loss, loss_info = self.compute_losses(pred,
                                                  batch,
                                                  records,
                                                  stage='train')

            self.optimizer['g'].zero_grad()
            loss.backward()
            self.optimizer['g'].step()

            # print information during training
            records['time'].update(time.time() - start_time)
            info = 'Epoch: [{}][{}/{}]\t' \
                   'Time: {:.3f} ({:.3f})\t{}'.format(epoch,
                                                      i_batch,
                                                      num_iters,
                                                      records['time'].val,
                                                      records['time'].avg,
                                                      loss_info)
            print(info)

    def test(self):
        print('Testing on REDS')
        for key in self.model.keys():
            self.model[key].eval()
        metrics = {}
        for metric_name in ['MSE', 'PSNR', 'SSIM']:
            metrics[metric_name] = AverageMeter()
        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(self.test_loader)):
                pred = self.model['g'](batch['blurry_frame'],
                                       batch['event_map'],
                                       batch['keypoints'])
                pred['coeffs'], _ = self.integrator(pred['derivative'],
                                                    batch['blurry_frame'],
                                                    batch['keypoints'])
                pred['frame_init'] = self.constructor(pred['coeffs'],
                                                      batch['timestamps'])
                frame = pred['frame_init']
                if self.args.lambda_ref > 0:
                    pred['frame_refine'], _ = self.model['r'](pred['frame_init'],
                                                              batch['event_map'])
                    frame = pred['frame_refine']
                frame = np.clip(frame.detach().cpu().numpy(), 0, 1)
                video_idx = batch['video_idx'].detach().cpu().numpy()
                frame_idx = batch['frame_idx'].detach().cpu().numpy()
                frame_gt = batch['sharp_frame'].detach().cpu().numpy()
                for i_example in range(frame.shape[0]):
                    save_dir = os.path.join(self.args.save_dir,
                                            'reds_output',
                                            '{:03d}'.format(video_idx[i_example]))
                    os.makedirs(save_dir, exist_ok=True)
                    for i_time in range(frame.shape[1]):
                        save_name = os.path.join(save_dir,
                                                '{:06d}_{}.png'.format(frame_idx[i_example],
                                                                       i_time))
                        cv2.imwrite(save_name, frame[i_example, i_time] * 255)
                        gt = np.uint8(frame_gt[i_example, i_time] * 255)
                        pred = np.uint8(frame[i_example, i_time] * 255)
                        for metric_name, metric in zip(['MSE', 'PSNR', 'SSIM'],
                                                       [skimage.metrics.normalized_root_mse,
                                                        skimage.metrics.peak_signal_noise_ratio,
                                                        skimage.metrics.structural_similarity]):
                            metrics[metric_name].update(metric(gt, pred))
        info = 'MSE: {:.3f}\tPSNR: {:.3f}\tSSIM: {:.3f}'.format(metrics['MSE'].avg,
                                                                metrics['PSNR'].avg,
                                                                metrics['SSIM'].avg)
        print('Results:')
        print(info)

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.save_dir, 'checkpoints')
        save_session(self.model, self.optimizer, ckpt_dir, epoch)
