import itertools

import torch
import torch.nn as nn

import pdb

class FrameConstructor(nn.Module):
    def __init__(self):
        super(FrameConstructor, self).__init__()

    def forward(self, coeffs, timestamps):
        # coeffs: [bs, n_deg+1, h, w]
        # timestamps: [bs, n_ts, h, w] or [bs, n_ts]
        n_deg = coeffs.shape[1] - 1
        n_ts = timestamps.shape[1]
        if len(timestamps.shape) == 2:
            timestamps = timestamps.unsqueeze(-1).unsqueeze(-1)
        # bases: [bs, n_deg+1, n_ts, h, w]
        bases = torch.stack([timestamps ** i for i in range(n_deg + 1)], dim=1)
        recon = coeffs.unsqueeze(2) * bases
        recon = torch.sum(recon, dim=1)
        return recon

class CurveIntegrator(nn.Module):
    def __init__(self):
        super(CurveIntegrator, self).__init__()

    def forward(self, derivative, blurry, keypoints, integrator_cache=None):
        # coeffs: [bs, n_kpts, h, w]
        # blurry: [bs, 1, h, w]
        # keypoints: [bs, n_kpts, h, w]
        bs, n_kpts, h, w = keypoints.shape
        n_deg = n_kpts # degree of the primitive signal
        if integrator_cache is None:
            integrator_cache = torch.zeros((bs, n_deg, n_kpts, h, w),
                                           device=keypoints.device)
            for i in range(n_kpts):
                denominator = torch.ones_like(blurry).squeeze(dim=1)
                for j in range(n_kpts):
                    if i != j:
                        denominator *= (keypoints[:, i] - keypoints[:, j])
                indices_wo_i = [j for j in range(n_kpts) if j != i]
                for order in range(n_kpts):
                    # the minimum order is 0
                    # the maximum order is n_kpts-1
                    constant_count = n_kpts - 1 - order
                    for indices in itertools.combinations(indices_wo_i, constant_count):
                        term = torch.prod(-keypoints[:, list(indices)], dim=1)
                        integrator_cache[:, order, i] += term / (order + 1) / denominator
        coeffs = torch.sum(integrator_cache * derivative[:, None], dim=2)
        integral = 2 * coeffs[:, 1] / 3
        for i in range(3, n_deg, 2):
            integral = integral + 2 * coeffs[:, i] / (i + 2)
        baseline = (2 * blurry[:, 0] - integral) / 2
        baseline = baseline.unsqueeze(dim=1)
        coeffs = torch.cat([baseline, coeffs], dim=1) # [bs, n_deg+1, h, w]
        return coeffs, integrator_cache
