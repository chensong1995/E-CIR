import argparse
import itertools
import os

import cv2
import h5py
import numpy as np
import scipy.io
from tqdm import tqdm

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timescale', type=float, default=1e6)
    parser.add_argument('--t_shift', type=float, default=-0.04)
    parser.add_argument('--n_kpts', type=int, default=10)
    parser.add_argument('--in_name', type=str, default='mat/camerashake1.mat')
    parser.add_argument('--out_name', type=str, default='hdf5/camerashake1.hdf5')
    args = parser.parse_args()
    return args

def create_voxel(events, num_bins=40, width=32, height=32):
    # events: [t, x, y, p]
    assert(num_bins % 2 == 0) # equal number of positive and negative
    voxel_grid = np.zeros((num_bins, height, width), np.float32)
    if len(events) == 0:
        return voxel_grid
    t_min, t_max = np.min(events[:, 0]), np.max(events[:, 0])
    x = np.int32(events[:, 1])
    y = np.int32(events[:, 2])
    bin_time = (t_max - t_min) / (np.ceil(num_bins / 2))
    for i_bin in range(num_bins):
        t_start = t_min + bin_time * (i_bin // 2)
        t_end = t_start + bin_time
        validity = (events[:, 0] >= t_start) & (events[:, 0] < t_end)
        if i_bin % 2 == 0:
            validity &= events[:, 3] > 0
        else:
            validity &= events[:, 3] <= 0
        np.add.at(voxel_grid[i_bin], (y[validity], x[validity]), 1)
    return voxel_grid

def extract_keypoints(events, n_kpts=10, width=240, height=180):
    # create uniform pivots
    keypoints = np.linspace(-1, 1, n_kpts)[:, None, None]
    interval = keypoints[1] - keypoints[0]
    left, right = keypoints[0], (keypoints[0] + keypoints[1]) / 2
    index, candidate = 0, np.full((height, width), np.nan)
    keypoints = np.tile(keypoints, (1, height, width))
    changes = np.zeros((height, width), np.uint8)
    if len(events) == 0:
        return keypoints
    # normalize timestamps to [-1, 1]
    events[:, 0] = (events[:, 0] - events[0, 0]) / (events[-1, 0] - events[0, 0])
    for t, x, y, _ in events:
        x, y = int(x), int(y)
        while t >= right:
            for yy in range(height):
                for xx in range(width):
                    if not np.isnan(candidate[yy, xx]):
                        keypoints[index, yy, xx] = candidate[yy, xx]
                        changes[yy, xx] += 1
                        candidate[yy, xx] = np.nan
            left, right = right, right + interval
            index += 1
        if np.isnan(candidate[y, x]):
            candidate[y, x] = t
        else:
            old_dist = np.abs(candidate[y, x] - keypoints[index, y, x])
            new_dist = np.abs(t - keypoints[index, y, x])
            if old_dist > new_dist:
                candidate[y, x] = t
    for yy in range(height):
        for xx in range(width):
            if not np.isnan(candidate[yy, xx]):
                keypoints[index, yy, xx] = candidate[yy, xx]
                changes[yy, xx] += 1
    return keypoints

def load_data(args):
    raw_data = scipy.io.loadmat(args.in_name)['matlabdata'][0, 0]['data'][0, 0]
    flip = os.path.basename(args.in_name) != 'rotatevideonew2_6.mat'
    samples = raw_data['frame'][0, 0]['samples']
    timeStampStart = np.float64(raw_data['frame'][0, 0]['timeStampStart'][:, 0])
    timeStampEnd = np.float64(raw_data['frame'][0, 0]['timeStampEnd'][:, 0])
    x = np.float32(raw_data['polarity'][0, 0]['x'][:, 0])
    y = np.float32(raw_data['polarity'][0, 0]['y'][:, 0])
    t = np.float32(raw_data['polarity'][0, 0]['timeStamp'][:, 0])
    p = np.float32(raw_data['polarity'][0, 0]['polarity'][:, 0])
    if flip:
        x = 239 - x
        y = 179 - y
    else:
        x = x - 1
        y = y - 1
    t = t / args.timescale
    return {
            'flip': flip,
            'samples': samples,
            'timeStampStart': timeStampStart,
            'timeStampEnd': timeStampEnd,
            'x': x,
            'y': y,
            't': t,
            'p': p
            }

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def collect_data(args, data, frame_idx, writer, write_idx):
    # blurry frame
    blurry_frame = np.float32(normalize(data['samples'][frame_idx, 0]))
    if data['flip']:
        blurry_frame = cv2.flip(blurry_frame, 0) # 0: vertical flip
    if blurry_frame.shape != [180, 240]:
        blurry_frame_ = np.zeros((180, 240), dtype=np.float32)
        ymax = min(blurry_frame.shape[0], 180)
        xmax = min(blurry_frame.shape[1], 240)
        blurry_frame_[:ymax, :xmax] = blurry_frame[:ymax, :xmax]
        blurry_frame = blurry_frame_
    blurry_frame = blurry_frame[None, :, :]
    # event map
    t_for = data['timeStampStart'][frame_idx + 1] / args.timescale - \
            data['timeStampEnd'][frame_idx] / args.timescale
    t_back = data['timeStampStart'][frame_idx] / args.timescale - \
             data['timeStampEnd'][frame_idx - 1] / args.timescale
    eventstart = data['timeStampStart'][frame_idx] / args.timescale + \
                 args.t_shift - \
                 t_back / 2
    eventend = data['timeStampEnd'][frame_idx] / args.timescale + \
               args.t_shift + \
               t_for / 2
    event_validity = (data['t'] >= eventstart) & (data['t'] <= eventend)
    x = data['x'][event_validity]
    y = data['y'][event_validity]
    p = data['p'][event_validity]
    t = data['t'][event_validity]
    txyp = np.stack([t, x, y, p], axis=1)
    event_map_small = create_voxel(txyp,
                                   num_bins=26,
                                   width=240,
                                   height=180)
    event_map_big = create_voxel(txyp,
                                 num_bins=198,
                                 width=240,
                                 height=180)
    # keypoints
    keypoints = extract_keypoints(txyp, n_kpts=args.n_kpts)
    # write
    writer['frame_idx'][write_idx] = frame_idx
    writer['blurry_frame'][write_idx, 0, :, :] = blurry_frame
    writer['event_map_small'][write_idx, :, :, :] = event_map_small
    writer['event_map_big'][write_idx, :, :, :] = event_map_big
    writer['keypoints'][write_idx, :, :, :] = keypoints
    write_idx += 1
    return write_idx

def main():
    args = parse_args()
    data = load_data(args)
    length = len(data['samples']) - 2
    os.makedirs(os.path.dirname(args.out_name), exist_ok=True)
    with h5py.File(args.out_name, 'w', libver='latest') as f:
        frame_idx = f.create_dataset('frame_idx', (length,), dtype='i')
        blurry_frame = f.create_dataset('blurry_frame',
                                        (length, 1, 180, 240),
                                        dtype='f',
                                        chunks=(1, 1, 180, 240),
                                        compression='gzip')
        event_map_small = f.create_dataset('event_map_small',
                                           (length, 26, 180, 240),
                                           dtype='f',
                                           chunks=(1, 1, 180, 240),
                                           compression='gzip')
        event_map_big = f.create_dataset('event_map_big',
                                          (length, 198, 180, 240),
                                          dtype='f',
                                          chunks=(1, 1, 180, 240),
                                          compression='gzip')
        keypoints = f.create_dataset('keypoints',
                                     (length, args.n_kpts, 180, 240),
                                     dtype='f',
                                     chunks=(1, 1, 180, 240),
                                     compression='gzip')
        write_idx = 0
        for i in tqdm(range(1, len(data['samples']) - 1)):
            write_idx = collect_data(args, data, i, f, write_idx)

if __name__ == '__main__':
    main()
