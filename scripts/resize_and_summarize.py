import argparse
import multiprocessing
import os

import cv2

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='raw')
    parser.add_argument('--output_dir', type=str, default='resized')
    parser.add_argument('--output_height', type=int, default=180)
    parser.add_argument('--output_width', type=int, default=240)
    args = parser.parse_args()
    return args

def payload(paras):
    args, i_video, split = paras
    in_dir = os.path.join(args.raw_dir, split, '{:03d}'.format(i_video))
    out_dir = os.path.join(args.output_dir, split, '{:03d}'.format(i_video))
    os.makedirs(out_dir, exist_ok=True)
    csv_name = os.path.join(out_dir, 'images.csv')
    with open(csv_name, 'w') as f:
        for i_frame in range(500):
            in_name = os.path.join(in_dir, '{:08d}.png'.format(i_frame))
            frame = cv2.imread(in_name, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, (args.output_width, args.output_height))
            out_name = os.path.join(out_dir, 'frames_{:010d}.png'.format(i_frame))
            cv2.imwrite(out_name, frame)
            timestamp = int(i_frame / 120 * 1e9) + 1
            f.write('{},frames_{:010d}.png\n'.format(timestamp, i_frame))

def main():
    args = parse_args()
    with multiprocessing.Pool(8) as pool:
        for split, num_videos in [('train', 240), ('val', 30)]:
            pool.map(payload,
                     [(args, i_video, split) for i_video in range(num_videos)])
if __name__ == '__main__':
    main()
