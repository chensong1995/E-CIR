import h5py
import numpy as np

from torch.utils.data import Dataset

import pdb

class EDIDataset(Dataset):
    def __init__(self,
                 hdf5_name='data/EDI/hdf5/camerashake1.hdf5'):
        super(EDIDataset, self).__init__()
        self.data = h5py.File(hdf5_name, 'r')

    def __len__(self):
        return len(self.data['frame_idx'])

    def __getitem__(self, idx):
        frame_idx = self.data['frame_idx'][idx]
        blurry_frame = self.data['blurry_frame'][idx]
        event_map_small = self.data['event_map_small'][idx]
        event_map_big = self.data['event_map_big'][idx]
        keypoints = self.data['keypoints'][idx]
        timestamps = np.linspace(-1, 1, 100, dtype=np.float32)
        return {
                'frame_idx': frame_idx, # scalar
                'blurry_frame': blurry_frame, # (1, 180, 240)
                'event_map_small': event_map_small, # (26, 180, 240)
                'event_map_big': event_map_big, # (198, 180, 240)
                'keypoints': keypoints, # (n_kpts, 180, 240)
                'timestamps': timestamps # (100,)
                }

if __name__ == '__main__':
    dataset = EDIDataset()
    item = dataset[0]
    pdb.set_trace()
    print(123)
