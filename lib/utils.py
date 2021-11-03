import os

import torch

import pdb

def save_session(model, optimizer, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    # save the model and optimizerizer state
    for key in model.keys():
        torch.save(model[key].state_dict(),
                   os.path.join(path, 'model_{}.pth'.format(key)))
    for key in optimizer.keys():
        torch.save(optimizer[key].state_dict(),
                   os.path.join(path, 'optimizer_{}.pth'.format(key)))
    print('Successfully saved model into {}'.format(path))

def load_session(model, optimizer, args):
    try:
        start_epoch = int(args.load_dir.split('/')[-1]) + 1
        for key in model.keys():
            path = os.path.join(args.load_dir, 'model_{}.pth'.format(key))
            model[key].load_state_dict(torch.load(path))
        for key in optimizer.keys():
            path = os.path.join(args.load_dir, 'optimizer_{}.pth'.format(key))
            optimizer[key].load_state_dict(torch.load(path))
            for g in optimizer[key].param_groups:
                g['lr'] = args.lr
        print('Successfully loaded model from {}'.format(args.load_dir))
    except Exception as e:
        pdb.set_trace()
        print('Could not restore session properly, check the load_dir')
    return model, optimizer, start_epoch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count
