#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import pandas as pd
import PIL
import PIL.Image
import PIL.ImageOps
import sys
import pdb

try:
    from .sepconv import sepconv # the custom separable convolution layer
except:
    sys.path.insert(0, './sepconv'); import sepconv # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'lf'
arguments_strPadding = 'improved'
arguments_strInDir = ''
arguments_strOutDir = ''

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, l1 or lf, please see our paper for more details
    if strOption == '--padding' and strArgument != '': arguments_strPadding = strArgument # which padding to use, the one used in the paper or the improved one
    if strOption == '--in' and strArgument != '': arguments_strInDir = strArgument
    if strOption == '--out' and strArgument != '': arguments_strOutDir = strArgument
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.netConv1 = Basic(6, 32)
        self.netConv2 = Basic(32, 64)
        self.netConv3 = Basic(64, 128)
        self.netConv4 = Basic(128, 256)
        self.netConv5 = Basic(256, 512)

        self.netDeconv5 = Basic(512, 512)
        self.netDeconv4 = Basic(512, 256)
        self.netDeconv3 = Basic(256, 128)
        self.netDeconv2 = Basic(128, 64)

        self.netUpsample5 = Upsample(512, 512)
        self.netUpsample4 = Upsample(256, 256)
        self.netUpsample3 = Upsample(128, 128)
        self.netUpsample2 = Upsample(64, 64)

        self.netVertical1 = Subnet()
        self.netVertical2 = Subnet()
        self.netHorizontal1 = Subnet()
        self.netHorizontal2 = Subnet()

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/sepconv/network-' + arguments_strModel + '.pytorch', file_name='sepconv-' + arguments_strModel).items() })
    # end

    def forward(self, tenOne, tenTwo):
        tenConv1 = self.netConv1(torch.cat([ tenOne, tenTwo ], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

        tenCombine = tenDeconv2 + tenConv2

        tenOne = torch.nn.functional.pad(input=tenOne, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')
        tenTwo = torch.nn.functional.pad(input=tenTwo, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')

        tenDot1 = sepconv.FunctionSepconv(tenInput=tenOne, tenVertical=self.netVertical1(tenCombine), tenHorizontal=self.netHorizontal1(tenCombine))
        tenDot2 = sepconv.FunctionSepconv(tenInput=tenTwo, tenVertical=self.netVertical2(tenCombine), tenHorizontal=self.netHorizontal2(tenCombine))

        return tenDot1 + tenDot2
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    if arguments_strPadding == 'paper':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ,int(math.floor(51 / 2.0))

    elif arguments_strPadding == 'improved':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = 0, 0, 0, 0

    # end

    intPreprocessedWidth = intPaddingLeft + intWidth + intPaddingRight
    intPreprocessedHeight = intPaddingTop + intHeight + intPaddingBottom

    if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
        intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1) << 7) # more than necessary
    # end
    
    if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
        intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1) << 7) # more than necessary
    # end

    intPaddingRight = intPreprocessedWidth - intWidth - intPaddingLeft
    intPaddingBottom = intPreprocessedHeight - intHeight - intPaddingTop

    tenPreprocessedOne = torch.nn.functional.pad(input=tenPreprocessedOne, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')
    tenPreprocessedTwo = torch.nn.functional.pad(input=tenPreprocessedTwo, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')

    return torch.nn.functional.pad(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), pad=[ 0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom ], mode='replicate')[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
    os.makedirs(arguments_strOutDir, exist_ok=True)
    in_csv_name = os.path.join(arguments_strInDir, 'images.csv')
    in_csv = pd.read_csv(in_csv_name,
                         header=None,
                         names=['t', 'name'],
                         engine='c')
    out_csv_name = os.path.join(arguments_strOutDir, 'images.csv')
    save_idx = 0
    with open(out_csv_name, 'w') as f:
        for i in range(len(in_csv['t']) - 1):
            left_t, right_t = float(in_csv['t'][i]), float(in_csv['t'][i + 1])
            left_name = os.path.join(arguments_strInDir, in_csv['name'][i])
            right_name = os.path.join(arguments_strInDir, in_csv['name'][i + 1])
            left = numpy.array(PIL.Image.open(left_name))
            left = numpy.float32(numpy.stack([left, left, left], axis=0))
            right = numpy.array(PIL.Image.open(right_name))
            right = numpy.float32(numpy.stack([right, right, right], axis=0))

            left = torch.FloatTensor(numpy.ascontiguousarray(left * (1.0 / 255.0)))
            right = torch.FloatTensor(numpy.ascontiguousarray(right * (1.0 / 255.0)))
            result = [None] * 8
            result[0] = left
            result[4] = estimate(left, right)
            result[2] = estimate(left, result[4])
            result[1] = estimate(left, result[2])
            result[3] = estimate(result[2], result[4])
            result[6] = estimate(result[4], right)
            result[5] = estimate(result[4], result[6])
            result[7] = estimate(result[6], right)
            for j in range(8):
                frame_name = os.path.join(arguments_strOutDir,
                                         'frames_{:010d}.png'.format(save_idx))
                PIL.ImageOps.grayscale(PIL.Image.fromarray((result[j].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))).save(frame_name)
                t = int(left_t + (right_t - left_t) / 8 * j)
                f.write('{},frames_{:010d}.png\n'.format(t, save_idx))
                save_idx += 1
