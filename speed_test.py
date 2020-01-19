#!/usr/bin/env python3

import torch
import numpy as np
import argparse
#import tensorrt as trt
from time import time
import sys
from PIL import Image

import torch.backends.cudnn as cudnn

PARAMETRIC = '-p' in sys.argv

WIDTH = 0.25

################# MODEL #################
if PARAMETRIC:
    from libs.parametric_models import encoder3, decoder3, MulLayer
    e3c = encoder3(0.25, False).eval().cuda()
    e3s = encoder3(0.25, False).eval().cuda()
    d3 = decoder3(0.25, False).eval().cuda()
    mat3 = MulLayer(0.25).eval().cuda()
    e3c.load_state_dict(torch.load('models/pruned/vgg_c_r31.pth'))
    e3s.load_state_dict(torch.load('models/pruned/vgg_s_r31.pth'))
    d3.load_state_dict(torch.load('models/pruned/dec_r31.pth'))
    mat3.load_state_dict(torch.load('models/pruned/matrix_r31.pth'))
    #e3c.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/vgg_c_r31.pth'))
    #e3s.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/vgg_s_r31.pth'))
    #d3.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/dec_r31.pth'))
    #mat3.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/matrix_r31.pth'))

else:
    from libs.models import encoder3, decoder3 
    from libs.Matrix import MulLayer
    e3c = encoder3().eval().cuda()
    e3s = encoder3().eval().cuda()
    d3 = decoder3().eval().cuda()
    e3c.load_state_dict(torch.load('models/regular/vgg_r31.pth'))
    e3s.load_state_dict(torch.load('models/regular/vgg_r31.pth'))
    d3.load_state_dict(torch.load('models/regular/dec_r31.pth'))

    mat3 = MulLayer('r31').eval().cuda()
    mat3.load_state_dict(torch.load('models/regular/r31.pth'))

################## STYLE ####################3

with torch.no_grad():
    style = Image.open("data/style/2.jpg")
    style = torch.from_numpy(np.asarray(style.resize((576, 1024))).transpose(2,0,1)).unsqueeze(0).float().cuda()
    sF = e3s(style)
    im = Image.open("data/content/057725.jpg")
    im = torch.from_numpy(np.asarray(im.resize((576, 1024))).transpose(2,0,1)).unsqueeze(0).float()
    tt = time()
    for _ in range(100):
        transfer = e3c(im.cuda())
        transfer = mat3(transfer, sF)
        transfer = d3(transfer).cpu()
    torch.cuda.synchronize()
    print(100/(time() - tt))

