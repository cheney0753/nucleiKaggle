#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:25:54 2018

@author: zhong

Email: zhchzhong@gmail.com
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from msd_pytorch.msd_module import (MSDModule, msd_dilation)

__all__ = ('msdNet', )

class msdNet(nn.Module):
    
    def __init__(self, c_in, c_out, depth, width):
        super(msdNet, self).__init__()
        
        
        # 3 input image channel, 8 output channels, 3*3 square convolution
        # kernel
        
        self.net = nn.Sequential(
                #nn.BatchNorm2d(3).cuda(),
                MSDModule(c_in, c_out, depth, width, msd_dilation, conv3d=False),
                nn.Softmax2d().cuda())

        
    def forward(self, x):
      
        
        return self.net(x)