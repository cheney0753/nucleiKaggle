#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:25:54 2018

@author: zhong

Email: zhchzhong@gmail.com
"""
from dens
class msdModule(nn.Module):
    
    def __init__(self):
        super(chromaticNet, self).__init__()
        
        
        # 3 input image channel, 8 output channels, 3*3 square convolution
        # kernel
        
        self.net =  nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 8, 3),
            nn.Conv2d(8, 16, 3, dilation= 2, padding = 2),
            nn.Conv2d(16, 16, 3, dilation= 2, padding = 2),

            nn.Conv2d(16, 32, 3, dilation= 4, padding = 6),
            nn.Conv2d(32, 16, 1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
            ).cuda()

#        self.conv1  = nn.Conv2d(3, 8, 3).cuda()
#        self.conv2  = nn.Conv2d(8, 8, 3).cuda()
#	
##        # an affine operation: y = Wx + b
##        self.fc1 = nn.Linear(16 * 15 * 15, 120)
##        self.fc2 = nn.Linear(120, 84)
##        self.fc3 = nn.Linear(84, 1)
#        
#        self.conv_dil_1 = nn.Conv2d(8, 16, 3, dilation= 2, padding = 3).cuda()
#        self.conv_dil_2 = nn.Conv2d(16, 16, 3, dilation= 4, padding = 6).cuda()
#
#        self.conv_bin_1 = nn.Conv2d(16, 4, 3).cuda()
#        self.conv_bin_2 = nn.Conv2d(4, 1, 1).cuda()
        
    def forward(self, x):
      
        
        return self.net(x)