#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:44:29 2018

@author: zhong

Email: zhchzhong@gmail.com
"""

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class chromaticNet(nn.Module):
    
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

        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features
      
class monochromNet(nn.Module):
    
    def __init__(self):
        super(chromaticNet, self).__init__()
        # 3 input image channel, 8 output channels, 3*3 square convolution
        # kernel
        self.conv1  = nn.Conv2d(1, 1, 3)
        self.conv2  = nn.Conv2d(1, 1, 3) 
	
#        # an affine operation: y = Wx + b
#        self.fc1 = nn.Linear(16 * 15 * 15, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 1)
        
        self.conv_dil_1 = nn.Conv2d(1, 1, 3, dilation= 2, padding = 3)
        self.conv_dil_2 = nn.Conv2d(1, 1, 3, dilation= 4, padding = 6)

        self.conv_bin_1 = nn.Conv2d(1, 1, 3)
        self.conv_bin_2 = nn.Conv2d(1, 1, 1)
        
    def forward(self, x):
        
        x = F.relu(self.conv2(self.conv1(x)))
        
        x = F.relu(self.conv_dil_2(self.conv_dil_1(x)))
        
        x = self.conv_bin_2(F.relu(self.conv_bin_1(x)))
        
        x = x.sigmoid()
        
        return x

        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features