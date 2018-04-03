#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:24:10 2018

@author: zhong
"""
from __future__ import absolute_import
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np
from torch.autograd import Variable
import torch
__all__ = ("plot_network_output",)

def plot_network_output(netWork, pandaSeries, axis = 1, ndshow = 3, withMasks = True, withCentroids = True):
    plt.ioff()   
    fig, ax = plt.subplots(2+int(withMasks), ndshow)
    
    for (_, c_row), (c_img, c_msk, c_out) in zip( pandaSeries.sample(ndshow).iterrows(), ax.T):
      c_img.imshow(c_row['images'])
      c_img.axis('off')
    
      img_ = Variable( torch.from_numpy( 
            np.moveaxis( c_row['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
      out_ = netWork(img_).data[:,axis,:,:].cpu().numpy().squeeze()
      
      c_out.imshow(out_)
      c_out.axis('off')
      c_out.set_title('predict')
                  
      if withMasks and not withCentroids:
          c_msk.imshow(c_row['masks'])
          c_msk.axis('off')
          c_msk.set_title('ground_truth')
      elif withCentroids:
          c_msk.imshow(c_row['centroids'])
          c_msk.axis('off')
          c_msk.set_title('ground_truth')
    #  return the handle
    return fig
