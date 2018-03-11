#!/usr/bin/env python3cuda()
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:49:10 2018

@author: zhong

To exam the nuclei data @ kaggle.com
"""
%matplotlib qt
isCUDA = False
#%% Import modules
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from matplotlib import pyplot as plt
import os, sys
#%%
cwd = %pwd
sys.path.append(os.path.abspath(os.path.join( os.pardir, cwd)))
from kernel import cnnModule
from kernel.lossFunc import diceLoss, IoU_mean
from torch.autograd import Variable
from utils import image, data
import torch.optim as optim
#%%
%%time

data_dir  = os.path.join('..', 'data')

traindata = data.trainDf(data_dir)

#%% show some images

# to numpy matrix : train_img_df.sample(1)['images'].as_matrix()[0]
n_img = 12

fig, ax = plt.subplots(2, n_img, figsize = (12, 4))

for (_, c_row), (c_im, c_lab) in zip( traindata.df_chromatic.sample(n_img).iterrows(), ax.T):
      c_im.imshow(c_row['images'])
      c_im.axis('off')
      c_im.set_title('Microscope')
      
      c_lab.imshow(c_row['masks'])
      c_lab.axis('off')
      c_lab.set_title('Labeled')


#%%
cnnNet = cnnModule.chromaticNet()
print(cnnNet)
#%% put the data into Variable

#criterion = diceLoss()

optimizer = optim.SGD(cnnNet.parameters(), lr=0.001)
epoch = 1

r_loss = 0.0

for ipc in range(epoch):
   for ide in range(traindata.df_monochrom.shape[0]):
      ind = 1
      
#      i = ide%train_img_df.shape[0]
       #eshape it to 4D
      if isCUDA:
          image_ = Variable( torch.from_numpy( 
              np.moveaxis( traindata.df_monochrom.iloc[ide]['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
          mask_ = Variable( torch.from_numpy(
              np.moveaxis(traindata.df_monochrom.iloc[ide]['masks'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
      else:
          image_ = Variable( torch.from_numpy( 
              np.moveaxis( traindata.df_monochrom.iloc[ide]['images'], -1, 0)[None, ...].astype('float32')).contiguous())
          mask_ = Variable( torch.from_numpy(
              np.moveaxis(traindata.df_monochrom.iloc[ide]['masks'], -1, 0)[None, ...].astype('float32')).contiguous())
   
      optimizer.zero_grad()
      
      #forward
      
      output_ = cnnNet(image_)
#      loss = diceLoss(mask_, output_)
      loss = diceLoss( output_, mask_>0)
      loss.backward()
      optimizer.step()
      
      r_loss += loss.data[0]
  #    print statistics
      if ide % 100 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (ipc + 1, ide + 1, r_loss / 100))
            r_loss = 0.0

print('Finished Training')
  #%%

ndshow= 6
fig, ax = plt.subplots(3, ndshow,figsize = (12, 6))

for (_, c_row), (c_img, c_msk, c_out) in zip( traindata.df_monochrom.sample(ndshow).iterrows(), ax.T):
  c_img.imshow(c_row['images'])
  c_img.axis('off')

  img_ = Variable( torch.from_numpy( 
        np.moveaxis( c_row['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
  out_ = cnnNet(img_).data.cpu().numpy().squeeze()
  
#  c_img.imshow(img_.data.cpu().numpy().squeeze())
#  c_img.axis('off')
#  
  c_out.imshow(out_)
  c_out.axis('off')
  c_out.set_title('predict')
    
  c_msk.imshow(c_row['masks'])
  c_msk.axis('off')
  c_msk.set_title('ground_truth')

  
#%% test using msd_module

#%%
  
  #%%
cnnNet_chro = cnnModule.chromaticNet()
print(cnnNet_chro)

chromaticDS = data.chromaticDataset(traindata.df_chromatic)

sample = chromaticDS[1]

dataloader = DataLoader(chromaticDS, batch_size=30,
                        shuffle=True, num_workers=4)

optimizer = optim.SGD(cnnNet_chro.parameters(), lr=0.0001)

epoch = 200
for ipc in range(epoch):
  
  
  for ide, (image_, mask_) in enumerate(dataloader):  

      optimizer.zero_grad()
      
      #forward
      
      output_ = cnnNet_chro(Variable(image_).cuda())
      loss = diceLoss(Variable(mask_).cuda(), output_)
      loss.backward()
      optimizer.step()
      
      r_loss += loss.data[0]
  #    print statistics
#      print( loss.data[0])
  
  print('[%d, %5d] loss: %.5f'%(ipc + 1, ide + 1, r_loss/(ide+1) ))
  r_loss = 0.0

  #%%

ndshow= 6
fig, ax = plt.subplots(3, ndshow,figsize = (12, 6))

for (_, c_row), (c_img, c_msk, c_out) in zip( traindata.df_chromatic.sample(ndshow).iterrows(), ax.T):
  c_img.imshow(c_row['images'])
  c_img.axis('off')

  img_ = Variable( torch.from_numpy( 
        np.moveaxis( c_row['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
  out_ = cnnNet_chro(img_).data.cpu().numpy().squeeze()
  
#  c_img.imshow(img_.data.cpu().numpy().squeeze())
#  c_img.axis('off')
#  
  c_out.imshow(out_)
  c_out.axis('off')
  c_out.set_title('predict')
    
  c_msk.imshow(c_row['masks'])
  c_msk.axis('off')
  c_msk.set_title('ground_truth')

  
#%% Define a msd network
  
from modules import msd_module

input_sz = list(img_.data.shape[2:])
batch_sz = img_.data.shape[0]
depth = 2
do_padding = False

msdNet = msd_module.MSDModule(input_sz, batch_sz, depth, False)

#%%
img_ = img_[:,1,:,:][None, ...]
prd = msdNet(img_)
  