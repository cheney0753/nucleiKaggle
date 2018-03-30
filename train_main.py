#!/usr/bin/env python3cuda()
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:49:10 2018

@author: zhong

To exam the nuclei data @ kaggle.com
"""
#%matplotlib qt
isCUDA = True
#%% Import modules
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from matplotlib import pyplot as plt
import os, sys
import datetime
now = datetime.datetime.now()
#%%
cwd = %pwd
sys.path.append(os.path.abspath(os.path.join( os.pardir, cwd)))
from nuclei.kernel import cnnModule
from nuclei.kernel.lossFunc import diceLoss, IoU_mean
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure
from sklearn import cluster
#%%
from torch.nn import MSELoss, L1Loss
from nuclei.utils import data

data_dir  = '/export/scratch1/zhong/PhD_Project/Projects/Kaggle/nuclei/data'

temp_dir = os.path.join( '/export/scratch1/zhong/PhD_Project/Projects/Kaggle/nuclei/temp', '%d%d%d'%(now.year, now.month, now.day))

try:
    assert os.path.exists(temp_dir)
except AssertionError:
    os.mkdir(temp_dir)

traindata = data.TrainDf(data_dir)

chromaticDS = data.NucleiDataset(traindata.df.query('chromatic==True').reset_index())

dataloader = DataLoader(chromaticDS, batch_size=1,
                        shuffle=True, num_workers=8)


#%%  msdNet


  
from nuclei.kernel import msdModule
from nuclei.kernel.lossFunc import crossEntropy2d_sum

msdNet_chro = msdModule.msdNet(3, 2, 20, 3 )

epoch =30
optimizer = optim.Adam(msdNet_chro.parameters(),   lr= 1e-3)
optimizer.zero_grad()

for ipc in range(epoch):
    r_loss = 0.0

  
    for ide, (image_, mask_2ch) in enumerate(dataloader):  
        optimizer.zero_grad()
      
      #forward
#        mask_2ch = torch.cat( (1 - mask_, mask_), dim=-3)[None, ...]
        
        output_ = msdNet_chro(Variable(image_).cuda())
        loss = crossEntropy2d_sum(output_.contiguous(), Variable(mask_2ch).cuda())
        loss.backward()
        optimizer.step()
      
        r_loss += loss.data[0]

    print('[%d, %5d] loss: %.5f'%(ipc + 1, ide + 1, r_loss/(ide+1) ))
    

    
    if ipc%5 == 0 :
        ndshow= 3
        fig, ax = plt.subplots(3, ndshow,figsize = (12, 6))
        for (_, c_row), (c_img, c_msk, c_out) in zip( traindata.df.query('chromatic==True').sample(ndshow).iterrows(), ax.T):
          c_img.imshow(c_row['images'])
          c_img.axis('off')
        
          img_ = Variable( torch.from_numpy( 
                np.moveaxis( c_row['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
          out_ = msdNet_chro(img_).data[:,1,:,:].cpu().numpy().squeeze()
          
          c_out.imshow(  out_)
          c_out.axis('off')
          c_out.set_title('predict')
          
        
            
          c_msk.imshow(c_row['masks'])
          c_msk.axis('off')
          c_msk.set_title('ground_truth')

        
#%%

ndshow= 6
fig, ax = plt.subplots(3, ndshow,figsize = (12, 6))

for (_, c_row), (c_img, c_msk, c_out) in zip( traindata.df_chromatic.sample(ndshow).iterrows(), ax.T):
  c_img.imshow(c_row['images'])
  c_img.axis('off')

  img_ = Variable( torch.from_numpy( 
        np.moveaxis( c_row['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
  out_ = msdNet_chro(img_).data.cpu().numpy().squeeze()
  
#  c_img.imshow(img_.data.cpu().numpy().squeeze())
#  c_img.axis('off')
#  
  c_out.imshow(out_)
  c_out.axis('off')
  c_out.set_title('predict')
    
  c_msk.imshow(c_row['masks'])
  c_msk.axis('off')
  c_msk.set_title('ground_truth')
  
  
  #%% put the data into Variable
_, train_rle_row = next(traindata.df.sample(1).iterrows())
  
train_row_rles = data.rle_fromImage(train_rle_row['masks'])

tl_rles = traindata.labels.query('ImageId=="{ImageId}"'.format(**train_rle_row))['EncodedPixels']

tl_rles = sorted(tl_rles, key = lambda x: int(x[0]))

train_row_rles = sorted(train_row_rles, key = lambda x: x[0])

for tl, tr in zip(tl_rles, train_row_rles):
    print(tl[0], tr[0])
match, mismatch = 0, 0
for img_rle, train_rle in zip(sorted(train_row_rles, key = lambda x: x[0]), 
                             sorted(tl_rles, key = lambda x: int(x[0]))):
    for i_x, i_y in zip(img_rle, train_rle):
        if i_x == int(i_y):
            match += 1
        else:
            mismatch += 1
print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' % (match, mismatch, 100.0*match/(match+mismatch)))
