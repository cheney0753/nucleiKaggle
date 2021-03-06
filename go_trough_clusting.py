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
#%%
#cwd = %pwd
sys.path.append(os.path.abspath(os.path.join( os.pardir, cwd)))

#%%
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure
#from sklearn import cluster

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
#%%
from torch.nn import MSELoss, L1Loss
from nuclei.utils import data


data_dir  = os.path.abspath( os.path.join( os.path.dirname( __file__), os.pardir, 'data_sample'))

merge_masks_data = data.TrainDf(data_dir, target_key = 'merge_masks')


chromaticDS = data.NucleiDataset(traindata.df.query('chromatic==True').reset_index())


dataloader = DataLoader(chromaticDS, batch_size=4,
                        shuffle=True, num_workers=4)
#%% show some images

# to numpy matrix : train_img_df.sample(1)['images'].as_matrix()[0]
n_img = 4

fig, ax = plt.subplots(2, n_img, figsize = (12, 4))

for (_, c_row), (c_im, c_lab) in zip( traindata.df.query('chromatic==True').sample(n_img).iterrows(), ax.T):
      c_im.imshow(c_row['images'])
      c_im.axis('off')
      c_im.set_title('Microscope')
      
#      c_lab.imshow(c_row['masks']+c_row['centroids'] +c_row['boundaries'])
      c_lab.imshow(c_row['centroids'] )
      c_lab.axis('off')
      c_lab.set_title('Labeled')
#%% get a test for k-means clusterin

# get the number of centroids
labled_centroids,k_n = measure.label(c_row['centroids'], return_num=True)        
print('The number of k: {}'.format(k_n))
_, ax = plt.subplots(1)
ax.imshow( labled_centroids)

## get the labelings on the overlapping mask based on k_means clustering 
##%%
#def _get_centroid( msk):
#    
#    nz =  np.nonzero( msk )
#    
#    ctr = [int(nzi.mean()) for nzi in nz]
#
#    return ctr
#
#def km_labeling(mask, c_n, label_centroids):
#    
#    mask_stack = np.zeros([ c_n, *(mask.shape)], dtype = float)
#    init_ct = np.zeros((c_n, 2))
#    
#    # get shape (n_samples, n_features)
#    points = np.transpose(np.array(np.nonzero(mask)))
#    
#    # get the initcentroids of 
#    for c in range(c_n):
#        
#        label_c = ( label_centroids == c).astype(float)
#        
#        init_ct[c,:] = _get_centroid( label_c)
#    
#    
#    cl = cluster.k_means( points, c_n, init = init_ct)
#    return init_ct, cl
#
#
#mask = c_row['masks']
#cl= km_labeling(mask, k_n, labled_centroids)
#
#points = np.transpose(np.array(np.nonzero(mask))).astype(int)
#
#lb_mask = np.zeros( mask.shape)
#
#for lb, p in zip(cl[1][1], points):
#    lb_mask[p[0],p[1]] = float(lb)
#    
#_, ax = plt.subplots(1,3)
#ax[0].imshow( mask)    
#ax[1].imshow( lb_mask)    
#ax[2].imshow( lb_mask - mask)    

#%%
mask = train_rle_row['masks']

distance = ndi.distance_transform_edt(mask)

#markers = ndi.label(local_maxi)
labels = watershed(-distance, labled_centroids, mask=mask)

_, ax = plt.subplots(1,3)
ax[0].imshow( mask)    
ax[1].imshow( labels)    
ax[2].imshow( mask - labels)    

#%%
      
from skimage.morphology import closing, opening, disk
def clean_img(x):
    return opening(closing(x, disk(1)), disk(3))
 #%%
cnnNet = cnnModule.ChromaticNet()
print(cnnNet)
#%% put the data into Variable


_, train_rle_row = next(traindata.df.sample(1).iterrows())

labled_centroids,k_n = measure.label(train_rle_row['centroids'], return_num=True)        

train_row_rles = data.rle_fromImage(train_rle_row['masks'], labled_centroids)

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

#%% 

def from_rles(rle, size):
    
    masks = np.zeros(size)
    
    for r in rle:
        
    
    
    
#%%
mask = train_rle_row['masks']

distance = ndi.distance_transform_edt(mask)

#markers = ndi.label(local_maxi)
labels = watershed(-distance, labled_centroids, mask=mask)

_, ax = plt.subplots(1,3)
ax[0].imshow( mask)    
ax[1].imshow( labels)    
ax[2].imshow( mask - labels)    


#%%  msdNet
  
from nuclei.kernel import msdModule
from nuclei.kernel.lossFunc import crossEntropy2d_sum


msdNet_chro = msdModule.msdNet(3, 2, 20, 3 )

epoch =3
optimizer = optim.Adam(msdNet_chro.parameters(),   lr= 1e-3)
optimizer.zero_grad()
criterion = L1Loss()

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
#          c_msk.imshow(msdNet_chro(img_).data[:,1,:,:].cpu().numpy().squeeze())
#          c_msk.axis('off')
#          c_msk.set_title('the second part')
        
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
  