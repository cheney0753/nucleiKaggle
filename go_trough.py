#!/usr/bin/env python3cuda()
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:49:10 2018

@author: zhong

To exam the nuclei data @ kaggle.com
"""
%matplotlib qt
#%% Import modules
import glob, platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pyqtgraph as pg
from scipy import ndimage
from skimage import io
from matplotlib import pyplot as plt
import os, sys
#%%
from kernel import cnnModule
from torch.autograd import Variable

import torch.optim as optim
#%%
data_dir  = os.path.join('..', 'data')

stage_label = 'stage1'

#%% load the input label csv

train_labels = pd.read_csv(
    os.path.join(data_dir, '{}_train_labels.csv'.format(stage_label)))

train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda en: [x for x in en.split(' ')] )

print(train_labels.sample()['EncodedPixels'])

#%% load the training files 
allimagepath = glob.glob(os.path.join( data_dir, '{}_*'.format(stage_label),
                                 '*' , '*', '*'))
img_df = pd.DataFrame({'path': allimagepath})

if platform.system() == 'Windows':
    split_str = '\\'
else:
    split_str = '/'

img_id = lambda inpath: inpath.split(split_str)[-3]
img_type = lambda inpath: inpath.split(split_str)[-2]
img_group =  lambda inpath: inpath.split(split_str)[-4].split('_')[-1]
img_stage =  lambda inpath: inpath.split(split_str)[-4].split('_')[0]

img_df['ImageId'] = img_df['path'].map( img_id)
img_df['ImageType'] = img_df['path'].map( img_type)
img_df['TrainingSplit'] = img_df['path'].map( img_group)
img_df['Stage'] = img_df['path'].map( img_stage)

img_df.sample(2)

#%% load the traing data
%%time

train_df = img_df.query('TrainingSplit=="train"')

train_rows = []

group_cols = ['Stage', 'ImageId']

for n_group, n_rows in train_df.groupby(group_cols):
  # n_group = ('stage1', 'ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48')
  # n_rows = DataFrame (25, 5) with the same 'Stage' and 'ImageId'
  
  # get the n_group into a dictionary
  c_row = {col_name: col_v for col_name, col_v, in zip(group_cols, n_group)}
  
  # get a list of the path for masks
  c_row['masks'] = n_rows.query('ImageType=="masks"')['path'].values.tolist()
  c_row['images'] = n_rows.query('ImageType=="images"')['path'].values.tolist()
  
  # a list containing the dictionary of {'ImageId', 'Stage', 'masks', 'images'}
  # and the paths
  train_rows += [c_row]
  
train_img_df = pd.DataFrame(train_rows)

IMG_CHANNELS = 3

# read the images and bin from RGB channels to gray images
 
def read_and_stack(in_img_list):
  return np.sum(np.stack([ io.imread(c_img) for c_img in in_img_list]), 0) / 255.0

train_img_df['images'] = train_img_df['images'].map(read_and_stack).map( lambda x: x[:,:,:IMG_CHANNELS])

train_img_df['masks'] = train_img_df['masks'].map(read_and_stack).map(lambda x: x.astype(int))
train_img_df.sample(1)


#%% define a function to test if the image is colorful or not

def isColorful( img ):
    assert img.shape[2] == IMG_CHANNELS;
    
    number_sample = 5
    sizeImg = img.shape[0:IMG_CHANNELS]
    rand_pix = [(a, b) for a, b in zip( np.random.choice(sizeImg[0], number_sample),
                np.random.choice(sizeImg[1], number_sample) )]
    for pix in rand_pix:
        pix_ch = img[pix[0], pix[1], :]
        if np.array_equal(pix_ch, np.roll(pix_ch, 1)):
            return False
        
    return True


#%% devide the data into chromatic and monochromatic sets

train_img_df['chromatic'] = train_img_df['images'].map( isColorful )  

# check dimensions of the images
train_img_df['images'].map( lambda x: x.shape).value_counts()

train_img_chromatic = train_img_df.query('chromatic==True')
train_img_monochrom = train_img_df.query('chromatic==False')
train_img_chromatic['images'].map( lambda x: x.shape).value_counts()
#%% show some images

# to numpy matrix : train_img_df.sample(1)['images'].as_matrix()[0]
n_img = 12

fig, ax = plt.subplots(2, n_img, figsize = (12, 4))

for (_, c_row), (c_im, c_lab) in zip( train_img_chromatic.sample(n_img).iterrows(), ax.T):
      c_im.imshow(c_row['images'])
      c_im.axis('off')
      c_im.set_title('Microscope')
      
      c_lab.imshow(c_row['masks'])
      c_lab.axis('off')
      c_lab.set_title('Labeled')


#%% define the loss function

def diceLoss(input_, target_):
      smooth = 1
      
      iflat = input_.view(-1)
      tflat = target_.view(-1)
      intersection = (iflat*tflat).sum()
      loss_ = -(2.* intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
      
      return loss_
#%%
cnnNet = cnnModule.chromaticNet()
print(cnnNet)
#%% put the data into Variable


#criterion = diceLoss()

optimizer = optim.SGD(cnnNet.parameters(), lr=0.001)
epoch = 1

r_loss = 0.0

for ipc in range(epoch):
  
  
  for ide in range(train_img_monochrom.shape[0]):
      ind = 1
      
      i = ide%train_img_df.shape[0]
       #eshape it to 4D
      image_ = Variable( torch.from_numpy( 
          np.moveaxis( train_img_monochrom.iloc[i]['images'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
      mask_ = Variable( torch.from_numpy(
          np.moveaxis(train_img_monochrom.iloc[i]['masks'], -1, 0)[None, ...].astype('float32')).cuda().contiguous())
      
#      
#      image_ = Variable( torch.from_numpy( 
#          np.moveaxis( np.stack(train_img_df.iloc[i:i+10]['images'].as_matrix()), -1, 1).astype('float32')).cuda().contiguous())
#      mask_ = Variable( torch.from_numpy(
#          np.moveaxis(np.stack(train_img_df.iloc[i:i+10]['masks'].as_matrix()), -1, 1).astype('float32')).cuda().contiguous())
#      
      optimizer.zero_grad()
      
      #forward
      
      output_ = cnnNet(image_)
      loss = diceLoss(mask_, output_)
      loss.backward()
      optimizer.step()
      
      r_loss += loss.data[0]
  #    print statistics
#      print( loss.data[0])
      if ide % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (ipc + 1, ide + 1, r_loss / 100))
            r_loss = 0.0
#  print('[%d, %5d] loss: %.5f' %(ipc + 1, ide + 1, r_loss / ide))
#  r_loss = 0.0

print('Finished Training')
  #%%

ndshow= 6
fig, ax = plt.subplots(3, ndshow,figsize = (12, 6))

for (_, c_row), (c_img, c_msk, c_out) in zip( train_img_monochrom.sample(ndshow).iterrows(), ax.T):
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
class chromaticDataset(Dataset):
  """ chromatic images dataset. """
  
  def __init__(self, df, transform = None):
    self.df = df
    self.transform = transform
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    images = np.moveaxis(self.df.iloc[index]['images'].astype('float32'), -1, 0)
    masks = np.moveaxis(self.df.iloc[index]['masks'].astype('float32'),  -1, 0)


    return (images, masks)
  
#%%
  
  #%%
cnnNet_chro = cnnModule.chromaticNet()
print(cnnNet_chro)

chromaticDS = chromaticDataset(train_img_chromatic)

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

for (_, c_row), (c_img, c_msk, c_out) in zip( train_img_chromatic.sample(ndshow).iterrows(), ax.T):
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
  