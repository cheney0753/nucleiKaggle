#!/usr/bin/env python3cuda()
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:49:10 2018

@author: zhong

To exam the nuclei data @ kaggle.com
"""
#%matplotlib qt
isCUDA = True
#isTest = False
#%% Import modules
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("-t","--test", help="set the program in a test mode with few data and expochs.",  action="store_true")
parser.add_argument("-e","--epoch",help="number of epochs for training", default = 150, type = int)
parser.add_argument("-d","--depth",help="number of depth for the MSD-net", default = 30, type = int)
parser.add_argument("-w","--width",help="number of width for the MSD-net", default = 2, type = int)
parser.add_argument("--training_type", help="types of training. 1: merged_masks, 2: eroded_masks, 3: both", type = int, default = 3)
parser.add_argument("--image_type", help="types of training. 1: monochrom, 2: chrom, 3: both", type = int, default =3)
parser.add_argument("--dir", help="Output dir", type = str)

args = parser.parse_args()
isTest = args.test
if args.test:
    print('In the test mode.')

try:
   assert args.image_type in (1,2,3)
except AssertionError:
    raise Exception('image_type set wrong')
    
try:
   assert args.training_type in (1,2,3)
except AssertionError:
    raise Exception('training_type set wrong')

#isTest = True    
#%%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from matplotlib import pyplot as plt
import os, sys
import datetime
import pandas as pd
now = datetime.datetime.now()
#%matplotlib agg
plt.ioff()
#%%
from nuclei.kernel import cnnModule
from nuclei.kernel.lossFunc import diceLoss, IoU_mean
from skimage import measure
from sklearn import cluster

from torchvision import transforms
#%%

from nuclei.kernel import msdModule
from nuclei.kernel.lossFunc import crossEntropy2d_sum

from nuclei.utils import data
from nuclei.utils import plot_data


#%% prepare the data 
cwdir = os.path.abspath(os.path.dirname(__file__))
data_dir  = os.path.join( cwdir, os.pardir, 'data')

if isTest:
    data_dir = os.path.join(cwdir, os.pardir, 'data_sample')
    
data_dir = os.path.abspath( data_dir)

if args.dir is not None:
    temp_dir = args.dir

temp_dir = os.path.abspath( os.path.join( cwdir, os.pardir, 'temp', '%02d%02d'%(now.month, now.day)))

try:
    assert os.path.exists(temp_dir)
except AssertionError:
    os.mkdir(temp_dir)
    
orig_stdout = sys.stdout

f_stdout = os.path.abspath(os.path.join(temp_dir, 'stdout.txt'))
f=open( f_stdout, 'w')

print('Print to: ',f_stdout)
sys.stdout = f

stage = 'train'

trdict = {1:('merged_masks',), 2: ('eroded_masks',), 3: ('merged_masks','eroded_masks') }
imdict = {1:('monochrom',), 2: ('chrom',), 3: ('monochrom', 'chrom')}
training_types = trdict[ args.training_type]
image_types = imdict[args.image_type]
print('Traing_type is {}.'.format( training_types))
print('Image_type is {}.\n'.format( image_types))


#%%  msdNet

ch_in= 3
ch_out = 2
depth =args.depth 
width = args.width
num_epoch = args.epoch
epoch_n = 20

#%%


for target_key in training_types:
    
        
    print('Reading from data folder: ', data_dir)
    traindata = data.TrainDf(data_dir, target_key= target_key )

    for imtype in image_types:
                
        
        if imtype == 'chrom':
            pdSeries = traindata.df.query('chromatic==True').reset_index()
        elif imtype == 'monochrom':
            pdSeries = traindata.df.query('chromatic == False').reset_index()
        else:
            raise Exception('The image type isn\'t right.')
            
        # epoch number
#        tsfm = transforms.Compose( ( data.RandomCrop(target_key, 255), data.ToTensor(target_key) ))
        tsfm = data.ToTensor(target_key)
        ds = data.NucleiDataset(pdSeries, key = target_key, transform= tsfm)
        dataloader = DataLoader( ds , batch_size=1,
                                shuffle=True, num_workers=8)


        whatsgoingon =  imtype+'_'+stage+'_'+target_key
        
        dir_wgo = os.path.abspath( os.path.join( temp_dir, whatsgoingon))
        
        try:
            assert os.path.exists( dir_wgo)
        except AssertionError:
            os.mkdir( dir_wgo)            
            
        # % record the loss, and the output for every 30 epochs
        
        msdNet = msdModule.msdSegModule(ch_in, ch_out, depth, width)

        loss_list = msdNet.train(dataloader, num_epochs= num_epoch, target_key= target_key, savefigures = True, num_fig= 10, save_dir = dir_wgo  )
          
        loss_pd = pd.DataFrame( data = {'Iteration': list(range(len(loss_list))), 'Loss': loss_list})
        loss_pd.to_csv( os.path.join( dir_wgo, 'loss.csv'))
         # save the loss variation plot
        fig, ax = plt.subplots(1)
        ax.plot( loss_list)
        ax.set_ylabel('BCE loss')
        ax.set_xlabel('Epoch')
        fig.savefig( os.path.join(dir_wgo, 'loss.png'))
        
        #% save the trained network
        msdNet.save_network(dir_wgo, 'msdNet.pytorch')
        print('msdNet of ' + whatsgoingon +' has been saved.' )

sys.stdout = orig_stdout
f.close()
#%% load the network and apply 
msdNet_rl = msdModule.msdSegModule(ch_in, ch_out, depth, width)

msdNet_rl.load_network(dir_wgo,'msdNet.pytorch' )
msdNet_rl.validate(dataloader, target_key)
msdNet_rl.save_output(os.path.join(dir_wgo, 'output_{}.png'.format( 'test')))
msdNet_rl.save_input(os.path.join(dir_wgo, 'input_{}.png'.format( 'test')))
msdNet_rl.save_target(os.path.join(dir_wgo, 'target_{}.png'.format( 'test')))
