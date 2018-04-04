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

args = parser.parse_args()
isTest = args.test
if args.test:
    print('In the test mode.')

#isTest = True    
#%%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from matplotlib import pyplot as plt
import os, sys
import datetime
now = datetime.datetime.now()
#%matplotlib agg
plt.ioff()
#%%
from nuclei.kernel import cnnModule
from nuclei.kernel.lossFunc import diceLoss, IoU_mean
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure
from sklearn import cluster
#%%

from nuclei.kernel import msdModule
from nuclei.kernel.lossFunc import crossEntropy2d_sum

from nuclei.utils import data
from nuclei.utils import plot_data

#%% 


#%% prepare the data 
cwdir = os.path.abspath(os.path.dirname(__file__))
data_dir  = os.path.join( cwdir, os.pardir, 'data')
if isTest:
    data_dir = os.path.join(cwdir, os.pardir, 'data_sample')

temp_dir = os.path.join( cwdir, os.pardir, 'temp', '%d%d%d'%(now.year, now.month, now.day))
orig_stdout = sys.stdout

f_stdout = os.path.join(temp_dir, 'stdout.txt')
f=open( f_stdout, 'w')

print('Print to: ',f_stdout)
sys.stdout = f

    
print('Reading from data folder: ', data_dir)
traindata = data.TrainDf(data_dir)

try:
    assert os.path.exists(temp_dir)
except AssertionError:
    os.mkdir(temp_dir)



stage = 'train'
#training_types = ('masks', 'centroids')
training_types = ('masks',)
image_types = ('monochrom', 'chrom')

#%%  msdNet

ch_in= 3
ch_out = 2
depth =40 
if isTest:
    depth = 10

width = 2
num_epoch = args.epoch
if isTest:
    num_epoch = 10
epoch_n = 20
#%%

for trtype in training_types:
    for imtype in image_types:
                
        
        if imtype == 'chrom':
            pdSeries = traindata.df.query('chromatic==True').reset_index()
        elif imtype == 'monochrom':
            pdSeries = traindata.df.query('chromatic == False').reset_index()
        else:
            raise Exception('The image type isn\'t right.')
            
        # epoch number
        
        
        dataloader = DataLoader(data.NucleiDataset(pdSeries), batch_size=1,
                                shuffle=True, num_workers=8)


        whatsgoingon =  imtype+'_'+stage+'_'+trtype
        
        dir_wgo = os.path.join( temp_dir, whatsgoingon)
        
        
        try:
            assert os.path.exists( dir_wgo)
        except AssertionError:
            os.mkdir( dir_wgo)
            
            
        # % record the loss, and the output for every 30 epochs
        
        msdNet = msdModule.msdSegModule(ch_in, ch_out, depth, width)

        loss_list = msdNet.train(dataloader, num_epochs= num_epoch, savefigures = True, num_fig= 10, save_dir = dir_wgo  )
            
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
msdNet_rl.validate(dataloader)
msdNet_rl.save_output(os.path.join(dir_wgo, 'output_{}.png'.format( 'test')))
msdNet_rl.save_input(os.path.join(dir_wgo, 'input_{}.png'.format( 'test')))
msdNet_rl.save_target(os.path.join(dir_wgo, 'target_{}.png'.format( 'test')))

#%%        # example of reusing
#        ch_in= 3
#        ch_out = 2
#        depth = 50
#        width = 2
##        
#msdNet_chro_centroids_reload = msdModule.msdNet(ch_in, ch_out, depth, width)
#msdNet_chro_centroids_reload.load_state_dict( torch.load(  os.path.join( dir_wgo, 'msdNet.pth.tar')))
#

#dir_wgo = '/export/scratch1/zhong/PhD_Project/Projects/Kaggle/nuclei/temp/201841/chrom_train_masks'
#msdNet_chro_centroids_reload = msdModule.msdNet(ch_in, ch_out, depth, width)
#msdNet_chro_centroids_reload.load_state_dict( torch.load(  os.path.join( dir_wgo, 'msdNet.pth.tar')))


#%%


##%% put the data into Variable
#_, train_rle_row = next(traindata.df.sample(1).iterrows())
#  
#train_row_rles = data.rle_fromImage(train_rle_row['masks'])
#
#tl_rles = traindata.labels.query('ImageId=="{ImageId}"'.format(**train_rle_row))['EncodedPixels']
#
#tl_rles = sorted(tl_rles, key = lambda x: int(x[0]))
#
#train_row_rles = sorted(train_row_rles, key = lambda x: x[0])
#
#for tl, tr in zip(tl_rles, train_row_rles):
#    print(tl[0], tr[0])
#match, mismatch = 0, 0
#for img_rle, train_rle in zip(sorted(train_row_rles, key = lambda x: x[0]), 
#                             sorted(tl_rles, key = lambda x: int(x[0]))):
#    for i_x, i_y in zip(img_rle, train_rle):
#        if i_x == int(i_y):
#            match += 1
#        else:
#            mismatch += 1
#print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' % (match, mismatch, 100.0*match/(match+mismatch)))
