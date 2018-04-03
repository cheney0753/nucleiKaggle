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
args = parser.parse_args()
isTest = args.test
if args.test:
    print('In the test mode.')

isTest = True    
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

try:
    assert os.path.exists(temp_dir)
except AssertionError:
    os.mkdir(temp_dir)

traindata = data.TrainDf(data_dir)

chromaticDS = data.NucleiDataset(traindata.df.query('chromatic==True').reset_index())

dataloader = DataLoader(chromaticDS, batch_size=1,
                        shuffle=True, num_workers=8)


stage = 'train'
training_types = ('masks', 'centroids')
image_types = ('chrom', 'monochrom')

#%%  msdNet

ch_in= 3
ch_out = 2
depth =70 
width = 2
epoch = 200
if isTest:
    epoch = 10
epoch_n = 10
#%%

for trtype in training_types:
    for imtype in image_types:
                
        msdNet_Net = msdModule.msdNet(ch_in, ch_out, depth, width )
        
        if imtype == 'chrom':
            pdSeries = traindata.df.query('chromatic==True')
        elif imtype == 'monochrom':
            pdSeries = traindata.df.query('chromatic == False')
        else:
            raise Exception('The image type isn\'t right.')
            
        # epoch number
        
        
        optimizer = optim.Adam(msdNet_Net.parameters())
        optimizer.zero_grad()
        
        whatsgoingon =  imtype+'_'+stage+'_'+trtype
        
        dir_wgo = os.path.join( temp_dir, whatsgoingon)
        
        try:
            assert os.path.exists( dir_wgo)
        except AssertionError:
            os.mkdir( dir_wgo)
            
            
        # % record the loss, and the output for every 30 epochs
        loss_var = []
        for ipc in range(epoch):
            r_loss = 0.0
            
            for ide, (image_, mask_2ch, centroids_2ch) in enumerate(dataloader):  
                optimizer.zero_grad()
              
              #forward
        #        mask_2ch = torch.cat( (1 - mask_, mask_), dim=-3)[None, ...]
                
                
                output_ = msdNet_Net(Variable(image_).cuda())
                
                if trtype == 'masks':
                    expect = mask_2ch
                elif trtype == 'centroids':
                    expect = centroids_2ch
                else:
                    raise Exception('The training type isn\'t right.')
                loss = crossEntropy2d_sum(output_.contiguous(), Variable(expect).cuda())

                loss.backward()
                optimizer.step()
              
                r_loss += loss.data[0]
                
           # print('[%d, %d print('[%d, %d] loss: %.5f'%(ipc + 1, ide + 1, r_loss/(ide+1) ))]')
            
            
            loss_var.append(r_loss/ide)
            
            if ipc%epoch_n == 0 :
                print('[%d, %d] loss: %.5f. time: %s'%(ipc + 1, ide + 1, r_loss/(ide+1), datetime.datetime.now() )) 
                fig = plot_data.plot_network_output(msdNet_Net, pdSeries)
                
                fig.savefig( os.path.join(dir_wgo, 'training_images_{}.png'.format(ipc)))
            
            
         # save the loss variation plot
        
        fig, ax = plt.subplots(1)
        ax.plot( loss_var)
        ax.set_ylabel('BCE loss')
        ax.set_xlabel('Epoch')
        fig.savefig( os.path.join(dir_wgo, 'loss.png'))
        
        #% save the trained network
        torch.save(msdNet_Net.state_dict(), os.path.join( dir_wgo, 'msdNet.pth.tar'))
        print('msdNet of ' + whatsgoingon +' has been saved.' )
sys.stdout = orig_stdout
f.close()
#%%        # example of reusing
#        ch_in= 3
#        ch_out = 2
#        depth = 50
#        width = 2
##        
#msdNet_chro_centroids_reload = msdModule.msdNet(ch_in, ch_out, depth, width)
#msdNet_chro_centroids_reload.load_state_dict( torch.load(  os.path.join( dir_wgo, 'msdNet.pth.tar')))
#

dir_wgo = '/export/scratch1/zhong/PhD_Project/Projects/Kaggle/nuclei/temp/201841/chrom_train_masks'
msdNet_chro_centroids_reload = msdModule.msdNet(ch_in, ch_out, depth, width)
msdNet_chro_centroids_reload.load_state_dict( torch.load(  os.path.join( dir_wgo, 'msdNet.pth.tar')))


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
