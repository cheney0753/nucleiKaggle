#%matplotlib qt
isCUDA = True
#isTest = False
#%% Import modules
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("-t","--test", help="set the program in a test mode with few data and expochs.",  action="store_true")
parser.add_argument("-odir","--output_dir",help="output dir", type = str)
parser.add_argument("-ndir","--network_dir", 
                    help="network dir: must have 4 sub directories:chrom_eroded_masks  chrom_merged_masks  monochrom_eroded_masks  monochrom_merged_masks", 
                    type = str)
parser.add_argument("-idir","--input_dir", help = "input dir", type = str)
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
import os, sys, time, glob
import datetime
from skimage import io, transform, util, morphology
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure

from nuclei.utils import data
from nuclei.kernel import msdModule
#from sklearn import cluster

now = datetime.datetime.now()

#%%
#%% prepare the data 
#cwdir = os.path.abspath(os.path.dirname(__file__))
data_dir  = r'/export/scratch1/zhong/PhD_Project/Projects/Kaggle/nuclei/data'

if args.output_dir is None:
    output_dir=  os.path.abspath( os.path.join( os.path.dirname( __file__) , os.pardir, 'temp','predict-{}'.format( datetime.datetime.now().date())))
else:
    output_dir = args.output_dir

if args.input_dir is None:
    input_dir=  os.path.abspath( os.path.join( os.path.dirname( __file__) , os.pardir, 'data'))
else:
    input_dir = args.input_dir

    
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
if not os.path.exists(input_dir):
    os.mkdir(input_dir)


network_dir=  '/export/scratch1/zhong/PhD_Project/Projects/Kaggle/nuclei/networks/0410' #args.network_dir

print('The directories are: ')
print('Output: {}'.format(output_dir))
print('Input: {}'.format(input_dir))
print('Network: {}'.format(network_dir))

f_nw_list = ['chrom_merged_masks',
                 'monochrom_merged_masks',
                 'chrom_eroded_masks',
                 'monochrom_eroded_masks']

#%% predict merged masks
# chrom_merged_masks

network_list = {}
for f_nw in f_nw_list:
    dir_curr =os.path.join( network_dir, f_nw)
    with open( os.path.join( dir_curr, 'stdout.txt'), 'r') as f:
        readin = [f.readline() for i in range(8)]

    pars_read = [int( readin[3+i].strip().split(':')[-1]) for i in range(4)]
    
    # respectively are c_int c_out  depth width
    print('Reading net work {}.'.format(f_nw), 'Network shape is {}'.format( pars_read))

    nw_curr = msdModule.msdSegModule(*pars_read)

    try:
        nw_curr.load_network( save_dir =  dir_curr, fname = 'msdNet.pytorch')
    except:
        raise Exception('Reading failed.')
    network_list[f_nw] = nw_curr

    
#%%    
# monochrom_merged_masks

testDf = data.TestDf(input_dir, cut_off= 0.5)

#testDf.save_images( output_dir, 'images')
#%%

testDf.predict_merged_masks(network_list['monochrom_merged_masks'], network_list['chrom_merged_masks'])

#testDf.save_images( output_dir, 'merged_masks')

#%%

testDf.predict_eroded_masks(network_list['monochrom_eroded_masks'], network_list['chrom_eroded_masks'])

#testDf.save_images( output_dir, 'eroded_masks')

#%%  get rle
testDf.lable_masks()

#%% 
#from skimage.measure import label
#testDf.df['labled_masks'] = testDf.df['eroded_masks'].map( label )
#%%
testDf.generate_rle()

#%%

out_pred_list = []
for _, c_row in testDf.df.iterrows():
    for c_rle in c_row['rle']:
        out_pred_list+=[dict(ImageId=c_row['ImageId'], 
                             EncodedPixels = ' '.join(np.array(c_rle).astype(str)))]
out_pred_df = pd.DataFrame(out_pred_list)

out_pred_df[['ImageId', 'EncodedPixels']].to_csv(os.path.join( output_dir, 'predictions.csv'), index = False)