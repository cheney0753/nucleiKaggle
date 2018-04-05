#%matplotlib qt
isCUDA = True
#isTest = False
#%% Import modules
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("-t","--test", help="set the program in a test mode with few data and expochs.",  action="store_true")
parser.add_argument("-e","--epoch",help="number of erodions for detaching the masks", default = 150, type = int)

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
now = datetime.datetime.now()

from skimage import io, transform, util, morphology
import platform
import pandas as pd
#%matplotlib agg
#plt.ioff()
#%%
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure
#from sklearn import cluster
#%%
IMG_CHANNELS = 3

  

def _read_and_erode( c_img):
    msk = io.imread(c_img) /255.0
    
    assert msk.max() == 1
    
    n_erode = 0
    while n_erode < 4 and msk.sum() > 100:
          msk = morphology.erosion(msk)
          n_erode += 1
    return msk
    
def _read_and_stack_eroded_masks( in_img_list):
    return np.sum(np.stack([ _read_and_erode(im) for im in in_img_list]), 0)

def _read_and_stack(in_img_list):
    return np.sum(np.stack([ io.imread(c_img) for c_img in in_img_list]), 0) / 255.0

def _read_images(dSeries):
    print('Run process (%s)... for reading images' % ( os.getpid()))
    return dSeries.map(_read_and_stack).map( lambda x: x[:,:,:IMG_CHANNELS])

def _read_bw_images(dSeries, f):
    print('Run process (%s)... for reading b\& w images' % ( os.getpid()))
    return dSeries.map(f).map( lambda x: x[:,:])

def _read_er_images(dSeries):
    print('Run process (%s)... for reading b\& w images' % ( os.getpid()))
    return dSeries.map(_read_and_erode)

#%% prepare the data 
#cwdir = os.path.abspath(os.path.dirname(__file__))
cwdir = %pwd
data_dir  = r'C:\Users\zhong\Documents\zhong_personal\works\kaggle\nuclei\data_sample'
#os.path.join( cwdir, os.pardir, 'data')

stage_label = 'stage1'

clock = time.time()

#% load the input label csv           
#train_labels = pd.read_csv(
#    os.path.join(data_dir, '{}_train_labels.csv'.format(stage_label)))
#
#train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda en: [x for x in en.split(' ')] )

print('Reading ' + '{}_train_labels.csv'.format(stage_label))

#% load the training files 
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

#img_df.sample(2)

#% load the traing data
train_df = img_df.query('TrainingSplit=="train"')
#%%
train_rows = []

group_cols = ['Stage', 'ImageId']

for n_group, n_rows in train_df.groupby(group_cols):

  # get the n_group into a dictionary
  c_row = {col_name: col_v for col_name, col_v, in zip(group_cols, n_group)}
  
  # get a list of the path for masks
  c_row['masks'] = n_rows.query('ImageType=="masks"')['path'].values.tolist()
  c_row['images'] = n_rows.query('ImageType=="images"')['path'].values.tolist()
  
  # create a directory for saving the merged masks and eroded masks
  c_row['merged_masks'] = os.path.abspath(os.path.join(
          os.path.abspath( os.path.dirname(n_rows.query('ImageType=="images"')['path'].values.tolist()[0])),
                                           os.pardir, 'merged_masks'))
  if not os.path.exists( c_row['merged_masks']):
      os.mkdir(c_row['merged_masks'] )
  
  merged_mask = _read_and_stack(c_row['masks'])
  
  io.imsave( os.path.join(c_row['merged_masks'], c_row['ImageId']+'.png'), (255*merged_mask).astype(np.int_))
  
  c_row['eroded_masks'] = os.path.abspath(os.path.join(
          os.path.abspath( os.path.dirname(n_rows.query('ImageType=="images"')['path'].values.tolist()[0])),
                                           os.pardir, 'eroded_masks'))
  
  if not os.path.exists( c_row['eroded_masks']):
      os.mkdir(c_row['eroded_masks'] )

  eroded_masks = _read_and_stack_eroded_masks(c_row['masks'])
  
  io.imsave( os.path.join(c_row['eroded_masks'], c_row['ImageId']+'.png'), (255*eroded_masks).astype(np.int_))
    
  
  # a list containing the dictionary of {'ImageId', 'Stage', 'masks', 'images'}
  # and the paths
  train_rows += [c_row]

#%%  
train_img_df = pd.DataFrame(train_rows)
 

#train_img_df['images'] = _read_images(train_img_df['images'])
# train_img_df['boundaries'] = self._get_boundaries(train_img_df['masks'])
# train_img_df['centroids'] = self._get_centroids(train_img_df['masks'])

print( 'Getting merged masks...')
#train_img_df['masks'] = self._get_masks(train_img_df['masks'])    
train_img_df['merged_masks'] = _read_bw_images(train_img_df['masks'], _read_and_stack)    

print( 'Getting eroded masks ...')
train_img_df['eroded_masks'] = _read_bw_images(train_img_df['masks'], _read_and_stack_eroded_masks)


print('Done.')
print('Reading time: ', time.time() - clock)

#train_img_df['chromatic'] = train_img_df['images'].map( isChromatic )  


















#%%



temp_dir = os.path.join( cwdir, os.pardir, 'temp', '%d%d%d'%(now.year, now.month, now.day))

try:
    assert os.path.exists(temp_dir)
except AssertionError:
    os.mkdir(temp_dir)
    
orig_stdout = sys.stdout

f_stdout = os.path.join(temp_dir, 'stdout.txt')
f=open( f_stdout, 'w')

print('Print to: ',f_stdout)
sys.stdout = f

    
print('Reading from data folder: ', data_dir)
traindata = data.TrainDf(data_dir)





stage = 'train'
#training_types = ('masks', 'centroids')
training_types = ('masks',)
image_types = ('monochrom', 'chrom')

#%%  msdNet

ch_in= 3
ch_out = 2
depth =30 
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
