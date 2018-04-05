#%matplotlib qt
isCUDA = True
#isTest = False
#%% Import modules
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("-t","--test", help="set the program in a test mode with few data and expochs.",  action="store_true")
parser.add_argument("-numerd", "--number-erosion", help = "set the number of erosion", type = int, default = 2)
parser.add_argument("-minerd", "--minimum-erosion", help = "set the minimal number of pixels after erosion", type = int, default = 100)

args = parser.parse_args()

isTest = args.test
if args.test:
    print('In the test mode.')

import numpy as np
import os, time, glob
import datetime
from skimage import io, morphology
import platform
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure
from matplotlib import pyplot as plt
#from sklearn import cluster
#%%

now = datetime.datetime.now()
plt.switch_backend('agg')
plt.ioff()
IMG_CHANNELS = 3
 

def _read_and_erode( c_img):
    msk = io.imread(c_img) /255.0
    
    assert msk.max() == 1
    
    n_erode = 0
    while n_erode < 4 and msk.sum() > 100:
          msk_tmp = morphology.erosion(msk)
          if msk_tmp.sum() >100:
              msk = msk_tmp
              n_erode += 1
          else:
              break
    return msk
    
def _read_and_stack_eroded_masks( in_img_list):
    return np.sum(np.stack([ _read_and_erode(im) for im in in_img_list]), 0)

def _read_and_stack(in_img_list):
    return np.sum(np.stack([ io.imread(c_img) / 255.0 for c_img in in_img_list]), 0)


#%% prepare the data 
cwdir = os.path.abspath(os.path.dirname(__file__))
#cwdir = %pwd
#data_dir  = r'C:\Users\zhong\Documents\zhong_personal\works\kaggle\nuclei\data_sample'
if isTest:
    data_dir = os.path.join( cwdir, os.pardir, 'data_sample')
else:
    data_dir = os.path.join( cwdir, os.pardir, 'data')
    
print('Wring merged masks and eroded masks to: ', os.path.abspath( data_dir))    
    
stage_label = 'stage1'

clock = time.time()

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
  
  fn =  os.path.join(c_row['merged_masks'], c_row['ImageId']+'.png')
  io.imsave(fn, (255*merged_mask).astype(np.int_))
  
  c_row['eroded_masks'] = os.path.abspath(os.path.join(
          os.path.abspath( os.path.dirname(n_rows.query('ImageType=="images"')['path'].values.tolist()[0])),
                                           os.pardir, 'eroded_masks'))
  
  if not os.path.exists( c_row['eroded_masks']):
      os.mkdir(c_row['eroded_masks'] )

  eroded_masks = _read_and_stack_eroded_masks(c_row['masks'])
  fn =  os.path.join(c_row['eroded_masks'], c_row['ImageId']+'.png')
  io.imsave(fn, (255*eroded_masks).astype(np.int_))
    

print('runtime is: ', time.time() - clock)

