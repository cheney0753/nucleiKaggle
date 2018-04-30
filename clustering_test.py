#%matplotlib qt
isCUDA = True
#isTest = True
#%% Import modules
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("-t","--test", help="set the program in a test mode with few data and expochs.",  action="store_true", default = False)
parser.add_argument("-numerd", "--number-erosion", help = "set the number of erosion", type = int, default = 2)
parser.add_argument("-minerd", "--minimum-erosion", help = "set the minimal number of pixels after erosion", type = int, default = 100)

args = parser.parse_args()

isTest = args.test
if args.test:
    print('In the test mode.')
isTest = True
import numpy as np
import os, time, glob, sys
import datetime
from skimage import io, morphology, measure
import platform
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
from skimage import measure
from matplotlib import pyplot as plt
import time
#from sklearn import cluster
#%%
import nuclei.utils.postprocess as pproc

now = datetime.datetime.now()
#plt.switch_backend('QT5Agg')
#plt.ioff()
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

def _read_multiply_stack(in_img_list):
    return np.sum(np.stack([ (i+1)*(io.imread(c_img) / 255.0) for (i, c_img) in enumerate(in_img_list)]), 0)

#%% prepare the data 
cwdir = os.path.abspath(os.path.dirname(__file__))
#cwdir = %pwd
#data_dir  = r'C:\Users\zhong\Documents\zhong_personal\works\kaggle\nuclei\data_sample'
if isTest:
    data_dir = os.path.join( cwdir, os.pardir, 'data_sample')
else:
    data_dir = os.path.join( cwdir, os.pardir, 'data')
    
    
stage_label = 'stage1'


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

train_labels = pd.read_csv(
            os.path.join(data_dir, '{}_train_labels.csv'.format(stage_label)))
        
train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda en: [x for x in en.split(' ')] )


#%%

def compare_rle(tl_rles, train_row_rles):
    
    tl_rles = sorted(tl_rles, key = lambda x: int(x[0]))
    
    train_row_rles = sorted(train_row_rles, key = lambda x: int(x[0]))
    
    for tl, tr in zip(tl_rles, train_row_rles):
#        print(tl[0], tr[0])
        match, mismatch = 0, 0
    for img_rle, train_rle in zip(train_row_rles, tl_rles):
        for i_x, i_y in zip(img_rle, train_rle):
            if int(i_x) == int(i_y):
                match += 1
            else:
                mismatch += 1
    print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' % (match, mismatch, 100.0*match/(match+mismatch)))

    return (match, mismatch, 100.0*match/(match+mismatch))
def get_coordinate( img):
    return np.stack( np.nonzero(img)).transpose()

def label_combined(merged_masks, eroded_masks, window_size = 10):
    # first label the eroded_masks, where connectivity is separated
    assert isinstance( window_size, (int, tuple))
    
    lab_img = measure.label( eroded_masks)
    
    comb_img = lab_img.copy()
    boundaries = (merged_masks - eroded_masks)>0
    
    bd_pixels = get_coordinate( boundaries)
    
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
        
    clock = time.time()
    
    img_shape = eroded_masks.shape
    mask_window = np.zeros( img_shape)
    
    for ip in range(bd_pixels.shape[0]):
        pix = bd_pixels[ip,]
        
        dist = sys.maxsize
        lab = 0 
        
        mask_window.fill(0)
        wr = (max(0 , pix[0]-window_size//2 ), min(img_shape[0], pix[0]+window_size//2 ))
        wc = (max(0 , pix[1]-window_size//2 ), min(img_shape[1], pix[1]+window_size//2 ))

        mask_window[wr[0]:wr[1], wc[0]:wc[1]] = 1
        
        for i in range(1, lab_img.max()+1):
            # apply a masking window to the mask of label i
            mkimg = (lab_img == i) * mask_window
            if mkimg.max()==0:
                continue
            
            pix_lab = get_coordinate( mkimg)

                
            pix_til = np.tile( pix, (pix_lab.shape[0], 1))
            
            dist_arr = pix_til-pix_lab
            dd = (( dist_arr * dist_arr).sum(axis = 1) **0.5 ).sum()/dist_arr.shape[0]

            if dist > dd:
                dist = dd
                lab = i
                
        comb_img[pix[0], pix[1]] = lab
#        print( 'pixel no. {}'.format(ip), ', label: {}'.format(lab), ' Total: {}'.format(bd_pixels.shape[0]))
#        print('Runtime for one pixel: {}'.format(time.time()-clock))
#        clock = time.time()
    print('Runtime for one image: {}'.format(time.time()-clock))
    clock = time.time()
    return comb_img.astype(int)

def img2rle(lab_img):
    return [rle_encoding(lab_img==i) for i in range(1, lab_img.max()+1)]


def rle_combined(merged_masks, eroded_masks):
    
    lab_img = label_combined(merged_masks, eroded_masks)
    
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
        
    rle_list = img2rle( lab_img)
    return (rle_list, lab_img)

def rle_encoding( image):
    """ Encode a binary image to rle"""
    bimage = np.where( image.T.flatten()==1)[0] ## transform the image 
    # The pixels are one-indexed and numbered from top to bottom, then left to right:
    # 1 is pixel (1,1), 2 is pixel (2,1), 
    
    r_length = []
    prev = -2
    for b in bimage:
        if (b > prev+1 ) : r_length += [b+1, 0]
        r_length[-1] +=1
        prev = b
        
    return r_length

#%%
train_rows = []

group_cols = ['Stage', 'ImageId']

result = []
for n_group, n_rows in train_df.groupby(group_cols):
    
    # get the n_group into a dictionary
    c_row = {col_name: col_v for col_name, col_v, in zip(group_cols, n_group)}
  
    # get a list of the path for masks
    c_row['masks'] = n_rows.query('ImageType=="masks"')['path'].values.tolist()
    c_row['images'] = n_rows.query('ImageType=="images"')['path'].values.tolist()
    c_row['merged_masks'] = n_rows.query('ImageType=="merged_masks"')['path'].values.tolist()
    c_row['eroded_masks'] = n_rows.query('ImageType=="eroded_masks"')['path'].values.tolist()
    # create a directory for saving the merged masks and eroded masks
    
    c_row['colored_masks'] = _read_multiply_stack( n_rows.query('ImageType=="masks"')['path'].values.tolist())
    
    merged_masks = (io.imread(c_row['merged_masks'][0])/255).astype(int)
    
    eroded_masks = (io.imread(c_row['eroded_masks'][0])/255).astype(int)
    
    true_rles = train_labels.query('ImageId=="{}"'.format(n_rows['ImageId'].iloc[0]))['EncodedPixels'].values.tolist()
    
#    tl_rles = pproc.img2rle( label_combined( merged_masks, eroded_masks))
    print((measure.label( eroded_masks)).max())
    tl_rles = pproc.img2rle( pproc.combined_label( merged_masks, eroded_masks, window_size=20))
#    cb_rle = pproc.img2rle( c_row['colored_masks'].astype(int))
    print(len(tl_rles), len(true_rles))
    result.append( compare_rle( tl_rles, true_rles))
  
    
#    print('runtime is: ', time.time() - clock)
 