#-*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:50:26 2018

@author: zhong
"""

from __future__ import absolute_import

from multiprocessing import Pool, Process

import numpy as np

from skimage import io, transform, util, morphology
from scipy import ndimage as ndi

import torch
from torch.utils.data import Dataset

import pandas as pd
import time, os, glob, platform
import unittest

IMG_CHANNELS = 3

from nuclei.utils.image import isChromatic



__all__ = ('TrainDf', 'NucleiDataset', 'Rescale', 'RandomCrop', 'ToTensor' )

def _multidilation(x, n):
    for i in range(n):
        x = morphology.dilation(x)
    return x

def _read_and_stack(in_img_list):
    return np.sum(np.stack([ io.imread(c_img) for c_img in in_img_list]), 0) / 255.0
  
    
def _read_and_get_boundary( mask_img):
    msk = io.imread(mask_img)>0
    return (msk^morphology.binary_erosion(msk)).astype(int)

def _stack_boundary(in_img_list):
    return (np.sum( np.stack( [_read_and_get_boundary(c_img) for c_img in in_img_list]), 0) > 0 ).astype(float)
    

def _read_and_get_centroid_image( mask_img):
    msk = io.imread(mask_img) /255.0
#    msk = msk/ msk.max()
    
    nz =  np.nonzero( msk )
    
    ctr = [int(nzi.mean()) for nzi in nz]
#    print(ctr)
    img = np.zeros( msk.shape, dtype= float)
    
    img[ctr[0], ctr[1]] = 1.0
    return img

def _stack_centroid_images( in_img_list):
    return np.sum(np.stack( [ _read_and_get_centroid_image(im) for im in in_img_list]), 0 ) 



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

def watershed_label(mask, labled_ct):
    """
    watershed labeling using the predicted centroids
    """
    distance = ndi.distance_transform_edt(mask)
    
    #markers = ndi.label(local_maxi)
    labels = morphology.watershed(-distance, labled_ct, mask=mask)
    
    return labels


def rle_fromImage(x, centroids, cut_off = 0.5):
    
    lab_img = watershed_label((x>cut_off).astype(float), centroids)
    
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
        
    return [rle_encoding(lab_img==i) for i in range(1, lab_img.max()+1)]

    
class TrainDf(object):
    
    def __init__( self, data_dir):
        """
        read the data from folder
        ------------
        Par : 
            data_dir: data directory
            
        Return:
            pandas.DataFrame: {stage: }
        """
        # record a clock time
        clock = time.time()
        
        stage_label = 'stage1'
        
        #% load the input label csv
        
        train_labels = pd.read_csv(
            os.path.join(data_dir, '{}_train_labels.csv'.format(stage_label)))
        
        train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda en: [x for x in en.split(' ')] )
        
        print('Reading ' + '{}_train_labels.csv'.format(stage_label))
        
#        print(train_labels.sample()['EncodedPixels'])
        
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
        
        train_rows = []
        
        group_cols = ['Stage', 'ImageId']
        
        for n_group, n_rows in train_df.groupby(group_cols):
        
          # get the n_group into a dictionary
          c_row = {col_name: col_v for col_name, col_v, in zip(group_cols, n_group)}
          
          # get a list of the path for masks
          c_row['masks'] = n_rows.query('ImageType=="masks"')['path'].values.tolist()
          c_row['images'] = n_rows.query('ImageType=="images"')['path'].values.tolist()
          
          # a list containing the dictionary of {'ImageId', 'Stage', 'masks', 'images'}
          # and the paths
          train_rows += [c_row]
          
        train_img_df = pd.DataFrame(train_rows)
        
#        
#        pool = Pool(processes = 4)
#        ## oops!! data to large to synchronize
#        
#        # using multi-processing to process them
#
##        print ('Reading images.')
#        # read the images and crop from RGBA channels to RGB images
#        result1 = pool.apply_async(self._read_images, ( train_img_df['images'],))
##        print('Getting boundary images...')
#        result2 = pool.apply_async(self._get_boundaries, ( train_img_df['masks'],))    
##        print( 'Getting dilated centroids...')
#        result3 = pool.apply_async(self._get_centroids, ( train_img_df['masks'],))
#        
#        train_img_df['images'] = result1.get(timeout=99999)
#        train_img_df['boundaries'] = result2.get(timeout = 99999)
#        train_img_df['centroids'] = result3.get(timeout = 99999)
#
#        pool.close()
#        pool.join()
#        
        
        train_img_df['images'] = self._read_images(train_img_df['images'])
        train_img_df['boundaries'] = self._get_boundaries(train_img_df['masks'])
        train_img_df['centroids'] = self._get_centroids(train_img_df['masks'])
        
        print( 'Getting overlapped masks...')
        train_img_df['masks'] = self._get_masks(train_img_df['masks'])    
        
        print('Done.')
        print('Reading time: ', time.time() - clock)
        
        train_img_df['chromatic'] = train_img_df['images'].map( isChromatic )  

        self.df = train_img_df
        self.labels = train_labels
        
    def _read_images(self, dSeries):
        print('Run process (%s)... for reading images' % ( os.getpid()))
        return dSeries.map(_read_and_stack).map( lambda x: x[:,:,:IMG_CHANNELS])

    def _get_boundaries(self, dSeries):
        print('Run process (%s)... for geting boundaries' % ( os.getpid()))
        return dSeries.map( _stack_boundary).map( lambda x: morphology.dilation( x.astype(float)) )


    
    def _get_centroids(self, dSeries):
        print('Run process (%s)... for geting centroids' % ( os.getpid()))
        return dSeries.map( _stack_centroid_images).map( lambda x:  _multidilation(x, 2))

    def _get_masks(self, dSeries):
        print('Run process (%s)... for reading masks' % ( os.getpid()))
        return dSeries.map(_read_and_stack).map(lambda x: x.astype(float))


    def count(self):
        """
        Print the data informations
        """
        self.df['images'].map( lambda x: x.shape).value_counts()
        print('Chromatic images: ')
        self.df_chromatic['images'].map( lambda x: x.shape).value_counts()
        print('Monochromatic images: ')
        self.df_monochrom['images'].map( lambda x: x.shape).value_counts()
        
        
        
        
        
        
        
        
        
        

class NucleiDataset(Dataset):
  """ chromatic images dataset. """
  
  def __init__(self, df, transform = None):
    self.df = df
    self.transform = transform
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    images = np.moveaxis(self.df.iloc[index]['images'].astype('float32'), -1, 0)
    
    mask_ = self.df.iloc[index]['masks'].astype('float32') # np.moveaxis(,  -1, 0)
    #mask_ = self.df.iloc[index]['masks'].astype(np.int_) # np.moveaxis(,  -1, 0)

    masks_2ch = np.stack( (1 - mask_, mask_), axis=0)
    
    centroids = self.df.iloc[index]['centroids'].astype('float32')
    centroids_2ch = np.stack( (1 - centroids, centroids), axis=0)

    return (images, masks_2ch)
  
     
class Rescale(object):
    """ Rescale the image in a sample
    -----
    Par :
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, mask = sample['images'], sample['masks']
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)
        
        new_image = transform.resize(image, (new_h, new_w))
        new_mask = transform.resize(mask, (new_h, new_w))
        
        return {'images': new_image, 'masks': new_mask}
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['images'], sample['masks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]
        return {'images': image, 'masks': mask}

class GaussianNoise(object):
    """ Add random Gaussian noise to the images
    
    """
    def __int__(self, eps = 0.001):
        self.eps = eps
    
    def __call__(self, sample):
        image =  sample['images']
        
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['images'], sample['masks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'images': torch.from_numpy(image),
                'masks': torch.from_numpy(mask)}

        
class TestData(unittest.TestCase):

    test_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)), os.pardir, 'testdata/2c840a94d216f5ef4e499b53ae885e9b022cbf639e004ec788436093837823b2/masks')

#    def test_Rescale(self):
#        rcl = Rescale(100)
       
    def test_stack_boundary(self):
        img_list = [ os.path.join( self.__class__.test_dir, img) for img in  os.listdir( self.__class__.test_dir) ]
        self.assertAlmostEqual( _stack_boundary(img_list).sum(), 3067) # 3067 is the the sum of pixels for this case
        
    def test_stack_centroid_images(self):
        img_list = [ os.path.join( self.__class__.test_dir, img) for img in  os.listdir( self.__class__.test_dir) ]

        self.assertAlmostEqual( _stack_centroid_images( img_list).sum(), len(img_list))
        
if __name__ == '__main__':
    unittest.main()    
