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
from scipy import misc

IMG_CHANNELS = 3

from nuclei.utils.image import isChromatic
from nuclei.utils import postprocess


__all__ = ('image_save','TrainDf','TestDf','NucleiDataset', 'Rescale', 'RandomCrop', 'ToTensor' )

def image_save(filename, imarr, image_type ='input'):
    #print('Image size: {}'.format(imarr.shape))
    
    try:
        assert len(imarr.shape) == 3 or len(imarr.shape) ==2
    except:
        print('Can\'t save {}.'.format(image_type), ' Image size: {}'.format(imarr.shape))
        return None
        
    if len(imarr.shape) == 3:
        try:
            assert imarr.shape[2] == 2 or imarr.shape[2] == 3
        except:
            imarr = np.moveaxis(imarr, 0, 2)
            try:
                assert imarr.shape[2] == 2 or imarr.shape[2] == 3
            except:
                raise Exception('Image shape wrong. ')
    if imarr.dtype == np.float_:
        imarr -= imarr.min()
        imarr /= imarr.max()
    if len(imarr.shape) == 2:
        misc.imsave(filename, imarr)
    elif imarr.shape[2] == 3:
        misc.imsave(filename, imarr)
    elif imarr.shape[2] == 2:
        misc.imsave(filename, np.concatenate((imarr[:,:,0].squeeze(), imarr[:,:,1].squeeze())))


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

def _read_and_erode( mask_img):
    msk = io.imread(mask_img) /255.0
    
    assert msk.max() == 1
    
    n_erode = 0
    while n_erode < 2 and msk.sum() > 25:
          msk = morphology.erode(msk)
    
    return msk
    
def _stack_eroded_masks( in_img_list):
    return np.sum(np.stack([ _read_and_erode(im) for im in in_img_list]), 0)
    

# --------
    # reading all types of iamges

def _read_images(dSeries):
    print('Run process (%s)... for reading images' % ( os.getpid()))
    return dSeries.map(_read_and_stack).map( lambda x: x[:,:,:IMG_CHANNELS].astype('float32'))

def _read_bw_images( dSeries):
    print('Run process (%s)... for reading b\& w images' % ( os.getpid()))
    return dSeries.map(_read_and_stack).map( lambda x: x.astype(float))

def _get_boundaries( dSeries):
    print('Run process (%s)... for geting boundaries' % ( os.getpid()))
    return dSeries.map( _stack_boundary).map( lambda x: morphology.dilation( x.astype(float)) )


def _get_centroids( dSeries):
    print('Run process (%s)... for geting centroids' % ( os.getpid()))
    return dSeries.map( _stack_centroid_images).map( lambda x:  _multidilation(x, 2))

def _get_masks( dSeries):
    print('Run process (%s)... for reading masks' % ( os.getpid()))
    return dSeries.map(_read_and_stack).map(lambda x: x.astype(float))

def _get_eroded_masks( dSeries):
    print('Run process (%s)... for reading and eroding masks' % ( os.getpid()))
    return dSeries.map(_stack_eroded_masks).map(lambda x: x.astype(float))

class TrainDf(object):
    
    def __init__( self, data_dir, target_key):
        """
        read the data from folder
        ------------
        Par : 
            data_dir: data directory
            target_type: str
                'merged_masks' or 'eroded_masks'
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
          c_row['images'] = n_rows.query('ImageType=="images"')['path'].values.tolist()
          c_row['merged_masks'] = n_rows.query('ImageType=="merged_masks"')['path'].values.tolist()
          c_row['eroded_masks'] = n_rows.query('ImageType=="eroded_masks"')['path'].values.tolist()
          # a list containing the dictionary of {'ImageId', 'Stage', 'masks', 'images', 'eroded_masks', 'merged_masks'}
          # and the paths
          train_rows += [c_row]
          
        train_img_df = pd.DataFrame(train_rows)

        
        train_img_df['images'] = _read_images(train_img_df['images'])
        
        print( 'Reading {}...'.format( target_key))
        train_img_df[target_key] = _read_bw_images(train_img_df[target_key])    

        print('Done. {} data have been read.'.format(train_img_df.size))
        print('Reading time: ', time.time() - clock)
        
        train_img_df['chromatic'] = train_img_df['images'].map( isChromatic )  

        self.df = train_img_df
        self.labels = train_labels
        
    def count(self):
        """
        Print the data informations
        """
        self.df['images'].map( lambda x: x.shape).value_counts()
        print('Chromatic images: ')
        self.df_chromatic['images'].map( lambda x: x.shape).value_counts()
        print('Monochromatic images: ')
        self.df_monochrom['images'].map( lambda x: x.shape).value_counts()
        

  
class TestDf(object):
    def __init__( self, data_dir,stage_label = 'stage1', cut_off = 0.5):
        """
        read the data from folder
        ------------
        Par : 
            data_dir: data directory
        """       
        #% load the input label csv
        self.set_cut_off(cut_off)
        
        allimagepath = glob.glob(os.path.join( data_dir, '{}_test'.format(stage_label),
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
        
        #% load the test data
        test_df = img_df.query('TrainingSplit=="test"')
        
        test_rows = []
        
        group_cols = ['Stage', 'ImageId']
        
        for n_group, n_rows in test_df.groupby(group_cols):
        
          # get the n_group into a dictionary
          c_row = {col_name: col_v for col_name, col_v, in zip(group_cols, n_group)}
          
          # get a list of the path for masks
          c_row['images'] = n_rows.query('ImageType=="images"')['path'].values.tolist()
          
          test_rows += [c_row]
          
        test_img_df = pd.DataFrame(test_rows)
    
        
        test_img_df['images'] = _read_images(test_img_df['images'])
               
        test_img_df['chromatic'] = test_img_df['images'].map( isChromatic )  
        
        self.df = test_img_df
    
    def set_cut_off(self, cut_off):
        self._cut_off = cut_off
    
    @staticmethod
    def _to_tensor(image):
        image = torch.from_numpy(np.expand_dims( np.moveaxis(image, -1, 0), axis = 0))
        return image
    
    @staticmethod
    def _to_numpy_bw(tensor):
        image = tensor.data.cpu().numpy()[:,1,:,:].squeeze()
        return image
        
    def predict_merged_masks(self, network_monochrom, network_chrom, set_clean = False):
        
        masks_list= []
        if set_clean:
            for ir in range(self.df.shape[0]):
                if self.df.iloc[ir]['chromatic']:
                    masks_list.append( TestDf._to_numpy_bw( network_chrom.predict( TestDf._to_tensor( self.df.iloc[ir]['images']))))
                else:
                    masks_list.append(  TestDf._to_numpy_bw( network_monochrom.predict( TestDf._to_tensor( self.df.iloc[ir]['images']))))
        else:
            for ir in range(self.df.shape[0]):
                if self.df.iloc[ir]['chromatic']:
                    masks_list.append( self._clean_mask( TestDf._to_numpy_bw( network_chrom.predict( TestDf._to_tensor( self.df.iloc[ir]['images'])))))
                else:
                    masks_list.append( self._clean_mask( TestDf._to_numpy_bw( network_monochrom.predict( TestDf._to_tensor( self.df.iloc[ir]['images'])))))
                    
        self.df=self.df.assign(merged_masks = pd.Series( masks_list).values, index = self.df.index)
#        self.df.query('chromatic==False')['merged_masks'] = self.df.query('chromatic==False')['images'].map(TestDf._to_tensor).map(network_monochrom.predict).map(TestDf._to_numpy).map(self._clean_mask)
#        self.df.query('chromatic==True')['merged_masks'] = self.df.query('chromatic==True')['images'].map(TestDf._to_tensor).map(network_chrom.predict).map(TestDf._to_numpy).map(self._clean_mask)

    
    def predict_eroded_masks(self, network_monochrom, network_chrom):
        masks_list= []
        for ir in range(self.df.shape[0]):
            if self.df.iloc[ir]['chromatic']:
                masks_list.append( self._clean_mask(TestDf._to_numpy_bw( network_chrom.predict( TestDf._to_tensor( self.df.iloc[ir]['images'])))))
            else:
                masks_list.append(  self._clean_mask(TestDf._to_numpy_bw( network_monochrom.predict( TestDf._to_tensor( self.df.iloc[ir]['images'])))))
        
        self.df=self.df.assign(eroded_masks = pd.Series( masks_list).values, index = self.df.index)
        
    def _clean_mask(self, mask):
        return postprocess.clean_mask(mask, self._cut_off)
    

        
    def save_images(self, output_dir, image_type ='merged_masks'):
        """
        Args: str
            image_type: 'merged_masks', 'eroded_masks', 'images'
        """
        image_dir = os.path.join(output_dir,image_type)
        if not os.path.exists(image_dir):
            os.mkdir( image_dir)
        for idex, c_row in self.df.iterrows():
            image_save( os.path.join( image_dir, c_row['ImageId']+'.png'), c_row[image_type])
            
            
    
    @staticmethod
    def _turn2label(df_tmp):
        cb_list = []
        for ir in range(df_tmp.shape[0]):
            cb_list.append( postprocess.combined_label(df_tmp.iloc[ir]['merged_masks'], df_tmp.iloc[ir]['eroded_masks']))
        return cb_list
    
    @staticmethod
    def _combined_rle( lableimg):
        return postprocess.label2rle(lableimg )[0]
    
    def lable_masks(self, method = 'combined'):
  
        self.df['labled_masks'] = TestDf._turn2label(self.df)
    
    def generate_rle(self, method = 'combined'):   
        self.df['rle'] = self.df['labled_masks'].map( TestDf._combined_rle)
    
class NucleiDataset(Dataset):
    """ chromatic images dataset. """
    def __init__(self, df, key = 'merged_masks', transform = None):
        '''
         Args:
             key: str or None. Can 'merged_masks' or 'eroded_masks'
        '''
        self.key = key
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
      
    def __getitem__(self, index):
        images = self.df.iloc[index]['images'].astype('float32')
        target = self.df.iloc[index][self.key].astype('float32') # np.moveaxis(,  -1, 0)
        target_2ch = np.stack( (1 - target, target), axis=2)
        
        sample = {'images': images, self.key: target_2ch}
        if self.transform:
                sample = self.transform(sample)
    
        return sample
      
     
class Rescale(object):
    """ Rescale the images in a sample
    -----
    Par :
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size, key):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.key = key
        
    def __call__(self, sample):
        image, target = sample['images'], sample[self.key]
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)
        
        new_image = transform.resize(image, (new_h, new_w, 3))
        new_target = transform.resize(target, (new_h, new_w, target.shape[-1]))
        
        return {'images': new_image, self.key: new_target}
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self,  key , output_size = None):
        
        if output_size is not None:
            assert isinstance(output_size, (int, tuple))
            
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2
            self.output_size = output_size
            
        self.key = key

    def __call__(self, sample):
        
        image, target = sample['images'], sample[self.key]
        
        h, w = image.shape[:2]
        
        if self.output_size is None:
            new_h, new_w = image.shape[0]//2, image.shape[1]//2
        else:
            new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w,
                      :]

        target = target[top: top + new_h,
                      left: left + new_w,
                      :]
        
        return {'images': image, self.key: target}

class GaussianNoise(object):
    """ Add random Gaussian noise to the images
    
    """
    def __int__(self, key, eps = 0.001):
        self.eps = eps
        self.key = key
        
    def __call__(self, sample):
        image =  sample['images']
        
        return {'images': image, self.key:  sample[self.key]}
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,  key):
        self.key = key
        
    def __call__(self, sample):
        image, target = sample['images'], sample[self.key]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.moveaxis(image, -1, 0)
        target = np.moveaxis(target, -1, 0)
        return {'images': torch.from_numpy(image),
                self.key: torch.from_numpy(target)}

        
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
