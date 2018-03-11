# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:50:26 2018

@author: zhong
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, transform
import pandas as pd
import time, os, glob, platform
from utils.image import isChromatic

import unitest
IMG_CHANNELS = 3

__all__ = ('trainDf', 'chromaticDataset' )

def _read_and_stack(in_img_list):
      return np.sum(np.stack([ io.imread(c_img) for c_img in in_img_list]), 0) / 255.0

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
        
        print(train_labels.sample()['EncodedPixels'])
        
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
        
        # read the images and crop from RGBA channels to RGB images
        train_img_df['images'] = train_img_df['images'].map(_read_and_stack).map( lambda x: x[:,:,:IMG_CHANNELS])
        train_img_df['masks'] = train_img_df['masks'].map(_read_and_stack).map(lambda x: x.astype(int))
    
        # add a column indicating if the images are chromatic or not
        train_img_df['chromatic'] = train_img_df['images'].map( isChromatic )  
    
        print('Reading time: ', time.time() - clock)
        
        self.df = train_img_df
        self.df_chromatic = train_img_df.query('chromatic==True')
        self.df_monochrom = train_img_df.query('chromatic==False')
    
    def count(self):
        self.df['images'].map( lambda x: x.shape).value_counts()
        print('Chromatic images: ')
        self.df_chromatic['images'].map( lambda x: x.shape).value_counts()
        print('Monochromatic images: ')
        self.df_monochrom['images'].map( lambda x: x.shape).value_counts()

class ChromaticDataset(Dataset):
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

        landmarks = landmarks - [left, top]

        return {'images': image, 'masks': mask}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['images'], sample['masks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'images': torch.from_numpy(image),
                'masks': torch.from_numpy(landmarks)}
        
class TestData(unitest.TestCase):
    
    def test_Rescale(self):
        rcl = rescale()
       
if __name__ = '__main__':
    