# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:46:19 2018

@author: zhong
"""
from __future__ import absolute_import

import time,sys, os
import numpy as np
from skimage import morphology,io, transform, measure
from scipy import ndimage as ndi

__all__ = ('rle_encoding', 'img2rle', 'compare_rle', 'watershed_label',
           'watershed_rle', 'combined_label','combined_rle' , 'clean_mask')

isPrint = False
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

def img2rle(lab_img):
    rle_list  =[rle_encoding(lab_img==i) for i in range(1, lab_img.max()+1)]
    
    # remove the empty rle
    for i, rle in enumerate(rle_list):
        if len(rle) == 0:
            rle_list.pop(i)
            
    return rle_list

def separate_masks(lab_img):
    try:
        assert lab_img.dtype == np.int_
    except:
        raise Exception('Not a labled image with "int" dtype.')
    image_list = [lab_img==i for i in range(1, lab_img.max()+1)]
    
    return image_list

def compare_rle(tl_rles, train_row_rles):
    tl_rles = sorted(tl_rles, key = lambda x: int(x[0]))
    
    train_row_rles = sorted(train_row_rles, key = lambda x: int(x[0]))
    
    for tl, tr in zip(tl_rles, train_row_rles):
        print(tl[0], tr[0])
        
    match, mismatch = 0, 0
    
    for img_rle, train_rle in zip(train_row_rles, tl_rles):
        for i_x, i_y in zip(img_rle, train_rle):
            if int(i_x) == int(i_y):
                match += 1
            else:
                mismatch += 1
                
    print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' % (match, mismatch, 100.0*match/(match+mismatch)))

    return (match, mismatch, 100.0*match/(match+mismatch))

def _get_coordinate(img):
    """Return the nonzero coordinates of an image in the shape of (N, 2)"""
    return np.stack( np.nonzero(img)).transpose()

def watershed_label(mask, labled_ct):
    """
    watershed labeling using the predicted centroids
    """
    distance = ndi.distance_transform_edt(mask)
    
    #markers = ndi.label(local_maxi)
    labels = morphology.watershed(-distance, labled_ct, mask=mask)
    
    return labels

def watershed_rle(img, centroids, cut_off = 0.5):
    
    lab_img = watershed_label((img>cut_off).astype(float), centroids)
    
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
        
    return img2rle(lab_img)

def combined_label(merged_masks, eroded_masks, window_size = 8):
    """
    Label the masks first given the eroded_masks, then label the mismatching pixels using the labels
    """
    
    assert isinstance( window_size, (int, tuple))
    
    lab_img = measure.label( eroded_masks)
    
    comb_img = lab_img.copy()
    boundaries = (merged_masks - eroded_masks)>0
    
    bd_pixels = _get_coordinate( boundaries)
    
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
            
            pix_lab = _get_coordinate( mkimg)               
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
    if isPrint:
        print('Runtime for one image: {}'.format(time.time()-clock))
        clock = time.time()

    return comb_img.astype(int)

def combined_rle(lab_img):
    
    #lab_img = combined_label(merged_masks, eroded_masks)
    
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
        
    rle_list = img2rle( lab_img)
    
    return (rle_list, lab_img)


def clean_mask(img, cut_off = 0.5):
    
    mask  = (img>cut_off).astype(int)
    return morphology.opening(morphology.closing(mask, morphology.disk(1)), morphology.disk(3))