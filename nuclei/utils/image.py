# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
IMG_CHANNELS = 3
import numpy as np

__all__ = ('isChromatic', )


def isChromatic( img ):
    """ 
    return if an image is chromatic or not
    ---------
    Parameters : 
        img: np.array
    Return : 
        bool
    """
    assert img.shape[2] == IMG_CHANNELS;
    
    number_sample = 5
    sizeImg = img.shape[0:IMG_CHANNELS]
    rand_pix = [(a, b) for a, b in zip( np.random.choice(sizeImg[0], number_sample),
                np.random.choice(sizeImg[1], number_sample) )]
    for pix in rand_pix:
        pix_ch = img[pix[0], pix[1], :]
        if np.array_equal(pix_ch, np.roll(pix_ch, 1)):
            return False
        
    return True

