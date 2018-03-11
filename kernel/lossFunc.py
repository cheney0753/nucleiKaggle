# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:56:23 2018

@author: zhong
"""

import unittest
import numpy as np
from torch.autograd import Variable
import torch

def diceLoss(input_, target_):
    """
    Dice loss function
    ------------
    Parameters : 
        input_ : pytorch.Variable \n
        output_ : pytorch.Variable
    Return : 
        loss_ : pytorch.Variable
    """
    smooth = 1
    iflat = input_.view(-1)
    tflat = target_.view(-1)
    intersection = (iflat*tflat).sum()
    loss_ = -(2.* intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
      
    return loss_

def _IoU_numpy(A, B):
    """
    Intersection over union (IoU) 
    ----------
    Par : 
        A: binary segmentation np.array(dtype = bool) \n
        B: ground truth segmentation
    Return:
        iou: float
    """
    if A.dtype != bool or B.dtype != bool:
        raise Exception('The input type must be bool numpy array!')
        
    if A.shape != B.shape:
        raise Exception('Sizes must be the same!')

    return (A & B) / (A | B)

def _intersect(segA, segB):
    """
    the intersection area of two varialbles
    ----------------
    """
    
    return (segA.view(-1) * segB.view(-1)).sum().float()

def IoU_torch(segA, segB):
    """
    IoU of two tensor variables:
    """
    
    return _intersect(segA, segB) / ( segA.view(-1).sum().float()+ segB.view(-1).sum().float() -
                     _intersect(segA, segB))    

def IoU_mean(varA, segB):
    """
    mean IoU (Intersection over union) for a list of thresholds
    ---------
    Par : 
        varA : torch.Variable \n
        varB : torch.Variable
    Return : 
        meanIoU : torch.Variable 
    """    
    thresholds = np.arange(0.5, 1.0, 0.05);
    
    IoU_mean = IoU_torch(varA > thresholds[0], segB) 
    for th in thresholds[1:]:
        IoU_mean = (IoU_mean + IoU_torch( varA > th, segB))/2
        
    return IoU_mean            
    
    
class TestlossFunc(unittest.TestCase):
    
    rand1 = np.random.rand( *[ 3, ]*4 )
    rand2 = np.random.rand( *[ 3, ]*4 )
    var1 = Variable(torch.from_numpy(rand1))
    var2 = Variable(torch.from_numpy(rand2))
    
    def test_diceLoss(self):
        
        
        flt1 = self.__class__.rand1.flatten()
        flt2 = self.__class__.rand2.flatten()
        ints = (flt1 * flt2).sum()
        loss = -(2*ints + 1) / (flt1.sum() + flt2.sum() + 1)
        
        varloss = diceLoss(self.__class__.var1, self.__class__.var2);
        
        #check if the result is good (there is some precision error)
        self.assertTrue(abs(varloss.data[0] - loss) < 0.01)
        
    def test_IoU(self):
        _IoU_numpy( (np.random.rand(3,3)>0.5), (np.random.rand(3,3)>0.5))
    
    def test_intersect(self):
        print( _intersect(self.__class__.var1 > 0.5, self.__class__.var2 >0.5))
        
    def test_IoU_torch(self):
        print('IoU_torch' , IoU_torch(self.__class__.var1 > 0.5, self.__class__.var2 >0.5) )
        
    def test_meanIoU(self):
        print( IoU_mean(self.__class__.var1, self.__class__.var2>0.5))

if __name__ == '__main__':
    unittest.main()
    