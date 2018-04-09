#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:25:54 2018

@author: zhong
most are copied from Allard.Hendrikson
Email: zhchzhong@gmail.com
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch as t
import torchvision.utils as tvu
from nuclei.kernel.lossFunc import crossEntropy2d_sum

import datetime
import numpy as np
from msd_pytorch.msd_module import (MSDModule, msd_dilation)
import os
__all__ = ('msdSegModule', )

from scipy import misc
from nuclei.kernel.lossFunc import crossEntropy2d_sum

def _image_save(filename, imarr, image_type ='input'):
    #print('Image size: {}'.format(imarr.shape))
    
    try:
        assert len(imarr.shape) == 3 or len(imarr.shape) ==2
    except:
        print('Can\'t save {}.'.format(image_type), ' Image size: {}'.format(imarr.shape))
        return None
        
    if len(imarr.shape) == 3:
        imarr = np.moveaxis(imarr, 0, 2)
        assert imarr.shape[2] == 2 or imarr.shape[2] == 3
    if imarr.dtype == np.float_:
        imarr -= imarr.min()
        imarr /= imarr.max()
    if len(imarr.shape) == 2:
        misc.imsave(filename, imarr)
    elif imarr.shape[2] == 3:
        misc.imsave(filename, imarr)
    elif imarr.shape[2] == 2:
        misc.imsave(filename, np.concatenate((imarr[:,:,0].squeeze(), imarr[:,:,1].squeeze())))



class msdSegModule(nn.Module):
    
    def __init__(self, c_in, c_out, depth, width, conv3d = False, reflect = True):
        super(msdSegModule, self).__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.depth, self.width = depth, width
        
        self.criterion =  nn.NLLLoss2d()
        

        # This part of the network can be used to renormalize the
        # input data. Its parameters are saved when the network is
        # saved.
        net_fixed = nn.Conv2d( c_in, c_in, 1)
        net_fixed.bias.requires_grad = False
        net_fixed.bias.data.zero_()
        net_fixed.weight.requires_grad = False
        net_fixed.weight.data.fill_(1)
        self.net_fixed = net_fixed
        
        # The rest of the network has parameters that are updated
        # during training.
        self.net_msd = MSDModule(c_in, c_out, depth, width, msd_dilation, reflect=reflect, conv3d=conv3d)
        
        #net_trained = nn.Sequential(self.net_msd, nn.Conv2d( c_out, c_out, 1), nn.LogSoftmax(dim = 1))
        net_trained = nn.Sequential(self.net_msd, nn.Conv2d( c_out, c_out, 1), nn.Softmax(dim = 1))
        
        
        self.net = nn.Sequential(net_fixed,
                                 net_trained)
        
        self.net.cuda()
        
        self.optimizer = optim.Adam(net_trained.parameters())

    def set_normalization(self, dataloader):
        mean = 0
        var = 0
        
        for (data_in, _) in dataloader:
            mean += data_in.mean()
            var += data_in.pow(2).mean()
            
        mean /= len(dataloader)
        var /= len(dataloader)
        std = np.sqrt(var - mean ** 2)

        # The input data should be roughly normally distributed after
        # passing through net_fixed.
        self.net_fixed.bias.data.fill_(- mean)
        self.net_fixed.weight.data.fill_(1 / std)    
    
    def set_input(self, data):
        assert self.c_in == data.shape[1]
        self.input = Variable(data.cuda())

    
    def set_target(self, data):
        # The class labels must be of long data type
        #data = data.long()
        # The class labels must reside on the GPU
        data = data.cuda()
        self.target = Variable(data)   
    
    
    def forward(self, x = None, target = None):
        
        if  x is not None:
            self.set_input(x)
            
        self.output = self.net( self.input)
        
        if target is not None:
            self.set_target(target)
        else:
            self.set_target(t.zeros( self.output.shape))
            
#        print(self.target.data.shape)
        self.loss = crossEntropy2d_sum(self.output, self.target)
#        self.loss = self.criterion(self.output,self.target.squeeze(1))
    def predict(self, x = None):
        
        if  x is not None:
            self.set_input(x)
            
        return self.net( self.input)
    
    def learn(self, x = None, target = None):
        self.forward(x, target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
     
    def print_training(self, epoch = 0, loss = 0.0,  printTime = True):
        print('Epoch: {}'.format(epoch), 'Loss: {}'.format(loss), 'Time: {}'.format( str(datetime.datetime.now())))

    def train(self, dataloader, num_epochs, target_key = 'merged_masks', savefigures = False, num_fig = 10, save_dir = '.' ):
        loss_list = []
        
        if savefigures:
            print('The figures during training are saved in: {}'.format( save_dir))
        for epoch in range(num_epochs):
            training_loss = 0
            for sample in dataloader:
                self.learn(sample['images'], sample[target_key])
                
                training_loss += self.get_loss()
                
            loss_list.append( training_loss/len(dataloader))
            
            self.print_training( epoch,  training_loss/len(dataloader))
            
            if savefigures and epoch %( num_epochs// num_fig) == 0:
                self.save_output(os.path.join(save_dir, 'output_{}.png'.format( epoch)))
                self.save_input(os.path.join(save_dir, 'input_{}.png'.format( epoch)))
                self.save_target(os.path.join(save_dir, 'target_{}.png'.format( epoch)))
                #self.save_diff(os.path.join(save_dir, 'diff_{}.png'.format( epoch)))
                self.save_network(save_dir, 'msdNet_{}.pytorch'.format(epoch))
                
        return loss_list
            
            
    def validate(self, dataloader, target_key = 'merged_masks'):
        validation_loss = 0 
        for sample in dataloader:
            self.learn(sample['images'], sample[target_key])
            
            validation_loss += self.get_loss()
            
        return validation_loss/len(dataloader)
            
            
    def print(self):
        print(self.net)

    def get_loss(self):
        return self.loss.data.sum()       
    
    def get_output(self):
        return self.output
    
    def get_network_path(self, save_dir, fname):
        save_path = os.path.join(save_dir, fname)
        return save_path
    
    def save_network(self, save_dir, fname):
        save_path = self.get_network_path(save_dir, fname)
        os.makedirs(save_dir, exist_ok=True)
        # Clear the L and G buffers before saving:
        self.net_msd.clear_buffers()

        t.save(self.net.state_dict(), save_path)
        return save_path
    
    def load_network(self, save_dir='.', fname= None, save_file=None):
        """Load network parameters from storage.

        :param save_dir: directory to save files in.
        :param name: name of the network.
        :param label: a label (such as current epoch) to add to the filename.
        :param save_file: a file path or stream-like object that overrides the default filename structure.
        :returns:
        :rtype:

        """
        if save_file is None:
            save_file = self.get_network_path(save_dir, fname)
        self.net.load_state_dict(t.load(save_file))
        self.net.cuda()

    def save_output(self, filename):
        #tvu.save_image(self.output.data.squeeze(), filename)
        imarr = self.output.data.squeeze().cpu().numpy()        
        _image_save(filename, imarr, 'output')
        
    def save_input(self, filename):
        #tvu.save_image(self.input.data.squeeze(), filename)
        imarr = self.input.data.squeeze().cpu().numpy()        
        _image_save(filename, imarr, 'input')
         
    def save_target(self, filename):
        #tvu.save_image(self.target.data.squeeze(), filename)
        imarr = self.target.data.squeeze().cpu().numpy()
        _image_save(filename, imarr, 'target')
        
    def save_diff(self, filename):
        #tvu.save_image(t.abs(self.target - self.output).data.squeeze(), filename)
        imarr = t.abs(self.target - self.output).data.squeeze().cpu().numpy()
        _image_save(filename, imarr, 'diff')

    def save_heatmap(self, filename):
        ''' Make a heatmap of the absolute sum of the convolution kernels
        '''

        # heatmap = t.zeros(self.depth, self.depth)

        # conv_ws = [w for k, w in self.net.state_dict().items()
        #            if 'convolution.weight' in k]

        # for i, w in enumerate(conv_ws):
        #     for j in range(w.shape[1]):
        #         heatmap[j, i] = w[:, j, :, :].abs().sum()
        L = self.net.L.clone()
        C = self.net.c_final.weight.data

        for i, c in enumerate(C.squeeze().tolist()):
            L[:, i, :, :].mul_(c)

        tvu.save_image(L[:, 1:, :, :].transpose(0, 1),
                       filename,
                       nrow=10)

    def save_g(self, filename):
        tvu.save_image(self.net.G[:, 1:, :, :].transpose(0, 1),
                       filename,
                       nrow=10)
  