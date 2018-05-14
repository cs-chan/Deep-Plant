# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:21:38 2018

@author: root
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:42:56 2017

@author: root
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:45:01 2016

@author: root
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:49:40 2016

@author: root
"""
# this code create a lookup table for the observation List
import numpy as np
#from tensorflow.python.ops import rnn, rnn_cell
import scipy.io as sio
import scipy.io as spio
from scipy.io import loadmat
import pickle
import re


class ConstructLookupTable:

    def __init__(self):
        self.ObsID = None
        self.mediaID = None  
        self.originalFolder = '/media/titanz/Data3TB/conv_f7_trainAL/'     #---> allsize for fc7
        self.originalFolderTest = '/media/titanz/Data3TB/conv_fc7_test_256/' #---> size 256 for fc7
        self.mediafolderTrain = '/media/titanz/Data3TB/tensorflowlist/VGG_multipath_res_bn_lastconv/train_obs_media/'
        self.mediafolderTest = '/media/titanz/Data3TB/tensorflowlist/VGG_multipath_res_bn_lastconv/test_obs_media_256/'
     


    def crete_object_Train(self, K):
        self.ObsID =K
        self.mediaID = []
        mat_contents2 = sio.loadmat(self.mediafolderTrain + self.ObsID + '.mat',mat_dtype=True)
        used = mat_contents2['B']
        
        LL = np.arange(used.shape[1])
        np.random.shuffle(LL)
           
        n = 0
        while n < used.shape[1]:
           readString = used[0,LL[n]] 
           readString = str(readString[0])
           k2 = readString.split('/')
           media_ID = k2[9].split('.mat')
           media_ID = media_ID[0].split('_')
           N = [0, 1, 2]
           np.random.shuffle(N)
           if N[0] == 0:
              pkl_file = self.originalFolder + '512_'+ media_ID[1] + '.pkl'
           elif N[0] == 1:
              pkl_file = self.originalFolder + '384_'+ media_ID[1] + '.pkl'
           else:
              pkl_file = self.originalFolder + '256_'+ media_ID[1] + '.pkl'
   
           self.mediaID.append(pkl_file) 
           n = n + 1

        return [self.ObsID,self.mediaID]
  
    def crete_object_Test(self, K):
        self.ObsID =K
        self.mediaID = []
        mat_contents2 = sio.loadmat(self.mediafolderTest + self.ObsID + '.mat',mat_dtype=True)
        used = mat_contents2['B']
        
        n = 0
        while n < used.shape[1]:
           readString = used[0,n] 
           readString = str(readString[0])
           k2 = readString.split('/')
           media_ID = k2[9].split('.mat')
           pkl_file = self.originalFolderTest + media_ID[0] + '.pkl'
           self.mediaID.append(pkl_file)
           n = n + 1
 
        return [self.ObsID,self.mediaID]      
           
           

        
              
    def main(self, ObsList, train_testID):
         step = 0
         LookupTable = []
 
         if train_testID ==1:
            while step < ObsList.shape[0]:  ## change to 27907 for training  13887 for testing
                K = str(int(ObsList[step,0]))
                LookupTable.append(self.crete_object_Train(K))    
                step += 1
         else:
            while step < ObsList.shape[0]:  ## change to 27907 for training  13887 for testing
                K = str(int(ObsList[step,0]))
                LookupTable.append(self.crete_object_Test(K))    
                step += 1
         return LookupTable
              

    
    
    
    
    
    
    
    
    
    
    
    
    
