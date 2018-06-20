# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:26:38 2017

@author: root
"""

# this code create a lookup table for the observation List
import numpy as np
import scipy.io as sio
import scipy.io as spio
from scipy.io import loadmat
import pickle
import re


class ConstructLookupTable:

    def __init__(self):
        self.ObsID = None
        self.mediaID = None
         ############ training images ##############
        self.originalFolder1 = '/media/titanz/data3TB_02/PlantClefolddata_augmented/caffe_v4/VGG_multipath_res_bn_lastconv/log_file_256/conv_5_6_7_f6_train_ALL/'   #---> for all images sizes (256, 384, 512) for layers conv5_3, 6, 7, fc6
        self.originalFolder2 = '/media/titanz/Data3TB/conv_f7_trainAL/'     #---> for all images sizes (256, 384, 512) for fc7
        self.originalFolder3 = '/media/titanz/Data3TB/conv_5_3_O_trainAL/'  #---> for all images sizes (256, 384, 512) for conv5_3_O

        ############ testing images ################ 
        self.originalFolderTest1 = '/media/titanz/data3TB_02/PlantClefolddata_augmented/caffe_v4/VGG_multipath_res_bn_lastconv/log_file_256/conv_5_6_7_f6_test_256/'  #---> image size 256 for conv5, 6, 7, fc6
        self.originalFolderTest2 = '/media/titanz/Data3TB/conv_fc7_test_256/' #---> image size 256 for fc7
        self.originalFolderTest3 = '/media/titanz/Data3TB/conv_5_3_O_test_256/'  #---> image size 256 for conv5_3_O
        
#        self.originalFolderTest = '/media/titanz/data3TB_02/PlantClefolddata_augmented/caffe_v4/VGG_multipath_res_bn_lastconv/log_file_384/conv_5_6_7_f6_test_384/'   #---> image size 384 for layers conv5, 6, 7, fc6 , fc7      
#        self.originalFolderTest3 = '/media/titanz/Data3TB/conv_5_3_O_test_384/' #---> image size 384 for layer conv5_3_O
#
#        self.originalFolderTest = '/media/titanz/data3TB_02/PlantClefolddata_augmented/caffe_v4/VGG_multipath_res_bn_lastconv/log_file_512/conv_5_6_7_f6_test_512/'   #---> image size 512 for layers conv5, 6, 7, fc6 , fc7          
#        self.originalFolderTest3 = '/media/titanz/Data3TB/conv_5_3_O_test_512/' #---> image size 512 for layer conv5_3_O


        self.mediafolderTrain = '/media/titanz/Data3TB/tensorflowlist/VGG_multipath_res_bn_lastconv/train_obs_media/'
        self.mediafolderTest = '/media/titanz/Data3TB/tensorflowlist/VGG_multipath_res_bn_lastconv/test_obs_media_256/'
     


    def crete_object_Train(self, K):
        self.ObsID =K
        self.mediaID = []
        mat_contents2 = sio.loadmat(self.mediafolderTrain + self.ObsID + '.mat',mat_dtype=True)
        used = mat_contents2['B']      
        
        LL = np.arange(used.shape[1])
        np.random.shuffle(LL)
        
        if used.shape[1]+1 > 40:
           new_used_shape = np.random.randint(1, 40)
        else:
           new_used_shape = np.random.randint(1, high = used.shape[1]+1)
        n = 0

        while n < new_used_shape:
           readString = used[0,LL[n]] 
           readString = str(readString[0])
           k2 = readString.split('/')
           media_ID = k2[9].split('.mat')
           media_ID = media_ID[0].split('_')
           N = [0, 1, 2]
           np.random.shuffle(N)
           
           if N[0] == 0:
              pkl_file1 = self.originalFolder1 + '512_'+ media_ID[1] + '.pkl'
              pkl_file2 = self.originalFolder2 + '512_'+ media_ID[1] + '.pkl'
              pkl_file3 = self.originalFolder3 + '512_'+ media_ID[1] + '.pkl'

           elif N[0] == 1:
              pkl_file1 = self.originalFolder1 + '384_'+ media_ID[1] + '.pkl'
              pkl_file2 = self.originalFolder2 + '384_'+ media_ID[1] + '.pkl'
              pkl_file3 = self.originalFolder3 + '384_'+ media_ID[1] + '.pkl'
              
           else:
              pkl_file1 = self.originalFolder1 + '256_'+ media_ID[1] + '.pkl'
              pkl_file2 = self.originalFolder2 + '256_'+ media_ID[1] + '.pkl'
              pkl_file3 = self.originalFolder3 + '256_'+ media_ID[1] + '.pkl'

           
           pkl_file = [pkl_file1,pkl_file2,pkl_file3]
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
           # for 256 image size
           pkl_file1 = self.originalFolderTest1 + media_ID[0] + '.pkl'
           pkl_file2 = self.originalFolderTest2 + media_ID[0] + '.pkl'
           pkl_file3 = self.originalFolderTest3 + media_ID[0] + '.pkl'
           pkl_file = [pkl_file1,pkl_file2,pkl_file3]
           
            # for 384 and 512 image size
#           pkl_file1 = self.originalFolderTest + media_ID[0] + '.pkl'
#           pkl_file3 = self.originalFolderTest3 + media_ID[0] + '.pkl'
#           pkl_file = [pkl_file1,pkl_file3]                 
           
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
              
           
           

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

