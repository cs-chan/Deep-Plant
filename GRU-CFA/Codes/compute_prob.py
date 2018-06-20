# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:35:57 2017

@author: root
"""


import tensorflow as tf
import numpy as np
import os
import struct
import scipy.io as sio
from array import array as pyarray
from numpy import array, int8, uint8, zeros
import collections
import pickle

import functools
import sets
from tensorflow.python.ops import rnn, array_ops
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell



from attn_7_1_ex import VariableSequenceClassification
from temp_createStruct5 import ConstructLookupTable


from time import gmtime, strftime

from logging_util import makelog

logfile=makelog()

class DataSet(object):


    def __init__(self, layername, numMap):
        """Construct a DataSet."""
        
        mat_contents = sio.loadmat('/home/titanz/Documents/SueHan/matlab/PlantClefVGG_net/RNN_plantclef/train_obs_list.mat')
        self._trainList = mat_contents['train_obs_list']
        
        mat_contents = sio.loadmat('/home/titanz/Documents/SueHan/matlab/PlantClefVGG_net/RNN_plantclef/train_obs_class.mat')
        self._trainLabels = mat_contents['train_obs_class']
        
        mat_contents = sio.loadmat('/home/titanz/Documents/SueHan/matlab/PlantClefVGG_net/RNN_plantclef/test_obs_list.mat')
        self._testList = mat_contents['test_obs_list']
        
        mat_contents = sio.loadmat('/home/titanz/Documents/SueHan/matlab/PlantClefVGG_net/RNN_plantclef/test_obs_class.mat')
        self._testLabels = mat_contents['test_obs_class']
        
        self.layerextract = layername
        self.numMap = numMap
        self._num_examples = self._trainLabels.shape[0]
        self._perm_list = np.arange(self._num_examples)
        np.random.shuffle(self._perm_list)
        
        self._trainLabelsPerm = self._trainLabels[self._perm_list]
        
        self._num_testexamples = self._testLabels.shape[0]
        self._perm_list_test = np.arange(self._num_testexamples)      
        
        self._batch_seq = 0      
        self._epochs_completed = 0
        self._index_in_epoch = 0  
        self._index_in_epoch_test = 0 
        self._max_seq = 0
        

        self.Batch_Up_model = ConstructLookupTable()    
        self.mydict2_test256 = self.Batch_Up_model.main(self._testList,2) # for train_testID ! = 1
        self.feature_size_conv = self.numMap*14*14
        self.feature_size_fc = 4096
        

           

    def trainList(self):
        return self._trainList
        
    def trainLabels(self):
        return self._trainLabels
        
    def trainLabelsPerm(self):
        return self._trainLabelsPerm
        
    def testList(self):
        return self._testList
        
    def testLabels(self):
        return self._testLabels
        
    def num_examples(self):
        return self._num_examples
        
    def num_testexamples(self):
        return self._num_testexamples
        
    def epochs_completed(self):
        return self._epochs_completed
        
    def index_in_epoch(self):
        return self._index_in_epoch
    
    def max_seq(self):
        return self._max_seq    
        
    def batch_seq(self):
        return self._batch_seq 
        

    def PrepareTrainingBatch(self,Newpermbatch, batch_size, indicator):
        if indicator == 1:
           mydictG = self.Batch_Up_model.main(self._trainList,1) # for train_testID == 1
        else:
           mydictG = self.mydict2_test256
           
        i = 0
        temp = np.zeros(batch_size)
        while i < batch_size:

              temp[i] = len(mydictG[Newpermbatch[i]][1])
              i = i + 1     
        self._max_seq = int(np.amax(temp))
        self._batch_seq = temp
        
            
                   


        
        batch_conv = np.zeros([batch_size,self._max_seq,self.feature_size_conv])
        batch_fc = np.zeros([batch_size,self._max_seq,self.feature_size_fc])   #'fc6 or fc7'
        
        i = 0
        while i < batch_size:
             #media_length = len(self.mydict2[Newpermbatch[i]][1])
              media_length = len(mydictG[Newpermbatch[i]][1])
              j = 0
              while j < media_length:  

 ### for 256 image size for testing
#                    pkl_file1 = open(mydictG[Newpermbatch[i]][1][j][0], 'rb')
#                    output = pickle.load(pkl_file1)
#                    pkl_file1.close()
#                    
#                    pkl_file2 = open(mydictG[Newpermbatch[i]][1][j][1], 'rb')
#                    output2 = pickle.load(pkl_file2)
#                    pkl_file2.close()
#                    
#                    pkl_file3 = open(mydictG[Newpermbatch[i]][1][j][2], 'rb')
#                    output3 = pickle.load(pkl_file3)
#                    pkl_file3.close()
#                    
#                    output.update(output2)
#                    output.update(output3)
#                    mat_contents = output[self.layerextract[0]]
#                    batch_conv[i][j][:] = mat_contents.reshape(self.feature_size_conv)  #'conv5_3'
#                    
#                    
#                    mat_contents = output[self.layerextract[1]]  
#                    batch_fc[i][j][:] = mat_contents  #'fc6 or fc7'
#                    
#                    
#
#                    j = j + 1

                    
######################################################################

                    
                  ## for 384,512 image size for testing  
                  if indicator == 1:   # training    ###################
                    pkl_file1 = open(mydictG[Newpermbatch[i]][1][j][0], 'rb')
                    output = pickle.load(pkl_file1)
                    pkl_file1.close()
                    
                    pkl_file2 = open(mydictG[Newpermbatch[i]][1][j][1], 'rb')
                    output2 = pickle.load(pkl_file2)
                    pkl_file2.close()
                    
                    pkl_file3 = open(mydictG[Newpermbatch[i]][1][j][2], 'rb')
                    output3 = pickle.load(pkl_file3)
                    pkl_file3.close()
                    
                    output.update(output2)
                    output.update(output3)
                    mat_contents = output[self.layerextract[0]]
                    batch_conv[i][j][:] = mat_contents.reshape(self.feature_size_conv)   #'conv5_3'

                    mat_contents = output[self.layerextract[1]]  
                    batch_fc[i][j][:] = mat_contents  #'fc6 or fc7'
                    j = j + 1
                    
                  else:   # testing  
                    
                    pkl_file1 = open(mydictG[Newpermbatch[i]][1][j][0], 'rb')
                    output = pickle.load(pkl_file1)
                    pkl_file1.close()
                    
                    pkl_file2 = open(mydictG[Newpermbatch[i]][1][j][1], 'rb')
                    output2 = pickle.load(pkl_file2)
                    pkl_file2.close()
                    
                    output.update(output2)
                    mat_contents = output[self.layerextract[0]]
                    batch_conv[i][j][:] = mat_contents.reshape(self.feature_size_conv)    #'conv5_3'
                    

                    mat_contents = output[self.layerextract[1]]  
                    batch_fc[i][j][:] = mat_contents  #'fc6 or fc7'
                    j = j + 1 
                    
#########################################################
                    
                    
              if indicator == 1:       
                 J = np.arange(media_length)
                 np.random.shuffle(J)
                 
                 temp_arr = batch_conv[i,:media_length,:]
                 temp_arr = temp_arr[J,:]
                 batch_conv[i,:media_length,:] = temp_arr
                 
                 temp_arr = batch_fc[i,:media_length,:]
                 temp_arr = temp_arr[J,:]
                 batch_fc[i,:media_length,:] = temp_arr
                 
              i = i + 1


        return batch_fc, batch_conv
        
    def dense_to_one_hot(self,labels_dense, num_classes=1000):
        labels_dense = labels_dense.astype(int)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        labels_one_hot = labels_one_hot.astype(np.float32)
        
        temp = zeros((labels_one_hot.shape[0],self._max_seq,num_classes))
        i=0
        while i < labels_one_hot.shape[0]:
              temp[i][0:int(self._batch_seq[i])] = labels_one_hot[i]
              i=i+1

        return temp

   # def next_batch(self, batch_size):
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm_list = np.arange(self._num_examples)
            np.random.shuffle(self._perm_list)
            #self._trainList = self._trainList[perm]
            self._trainLabelsPerm = self._trainLabels[self._perm_list]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self.PrepareTrainingBatch(self._perm_list[start:end], batch_size, 1), self.dense_to_one_hot(self._trainLabelsPerm[start:end])
        
    def PrepareTestingBatch(self,test_total):
        start = self._index_in_epoch_test
        self._index_in_epoch_test += test_total
        if self._index_in_epoch_test > self._num_testexamples:
            start = 0
            self._index_in_epoch_test = test_total
            assert test_total <= self._num_testexamples
        end = self._index_in_epoch_test
             

        return self.PrepareTrainingBatch(self._perm_list_test[start:end], test_total, 0), self.dense_to_one_hot(self._testLabels[start:end])
        
     ## Testing   
    def Reset_index_in_epoch_test(self):   
        self._index_in_epoch_test = 0            

####### Network Parameters ########       
training_iters = 10000000 # run 10000 epoch
batch_size =  20    #30
display_step = 280  
#display_step = 1 
test_num_total = 30
layername_conv = 'conv5_3'
layername_fc = 'fc7_final' 
layername = [layername_conv, layername_fc]
numMap = 512
num_classes = 1000 
dropTrain = 0.5 
dropTest = 1
prob_path = '/media/titanz/Data3TB/tensorboard_log/model_20180127/prob_512/'
#################################

if layername == 'fc6_final' or layername == 'fc7_final':
   row_size = 4096
else:
   row_size = numMap*14*14  
   
plantclefdata = DataSet(layername,numMap)

# tf Graph input   
x = tf.placeholder("float", [None, None, 4096])
data = tf.placeholder("float", [None, None, numMap*14*14])
target = tf.placeholder("float", [None, None, num_classes])
dropout = tf.placeholder(tf.float32)
batch_size2 = tf.placeholder(tf.int32)


model = VariableSequenceClassification(x, data, target, dropout, batch_size2)

sess = tf.Session()

saver = tf.train.Saver(max_to_keep = None)
saver.restore(sess, "/media/titanz/Data3TB/tensorboard_log/model_20180127/model_19320")
pkl_file_test = open('/home/titanz/tensor_flow/tensorflow-master/tensorflow/examples/RNN/myfile_test_256.pkl', 'rb')

mydict2_test = pickle.load(pkl_file_test)
pkl_file_test.close()
        
          
plantclefdata.Reset_index_in_epoch_test() 
ACC = 0         
num = 0
j = 0
while num < 13887:
        print('Obs num {:7.0f}'.format(num))

        (test_data_x, test_data_conv), test_label = plantclefdata.PrepareTestingBatch(1)
        prob = sess.run(model.prediction, feed_dict={x: test_data_x, data: test_data_conv, batch_size2: 1, target: test_label, dropout: dropTest})#, batch_size: batch_size})
 
        while j < test_label.shape[1]:
               L =  mydict2_test[num][1][j].split('/')
               name_str = L[len(L)-1]
               KOK = name_str.split('.mat')
               sio.savemat(prob_path + name_str , mdict = {'prob': prob[0][j][:]})
               j=j+1

        num = num + 1
        j = 0
