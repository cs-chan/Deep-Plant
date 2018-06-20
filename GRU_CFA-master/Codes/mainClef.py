# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:26:45 2017

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
        batch_fc = np.zeros([batch_size,self._max_seq,self.feature_size_fc])

        
        i = 0
        while i < batch_size:

              media_length = len(mydictG[Newpermbatch[i]][1])
              j = 0
              while j < media_length:     
                    ### for 256 image size for testing
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
                    batch_conv[i][j][:] = mat_contents.reshape(self.feature_size_conv)  #'conv5_3'
                    
                    mat_contents = output[self.layerextract[1]]  
                    batch_fc[i][j][:] = mat_contents  #'convfc7'

                    j = j + 1


                 ## for 384,512 image size for testing  
#                  if indicator == 1:   # training    ###################
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
#                    batch_conv[i][j][:] = mat_contents.reshape(self.feature_size_conv)   #'conv5_3'
#                    
#                    mat_contents = output[self.layerextract[1]]
#                    batch_fc[i][j][:] = mat_contents.reshape(self.feature_size_conv)   #'conv5_3_O'
#
#                    j = j + 1
#                    
#                  else:   # testing  
#                    
#                    pkl_file1 = open(mydictG[Newpermbatch[i]][1][j][0], 'rb')
#                    output = pickle.load(pkl_file1)
#                    pkl_file1.close()
#                    
#                    pkl_file2 = open(mydictG[Newpermbatch[i]][1][j][1], 'rb')
#                    output2 = pickle.load(pkl_file2)
#                    pkl_file2.close()
#                    
#                    output.update(output2)
#                    mat_contents = output[self.layerextract[0]]
#                    batch_conv[i][j][:] = mat_contents.reshape(self.feature_size_conv)    #'conv5_3'
#                    
#                    mat_contents = output[self.layerextract[1]]
#                    batch_fc[i][j][:] = mat_contents.reshape(self.feature_size_conv)   #'conv5_3_O'
#
#                    j = j + 1 
                    
#########################################################
                    
                    
              # random shuffle organ sequeces                    
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
        
        



####### Network Parameters ########       
training_iters = 10000000 
batch_size = 15   
display_step = 280
test_num_total = 15
layername_conv = 'conv5_3' 
layername_fc = 'fc7_final'
layername = [layername_conv, layername_fc]
numMap = 512#20
num_classes = 1000  
dropTrain = 0.5 
dropTest = 1


plantclefdata = DataSet(layername,numMap)

# tf Graph input   
x = tf.placeholder("float", [None, None, 4096])
data = tf.placeholder("float", [None, None, numMap*14*14])
target = tf.placeholder("float", [None, None, num_classes])
dropout = tf.placeholder(tf.float32)
batch_size2 = tf.placeholder(tf.int32)


#saved Model directory
save_dir = '/media/titanz/Data3TB/tensorboard_log/model_20180418/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
   
model = VariableSequenceClassification(x, data, target, dropout, batch_size2)

#combine all summaries for tensorboard
summary_op = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep = None)
sess = tf.Session()

sess.run(tf.global_variables_initializer())   

# Resume training
#saver.restore(sess, "/media/titanz/Data3TB/tensorboard_log/model_20180418/model_13160")

# declare tensorboard folder
log_path = '/media/titanz/Data3TB/tensorboard_log/20180418'
train_writer = tf.summary.FileWriter(log_path + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(log_path + '/test')


step = 1

while step * batch_size < training_iters:  # step = 280 is equal to one epoch

        (batch_x_fc, batch_x_conv), batch_y = plantclefdata.next_batch(batch_size)
        loss = sess.run(model.cost, feed_dict={x: batch_x_fc, data: batch_x_conv, batch_size2: batch_size, target: batch_y, dropout: dropTrain})
        train_acc = sess.run(model.error, feed_dict={x: batch_x_fc, data: batch_x_conv, batch_size2: batch_size, target: batch_y, dropout: dropTrain})
        _,summary = sess.run([model.optimize, summary_op], feed_dict={x: batch_x_fc, data: batch_x_conv, batch_size2: batch_size, target: batch_y, dropout: dropTrain})
        # write log
        train_writer.add_summary(summary, step * batch_size)   
        
        if step % display_step == 0:

           strftime("%Y-%m-%d %H:%M:%S", gmtime())

           logfile.logging("Epoch" + str(step) + ", Minibatch Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy = " + \
                            "{:.5f}".format(train_acc) + ", lengthData= " + "{:.1f}".format(plantclefdata.max_seq()))

           
        if step % display_step == 0:

           saveid  = 'model_%s' %step
           save_path = save_dir + saveid
           saver.save(sess, save_path)
           (test_data_x, test_data_conv), test_label = plantclefdata.PrepareTestingBatch(test_num_total) # step/epoch = 694.35 = All testing data tested
           test_loss = sess.run(model.cost, feed_dict={x: test_data_x, data: test_data_conv, batch_size2: test_num_total, target: test_label, dropout: dropTest})
           test_acc,summary = sess.run([model.error, summary_op], feed_dict={x: test_data_x, data: test_data_conv, batch_size2: test_num_total, target: test_label, dropout: dropTest})
           logfile.logging('testing accuracy {:3.5f}%'.format(test_acc) + ", testbatch Loss= " + \
                            "{:.6f}".format(test_loss))
           test_writer.add_summary(summary, step * batch_size)
        step += 1

print("Optimization Finished!")

