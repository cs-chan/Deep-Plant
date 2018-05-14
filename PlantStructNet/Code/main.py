# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:51:50 2017

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
from Var_seq2seq_classification_bidirectRNN import VariableSequenceClassification
from temp_createStruct2_2 import ConstructLookupTable
from time import gmtime, strftime
from logging_util import makelog

logfile=makelog()

class DataSet(object):

   
    def __init__(self, layername, numMap):
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

        
        
        if self.layerextract != 'fc7_final':
           self.feature_size = self.numMap*14*14    
        else:
           self.feature_size = 4096 
           

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
        
        
                   


        
        batch = np.zeros([batch_size,self._max_seq,self.feature_size])
        
        i = 0
        while i < batch_size:
              media_length = len(mydictG[Newpermbatch[i]][1])
              j = 0
              while j < media_length:     
                    pkl_file = open(mydictG[Newpermbatch[i]][1][j], 'rb')
                    output = pickle.load(pkl_file)
                    pkl_file.close()
                    mat_contents = output[self.layerextract]
                    
                    if self.layerextract == 'fc7_final':
                       batch[i][j][:] = mat_contents    
                    else:
                       batch[i][j][:] = mat_contents.reshape(self.feature_size)

                    j = j + 1
              # random shuffle organ sequeces
              if indicator == 1:
                 temp_arr = batch[i][0:media_length]  #x = np.random.randn(2, 4, 4).astype('f') B = x[0][0:3]
                 np.random.shuffle(temp_arr)
                 batch[i][0:media_length] = temp_arr

              i = i + 1
       
        return batch
        
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
training_iters = 10000000 # run 10000 epoch
batch_size =  30  
display_step = 280  #280
test_num_total = 1000
layername = 'fc7_final'  #conv_7' # 'conv_7'  #'fc6_final' #'conv_7'
numMap = 512#20
num_classes = 1000  ### number classes of output
dropTrain = 0.5 #0.5
dropTest = 1

#################################

if layername != 'fc7_final':
   row_size = numMap*14*14    
else:
   row_size = 4096  


plantclefdata = DataSet(layername,numMap)

# tf Graph input   

data = tf.placeholder("float", [None, None, row_size])
target = tf.placeholder("float", [None, None, num_classes])
dropout = tf.placeholder(tf.float32)


#saved Model directory
save_dir = '/media/titanz/Data3TB/tensorboard_log/model_20180227/'
#save_dir = '/media/titanz/data3TB_02/TEMP/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
   
model = VariableSequenceClassification(data, target, dropout)

#combine all summaries for tensorboard
summary_op = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep = None)
sess = tf.Session()

sess.run(tf.global_variables_initializer())   
# Resume training

#saver.restore(sess, "/media/titanz/Data3TB/tensorboard_log/model_20170717/model_11760")
#tensorboard --logdir='/home/r720/Spyder/TF_tutor/TB/fcn32_1/6.5'

# declare tensorboard folder
log_path = '/media/titanz/Data3TB/tensorboard_log/20180227'

train_writer = tf.summary.FileWriter(log_path + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(log_path + '/test')


step = 1



while step * batch_size < training_iters:  

        batch_x, batch_y = plantclefdata.next_batch(batch_size)

        
        # change target_list from 2 dim to 3 dim -> (batch_size, max_seq, num_classes)
        
        loss = sess.run(model.cost, feed_dict={data: batch_x, target: batch_y, dropout: dropTrain})
        train_acc = sess.run(model.error, feed_dict={data: batch_x, target: batch_y, dropout: dropTrain})

        _,summary = sess.run([model.optimize, summary_op], feed_dict={data: batch_x, target: batch_y, dropout: dropTrain})

        train_writer.add_summary(summary, step * batch_size)   
        
        if step % display_step == 0:

           strftime("%Y-%m-%d %H:%M:%S", gmtime())
           logfile.logging("Epoch" + str(step) + ", Minibatch Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy = " + \
                            "{:.5f}".format(train_acc) + ", lengthData= " + "{:.1f}".format(plantclefdata.max_seq()))

           
        if step % display_step == 0:

           #save the model
           saveid  = 'model_%s' %step
           save_path = save_dir + saveid
           saver.save(sess, save_path)
           #calcutae test accuracy
           test_data, test_label = plantclefdata.PrepareTestingBatch(test_num_total) # step/epoch = 694.35 = All testing data tested
           test_loss = sess.run(model.cost, feed_dict={data: test_data, target: test_label, dropout: dropTest})
           test_acc,summary = sess.run([model.error, summary_op], feed_dict={data: test_data, target: test_label, dropout: dropTest})
           #print('testing accuracy {:3.5f}%'.format(test_acc))
           logfile.logging('testing accuracy {:3.5f}%'.format(test_acc) + ", testbatch Loss= " + \
                            "{:.6f}".format(test_loss))
           test_writer.add_summary(summary, step * batch_size)
        step += 1
         
print("Optimization Finished!")

