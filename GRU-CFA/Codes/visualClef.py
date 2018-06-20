# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:24:59 2018

@author: root
"""

import cPickle as pkl
import numpy

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage
import skimage.transform
import skimage.io
from PIL import Image, ImageEnhance
import scipy.misc



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
from tensorflow.python import debug as tf_debug


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
#                    mat_contents = output[self.layerextract[1]]  
##                    batch_fc[i][j][:] = mat_contents.reshape(self.feature_size_conv)    #'conv5_3_O'
#                    batch_fc[i][j][:] = mat_contents  #'convfc7'
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
                    batch_fc[i][j][:] = mat_contents.reshape(self.feature_size_fc)   #'conv5_3_O'

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
                    batch_fc[i][j][:] = mat_contents.reshape(self.feature_size_fc)   #'conv5_3_O'

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


    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm_list = np.arange(self._num_examples)
            np.random.shuffle(self._perm_list)
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
    def Reset_index_in_epoch_test(self, init_v = 0):   
        self._index_in_epoch_test = init_v         


    def crop_image(self, x, target_height=224, target_width=224):
        image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

        if len(image.shape) == 2:
            image = np.tile(image[:,:,None], 3)
        elif len(image.shape) == 4:
            image = image[:,:,:,0]
    
        height, width, rgb = image.shape
        if width == height:
            resized_image = cv2.resize(image, (target_height,target_width))
      
        elif height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

        return cv2.resize(resized_image, (target_height, target_width))    
    
####### Network Parameters ########       
training_iters = 10000000 # run 10000 epoch
batch_size =  15  
display_step = 280  #280
test_num_total = 15
layername_conv = 'conv5_3' 
layername_fc = 'fc7_final' 
layername = [layername_conv, layername_fc]
numMap = 512
featMap = 14*14
num_classes = 1000  
dropTrain = 0.5 
dropTest = 1
prob_path = '/media/titanz/Data3TB/tensorboard_log/model_20180211/prob_256/'
savefigfile ='/media/titanz/Data3TB/tensorboard_log/model_20180309/attn_visual_imsz384/'
#################################

plantclefdata = DataSet(layername,numMap)

# tf Graph input   
x = tf.placeholder("float", [None, None, 4096])
data = tf.placeholder("float", [None, None, numMap*14*14])
target = tf.placeholder("float", [None, None, num_classes])
dropout = tf.placeholder(tf.float32)
batch_size2 = tf.placeholder(tf.int32)

model = VariableSequenceClassification(x, data, target, dropout, batch_size2)   
sess = tf.InteractiveSession()
#sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)   

saver = tf.train.Saver(max_to_keep = None)
saver.restore(sess, "/media/titanz/Data3TB/tensorboard_log/model_20180309/model_55160")

        
#################################################################################
mat_contents = sio.loadmat('/home/titanz/Documents/SueHan/matlab/PlantClefVGG_net/RNN_plantclef/test_obs_list.mat')
testList = mat_contents['test_obs_list']

mat_contents = sio.loadmat('/media/titanz/Data3TB/tensorboard_log/model_20180309/obs_tp_media_re.mat')
obs_tp_media_re = mat_contents['obs_tp_media_re']

imagesz = 1   # choose image size 1 = 256 , 2 = 384, 3 = 512
if imagesz == 1:  #256
    path_folder = '/media/titanz/Data3TB/PlantclefVGA2/PlantClefImageTest/resize_species/species/'
elif imagesz == 2:  #384
    path_folder = '/media/titanz/data3TB_02/PlantClefolddata_augmented/PlantClefImageTest_SR/resize_species_384/'       
else: #512
    path_folder = '/media/titanz/data3TB_02/PlantClefolddata_augmented/PlantClefImageTest_SR/resize_species_512/'
    
smooth = True

# read python dict back from the testing_256 file
pkl_file_test = open('/home/titanz/tensor_flow/tensorflow-master/tensorflow/examples/RNN/myfile_test_256.pkl', 'rb')
mydict2_test = pickle.load(pkl_file_test)
pkl_file_test.close()

mat_contents = sio.loadmat('/media/titanz/Data3TB/tensorflowlist/species/ClassID_CNNID.mat')
classIDList = mat_contents['ClassID_CNNID']

mediafolderTest_content = '/media/titanz/Data3TB/tensorflowlist/VGG_multipath_res_bn_lastconv/test_obs_media_content_256/'

mat_contents = sio.loadmat('/home/titanz/Documents/SueHan/matlab/PlantClefVGG_net/RNN_plantclef/test_obs_class.mat')
testLabels = mat_contents['test_obs_class']
################################################################################         



for count in xrange(obs_tp_media_re.shape[0]):
    
    print('Obs num {:7.0f}'.format(count))
    
    ObsIDchosen = obs_tp_media_re[count][0].astype(int) - 1#8258#12# chose 0 <= x < 13887

    Obs_name = testList[ObsIDchosen].astype(int) 
    Obs_name = str(Obs_name).split('[')
    Obs_name = Obs_name[1].split(']')
    directory = savefigfile + str(Obs_name[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plantclefdata.Reset_index_in_epoch_test(init_v = ObsIDchosen) 
       
    (test_data_x, test_data_conv), test_label = plantclefdata.PrepareTestingBatch(test_num_total)  


    pred, alpha_forward1, alpha_forward2, alpha_forward3, alpha_backward1, alpha_backward2, alpha_backward3 = sess.run(model.alpha_list_com, feed_dict={x: test_data_x, data: test_data_conv, batch_size2: test_num_total, target: test_label, dropout: dropTest})#, batch_size: batch_size})
    pred_re = pred[0,:,:]
    B = np.argmax(pred_re,axis=1) 
    alpha_forward1 = np.array(alpha_forward1).swapaxes(1,0)  # alpha(max_seq, batch, 196)
    alpha_forward2 = np.array(alpha_forward2).swapaxes(1,0)  # alpha(max_seq, batch, 196)
    alpha_forward3 = np.array(alpha_forward3).swapaxes(1,0)  # alpha(max_seq, batch, 196)
    mat_contents2 = sio.loadmat(mediafolderTest_content + str(mydict2_test[ObsIDchosen][0]) + '.mat',mat_dtype=True)
    used = mat_contents2['v']                 
    alphas1 = np.array(alpha_forward1).swapaxes(1,0)  # alpha(max_seq, batch, 196)
    alphas1 = alphas1[0:used.shape[1],:,:]
    alphas2 = np.array(alpha_forward2).swapaxes(1,0)  # alpha(max_seq, batch, 196)
    alphas2 = alphas2[0:used.shape[1],:,:]
    alphas3 = np.array(alpha_forward3).swapaxes(1,0)  # alpha(max_seq, batch, 196)
    alphas3 = alphas3[0:used.shape[1],:,:]
    pred_re = pred[0,:,:]
    pred_re = pred_re[0:used.shape[1],:]
    B = np.argmax(pred_re,axis=1) 
    B  = B[0:used.shape[1]]
        
    class_picken = testLabels[ObsIDchosen]
    class_picken = class_picken.astype(int)
    index_plot = 1;
    index_plotnc = 1;
    for ii in xrange(alphas1.shape[0]):    # eg: 0,1,2  #list(range(0,alphas.shape[0]*2,2)):
       organlabel = int(used[0,ii])
       if organlabel == 0: 
            organlabelD = 'Branch'
       elif organlabel == 1:  
            organlabelD = 'Entire'    
       elif organlabel == 2:  
            organlabelD = 'Flower' 
       elif organlabel == 3: 
            organlabelD = 'Fruit' 
       elif organlabel == 4:  
            organlabelD = 'Leaf' 
       elif organlabel == 5:  
            organlabelD = 'LeafScan' 
       else: 
            organlabelD = 'Stem'
       
                      
       plt.figure(1)    
       L =  mydict2_test[ObsIDchosen][1][ii].split('/')
       name_str = L[len(L)-1]
       name_str2 = name_str.split('.mat')
       name_str3 = name_str2[0]
       path = path_folder + '{:04d}'.format(class_picken[0]) + '-' + str(classIDList[class_picken[0]][0]) +'/' + name_str3
       img = plantclefdata.crop_image(path)
       plt.imshow(img)
       if smooth:
           alpha_img = skimage.transform.pyramid_expand(alphas1[ii,0,:].reshape(14,14), upscale=16, sigma=20)
       else:
           alpha_img = skimage.transform.resize(alphas1[ii,0,:].reshape(14,14), [img.shape[0], img.shape[1]])
        
       plt.imshow(alpha_img, alpha=0.7)
       plt.set_cmap(cm.Greys_r)
       plt.axis('off')
       plt.savefig(directory + '/' + "course" + str(ii) + ".png")    
       plt.imshow(img)
       plt.axis('off')
       lab2 = organlabelD + '_'+ str(class_picken[0]) + '-' + str(B[ii])
       plt.savefig(directory + '/'  + lab2 + '_' + str(ii) + ".png") 
       plt.figure(2)
       plt.imshow(img)
        
       if smooth:
           alpha_img = skimage.transform.pyramid_expand(alphas2[ii,0,:].reshape(14,14), upscale=16, sigma=20)
       else:
           alpha_img = skimage.transform.resize(alphas2[ii,0,:].reshape(14,14), [img.shape[0], img.shape[1]])
        
       plt.imshow(alpha_img, alpha=0.7)    # show attn
       plt.set_cmap(cm.Greys_r)
       plt.axis('off')
       plt.savefig(directory + '/'  + "fine" + str(ii) + ".png") 
        
       








          


