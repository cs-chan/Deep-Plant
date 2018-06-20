# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 01:10:00 2017

@author: root
"""# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 14:03:06 2014

@author: holmes
"""

import numpy as np
import caffe
import sys
import scipy.io as io
import glob
import os
import shutil
import cv2
import pickle
from PIL import Image


def get_name(image_path):
    l = len(image_path)
    while (image_path[l-1]!='/'):
        l = l - 1
    return image_path[l:len(image_path)]
    

image_path = "/media/titanz/data3TB_02/PlantClefolddata_augmented/PlantClefImageTrain_SR/"  # for train>>>>>>
net_definition = '/home/titanz/caffe_v4/caffe/examples/VGG_multipath_bn_yl_lastconv/PlantClef_VGGmultipath_deploy.prototxt'
caffemodel = '/media/titanz/data3TB_02/PlantClefolddata_augmented/caffe_v4/VGG_multipath_res_bn_lastconv/PlantClef_vgg_species_organ_iter_180854.caffemodel'
mean_file = '/media/titanz/data3TB_02/PlantClefolddata_augmented/species_mean_aug.npy'
save_path = '/media/titanz/Data3TB/conv_f7_trainAL'
layer_name = ['conv5_3','conv_6','conv_7','fc6_final','fc7_final']
read_testing_txt = '/media/titanz/data3TB_02/PlantClefolddata_augmented/species_train.txt';


batch = 50
input_shape = (batch,256,256,3)

if not os.path.exists(save_path):
    os.makedirs(save_path)


caffe.set_mode_gpu()

caffenet = caffe.Classifier(net_definition,
                            caffemodel,
                            mean = np.load(mean_file).mean(1).mean(1),
                            channel_swap=(2,1,0),
                            raw_scale=255,
                            image_dims=(256,256))







class_idx = []
with open(read_testing_txt,'r+') as fid:
    for lines in fid:
         class_idx.append(lines.rstrip('\r\n')) 
         
count = 0;


class_buffer = []
for class_name in class_idx:
    a,b = class_name.split(' ')
    class_buffer.append(image_path + a)
    count = count + 1;
    input_image = np.zeros(input_shape,dtype=np.float32)
    c = 0
    if (len(class_buffer) == batch) or (count == len(class_idx)):
        for imagepath in class_buffer:
            input_image[c] = caffe.io.load_image(imagepath)
            c+=1
        
        prediction = caffenet.predict(input_image,oversample=False)
        

        


        for saveidx in range(c):
            sub_feat = {}
            for y in range(0,len(layer_name)):
                sub_feat[layer_name[y]] = caffenet.blobs[layer_name[y]].data[saveidx]
            
            k1,k2 = class_buffer[saveidx].split('/')[-2:]
            output_file = save_path + '/' + k2 + '.pkl'
            with open(output_file, 'wb') as output:   
                pickle.dump(sub_feat, output)  
            line_out = "%d : %s processed" % (count-c+saveidx,class_buffer[saveidx])
            print line_out
        class_buffer = []
        


