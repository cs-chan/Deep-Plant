# -*- coding: utf-8 -*-
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
from PIL import Image




def get_name(image_path):
    l = len(image_path)
    while (image_path[l-1]!='/'):
        l = l - 1
    return image_path[l:len(image_path)]



 
image_path = "/media/titanz/data3TB_02/PlantCLEF2015Test/img_256/"
image_format = 'jpg'
net_definition = './model_species/PlantClef_VGGmultipath_deploy.prototxt'
caffemodel = './model_species/PlantClef_vgg_species_organ_iter_524332'
mean_file = './species_mean_aug'
save_path = '/media/titanz/data3TB_02/PlantCLEF2015Test/img_256/testprob_256'
layer_name = ['prob']



    

if not os.path.exists(save_path):
#    shutil.rmtree(save_path)
    os.makedirs(save_path)


caffe.set_mode_gpu()

caffenet = caffe.Classifier(net_definition,
                            caffemodel,
                            mean = np.load(mean_file).mean(1).mean(1),
                            channel_swap=(2,1,0),
                            raw_scale=255,
                            image_dims=(256,256))



dir_source = image_path + '/*.' + image_format
files=glob.glob(dir_source)
length = len(files)

for k in range(0,length):
    image_name = files[k]   
    input_image = caffe.io.load_image(image_name)

    prediction = caffenet.predict([input_image])
    
    blob_item = caffenet.blobs.items()
    
    nlayer = len(blob_item)
    sub_feat = {}
    sub_feat[layer_name[0]] = caffenet.blobs[layer_name[0]].data[0]     
    mdict = {}
    mdict['feature_map'] = sub_feat    
#    mdict['kernel'] = filt
    output_file = save_path + '/' + get_name(files[k]) + '.mat'
    io.savemat(output_file,mdict)
    line_out = "%d out of %d processed" % (k+1,length)
    print line_out

