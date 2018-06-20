# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:25:58 2018

@author: root
"""
import math
import functools
import sets
import tensorflow as tf

from tensorflow.python.ops import rnn, array_ops
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell

import numpy as np


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification:
    def __init__(self, x, data, target, dropout, batch_size2, num_hidden=200, dim_fv=4096, ctx_shape=[196,512]):
        self.x = x
        self.dim_fv = dim_fv
        self.data = data 
        self.ctx_shape = ctx_shape
        self.target = target
        self.batch_size = batch_size2
        self._num_hidden = num_hidden
        self._dropout = dropout
        self.num_cas = 3

        
        self.image_att_w = None
        self.image_att_wX = None
        self.image_att_b = None  
        self.x_t_w = None
        self.x_t_w_re = None
        self.x_t_b = None 
        self.x_t_b_re = None
        self.hidden_att_w = None
        self.hidden_att_b = None 
        self.att_w = None
        self.att_wX = None
        self.att_b = None 
        self.att_bX = None 
        self.gru_W = None
        self.gru_b = None 
        self.image_encode_W = None
        self.image_encode_b = None 
        self.decode_gru_W = None
        self.decode_gru_b = None 
        self.decode_class_W = None
        self.decode_class_b = None 
        self.init_hidden_W = None
        self.init_hidden_b = None 
        self.alpha_list1 = None
        self.alpha_list1_re = None
        self.alpha_list2 = None
        self.alpha_list2_re = None
        self.alpha_list3 = None
        self.alpha_list3_re = None

        self.hidden_att_w1 = None
        self.hidden_att_b1 = None
        self.hidden_att_w2 = None 
        self.hidden_att_b2 = None
        self.hidden_att_w3 = None
        self.hidden_att_b3 = None
           
        self.image_att_w1 = None
        self.image_att_b1 = None       
        self.image_att_w2 = None
        self.image_att_b2 = None        
        self.image_att_w3 = None 
        self.image_att_b3 = None           
           
        self.att_w1 = None
        self.att_b1 = None
        self.att_w2 = None
        self.att_b2 = None         
        self.att_w3 = None
        self.att_b3 = None 
        
        self.hidden_att_w_re1 = None
        self.hidden_att_b_re1 = None
        self.hidden_att_w_re2 = None
        self.hidden_att_b_re2 = None
        self.hidden_att_w_re3 = None 
        self.hidden_att_b_re3 = None
 
        self.image_att_w_re1 = None
        self.image_att_b_re1 = None
        self.image_att_w_re2 = None
        self.image_att_b_re2 = None
        self.image_att_w_re3 = None 
        self.image_att_b_re3 = None
        
        self.att_w_re1 = None
        self.att_b_re1 = None
        self.att_w_re2 = None 
        self.att_b_re2 = None   
        self.att_w_re3 = None
        self.att_b_re3 = None
           
        self.chanel_w = None
        self.chanel_b = None
        self.chanel_w_re =None
        self.chanel_b_re = None

        
        self.prediction
        self.error
        self.optimize
        self.alpha_list_com

        
    @lazy_property
    def max_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        max_length_num = tf.reduce_max(length)
        return max_length_num



    @lazy_property        
    def prediction(self):  
        max_length_com = tf.shape(self.target)[1]
        num_classes = int(self.target.get_shape()[2])

        with tf.variable_scope("ForwardGRU"):
           gru_cell_fw = GRUCell(self._num_hidden)
           gru_cell_fw = DropoutWrapper(gru_cell_fw, output_keep_prob=self._dropout)
           
           def cond(ind, h, output, alpha_list1, alpha_list2, alpha_list3):
               ml = self.max_length 
               return tf.less(ind, ml)
                  
           def body(ind, h, output, alpha_list1, alpha_list2, alpha_list3):

                 aAll = tf.zeros([self.num_cas, self.batch_size , self.ctx_shape[0]], tf.float32) #(3, batch_size, 196)
                 context = self.data[:,ind,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
                 context = tf.reshape(context,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
                 context = tf.transpose(context, [0, 2, 1]) #batch,196,512
                 context_flat = tf.reshape(context, [-1, self.ctx_shape[1]])  # (batch*196,512)                 
                 
                 ##############################  attn1 ###########################              
                 h_attn = tf.matmul(h, self.hidden_att_w1) + self.hidden_att_b1         
                 context_encode = tf.matmul(context_flat, self.image_att_w1) + self.image_att_b1 
                 context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
                 context_encode = context_encode + tf.expand_dims(h_attn, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
                 context_encode = tf.nn.tanh(context_encode)

                 context_encode_flat = tf.reshape(context_encode, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
                 alpha = tf.matmul(context_encode_flat, self.att_w1) + self.att_b1  # (batch_size*196, 1)
                 alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]]) # (batch_size, 196)
                 alpha = tf.nn.softmax(alpha) + 1e-10
                 weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) # (batch, 512)         
                          

                 h, _ = gru_cell_fw(weighted_context, h)
                 tf.get_variable_scope().reuse_variables()
                 context = context * tf.expand_dims(alpha, 2)
                 alpha_list1 = alpha_list1.write(ind, tf.expand_dims(alpha, 1)) # (batch,1, 196)                 
                 ##############################  attn2 ###########################              
                 h_attn = tf.matmul(h, self.hidden_att_w2) + self.hidden_att_b2        
                 context_flat = tf.reshape(context, [-1, self.ctx_shape[1]])  
                 context_encode = tf.matmul(context_flat, self.image_att_w2) + self.image_att_b2            
                 context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
                 context_encode = context_encode + tf.expand_dims(h_attn, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
                 context_encode = tf.nn.tanh(context_encode)

                 # compute alpha_ti --> evaluate per pixel info accross 512 maps
                 context_encode_flat = tf.reshape(context_encode, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
                 alpha = tf.matmul(context_encode_flat, self.att_w2) + self.att_b2                        
                 alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]]) # (batch_size, 196)
                 alpha = tf.nn.softmax(alpha) + 1e-10
                 weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) # (batch, 512)  
                         
                    
                 h, _ = gru_cell_fw(weighted_context, h)
                 tf.get_variable_scope().reuse_variables()
                 alpha_list2 = alpha_list2.write(ind, tf.expand_dims(alpha, 1)) # (batch,1, 196)  
                    
                 ##############################  attn2 ###########################   
#                 h_attn = tf.matmul(h, self.hidden_att_w3) + self.hidden_att_b3  
#                 context_flat = tf.reshape(context, [-1, self.ctx_shape[1]])  
#                 context_encode = tf.matmul(context_flat, self.image_att_w3) + self.image_att_b3            
#                 context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
#                 context_encode = context_encode + tf.expand_dims(h_attn, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
#                 context_encode = tf.nn.tanh(context_encode)
#
#                         # compute alpha_ti --> evaluate per pixel info accross 512 maps
#                 context_encode_flat = tf.reshape(context_encode, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
#                 alpha = tf.matmul(context_encode_flat, self.att_w3) + self.att_b3                        
#                 alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]]) # (batch_size, 196)
#                 alpha = tf.nn.softmax(alpha) + 1e-10
#                 weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) # (batch, 512)   
#                        
#
#                 h, _ = gru_cell_fw(weighted_context, h)
#                 tf.get_variable_scope().reuse_variables()
                 alpha_list3 = alpha_list3.write(ind, tf.expand_dims(alpha, 1)) # (batch,1, 196)
                 
                 output = output.write(ind, tf.expand_dims(h, 1)) # (batch,1, 200)
                 ind += 1

                 return ind, h, output, alpha_list1, alpha_list2, alpha_list3

            
           ind = tf.constant(0) 
           context = self.data[:,ind,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
           context = tf.reshape(context,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
           context = tf.transpose(context, [0, 2, 1]) #batch,196,512
           
           h, self.init_hidden_W, self.init_hidden_b =self._linear2('init_hidden_W', tf.reduce_mean(context, 1), self._num_hidden, transferweight = 0, tantanh= True) # (batch,256)    
           initial_output = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list1 = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list2 = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list3 = tf.TensorArray(dtype=tf.float32, size=self.max_length)

            ################################ weight init #####################
           self.hidden_att_w1, self.hidden_att_b1 = self._linear3('hidden_att_W1', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.image_att_w1, self.image_att_b1 = self._linear3('image_att_W1', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer         
           self.att_w1, self.att_b1 = self._linear3('att_W1', self.ctx_shape[1], 1, transferweight = 0)
         
           self.hidden_att_w2, self.hidden_att_b2 = self._linear3('hidden_att_W2', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.image_att_w2, self.image_att_b2 = self._linear3('image_att_W2', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer         
           self.att_w2, self.att_b2 = self._linear3('att_W2', self.ctx_shape[1], 1, transferweight = 0)
         
           self.hidden_att_w3, self.hidden_att_b3 = self._linear3('hidden_att_W3', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.image_att_w3, self.image_att_b3 = self._linear3('image_att_W3', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer         
           self.att_w3, self.att_b3 = self._linear3('att_W3', self.ctx_shape[1], 1, transferweight = 0)
         
            ##########################################################################3
         
           t,_,output, alpha_list1, alpha_list2, alpha_list3 = tf.while_loop(cond, body, [ind, h, initial_output, initial_alpha_list1, initial_alpha_list2, initial_alpha_list3], swap_memory=True)
           output_final = output.stack()            
           output_final = tf.reshape(output_final,[-1, self.batch_size, self._num_hidden])  # (max_seq,batch,200)
           output_final = tf.transpose(output_final, [1, 0, 2]) #batch,max_seq,200

           alpha_list1_final = alpha_list1.stack()    # (max_seq,batch,1,196)
           alpha_list1_final = tf.reshape(alpha_list1_final,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list1_final = tf.transpose(alpha_list1_final, [1, 0, 2]) #batch,max_seq,196
           self.alpha_list1 = alpha_list1_final
           

           alpha_list2_final = alpha_list2.stack()    # (max_seq,batch,1,196)
           alpha_list2_final = tf.reshape(alpha_list2_final,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list2_final = tf.transpose(alpha_list2_final, [1, 0, 2]) #batch,max_seq,196
           self.alpha_list2 = alpha_list2_final
           
           alpha_list3_final = alpha_list3.stack()    # (max_seq,batch,1,196)
           alpha_list3_final = tf.reshape(alpha_list3_final,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list3_final = tf.transpose(alpha_list3_final, [1, 0, 2]) #batch,max_seq,196
           self.alpha_list3 = alpha_list3_final
           
           

        ############################################## backward ###################################################################
        with tf.variable_scope("BackwardGRU"):
           gru_cell_re = GRUCell(self._num_hidden)
           gru_cell_re = DropoutWrapper(gru_cell_re, output_keep_prob=self._dropout)
           
           def cond_re(ind_re, h_re, output_re, alpha_list_re1, alpha_list_re2, alpha_list_re3):
               ml_re = self.max_length
               return tf.less(ind_re, ml_re)
            
          
           def body_re(ind_re, h_re, output_re, alpha_list_re1, alpha_list_re2, alpha_list_re3):
                 ind2_re = tf.constant(0) 
                 
                 aAll_re = tf.zeros([self.num_cas, self.batch_size , self.ctx_shape[0]], tf.float32) #(3, batch_size, 196)
                 data_reverse =array_ops.reverse_sequence(
                     input=self.data, seq_lengths=self.length,
                     seq_dim=1, batch_dim=0)

                 context_re = data_reverse[:,ind_re,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
                 context_re = tf.reshape(context_re,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
                 context_re = tf.transpose(context_re, [0, 2, 1]) #batch,196,512
                 context_flat_re = tf.reshape(context_re, [-1, self.ctx_shape[1]])  # (batch*196,512)


                 ##############################  attn1 ###########################                  
                 h_attn_re = tf.matmul(h_re, self.hidden_att_w_re1) + self.hidden_att_b_re1
                 context_encode_re = tf.matmul(context_flat_re, self.image_att_w_re1) + self.image_att_b_re1
                 context_encode_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
                 context_encode_re = context_encode_re + tf.expand_dims(h_attn_re, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
                 context_encode_re = tf.nn.tanh(context_encode_re)
                 # compute alpha_ti --> evaluate per pixel info accross 512 maps
                 context_encode_flat_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
                 alpha_re = tf.matmul(context_encode_flat_re, self.att_w_re1) + self.att_b_re1   # (batch_size*196, 1)
                 alpha_re = tf.reshape(alpha_re, [-1, self.ctx_shape[0]]) # (batch_size, 196)
                 alpha_re = tf.nn.softmax(alpha_re) + 1e-10
                 weighted_context_re = tf.reduce_sum(context_re * tf.expand_dims(alpha_re, 2), 1) # (batch, 512) 
                                                   
                 
                 h_re, _ = gru_cell_re(weighted_context_re, h_re)
                 tf.get_variable_scope().reuse_variables()
                 context_re = context_re * tf.expand_dims(alpha_re, 2)
                 alpha_list_re1 = alpha_list_re1.write(ind_re, tf.expand_dims(alpha_re, 1)) # (batch,1, 196)
                 ##############################  attn2 ########################### 
                         
                 h_attn_re = tf.matmul(h_re, self.hidden_att_w_re2) + self.hidden_att_b_re2
                 context_flat_re = tf.reshape(context_re, [-1, self.ctx_shape[1]])  # (batch*196,512)
                 context_encode_re = tf.matmul(context_flat_re, self.image_att_w_re2) + self.image_att_b_re2
                 context_encode_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
                 context_encode_re = context_encode_re + tf.expand_dims(h_attn_re, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
                 context_encode_re = tf.nn.tanh(context_encode_re)

                 # compute alpha_ti --> evaluate per pixel info accross 512 maps
                 context_encode_flat_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
                 alpha_re = tf.matmul(context_encode_flat_re, self.att_w_re2) + self.att_b_re2                     
                 alpha_re = tf.reshape(alpha_re, [-1, self.ctx_shape[0]]) # (batch_size, 196)
                 alpha_re = tf.nn.softmax(alpha_re) + 1e-10
                 weighted_context_re = tf.reduce_sum(context_re * tf.expand_dims(alpha_re, 2), 1) # (batch, 512)                       


                 h_re, _ = gru_cell_re(weighted_context_re, h_re)
                 tf.get_variable_scope().reuse_variables()
                 alpha_list_re2 = alpha_list_re2.write(ind_re, tf.expand_dims(alpha_re, 1)) # (batch,1, 196)                             
                 ##############################  attn3 ###########################                        
                         
#                 h_attn_re = tf.matmul(h_re, self.hidden_att_w_re3) + self.hidden_att_b_re3
#                 context_flat_re = tf.reshape(context_re, [-1, self.ctx_shape[1]])  # (batch*196,512)
#                 context_encode_re = tf.matmul(context_flat_re, self.image_att_w_re3) + self.image_att_b_re3
#                 context_encode_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
#                 context_encode_re = context_encode_re + tf.expand_dims(h_attn_re, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
#                 context_encode_re = tf.nn.tanh(context_encode_re)
#
#                         # compute alpha_ti --> evaluate per pixel info accross 512 maps
#                 context_encode_flat_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
#                 alpha_re = tf.matmul(context_encode_flat_re, self.att_w_re3) + self.att_b_re3                     
#                 alpha_re = tf.reshape(alpha_re, [-1, self.ctx_shape[0]]) # (batch_size, 196)
#                 alpha_re = tf.nn.softmax(alpha_re) + 1e-10
#                 weighted_context_re = tf.reduce_sum(context_re * tf.expand_dims(alpha_re, 2), 1) # (batch, 512) 
#           
#                 h_re, _ = gru_cell_re(weighted_context_re, h_re) 
#                 tf.get_variable_scope().reuse_variables() 
                 alpha_list_re3 = alpha_list_re3.write(ind_re, tf.expand_dims(alpha_re, 1)) # (batch,1, 196)

                 output_re = output_re.write(ind_re, tf.expand_dims(h_re, 1)) # (batch,1, 200)

                
                 ind_re += 1                
                 
                 
                 return ind_re, h_re, output_re, alpha_list_re1, alpha_list_re2, alpha_list_re3

                 
                 
           ind_re = tf.constant(0)  
           data_reverse =array_ops.reverse_sequence(
               input=self.data, seq_lengths=self.length,
               seq_dim=1, batch_dim=0)

           context_re = data_reverse[:,ind_re,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
           context_re = tf.reshape(context_re,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
           context_re = tf.transpose(context_re, [0, 2, 1]) #batch,196,512
 

           h_re, self.init_hidden_W_re, self.init_hidden_b_re =self._linear2('init_hidden_W_re', tf.reduce_mean(context_re, 1), self._num_hidden, transferweight = 0, tantanh= True) # (batch,256)    
           initial_output_re = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list_re1 = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list_re2 = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list_re3 = tf.TensorArray(dtype=tf.float32, size=self.max_length)
         
           ######################### weight initialisation #########################

           self.hidden_att_w_re1, self.hidden_att_b_re1 = self._linear3('hidden_att_W_re1', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.hidden_att_w_re2, self.hidden_att_b_re2 = self._linear3('hidden_att_W_re2', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.hidden_att_w_re3, self.hidden_att_b_re3 = self._linear3('hidden_att_W_re3', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
 
           self.image_att_w_re1, self.image_att_b_re1 = self._linear3('image_att_W_re1', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer         
           self.image_att_w_re2, self.image_att_b_re2 = self._linear3('image_att_W_re2', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer            
           self.image_att_w_re3, self.image_att_b_re3 = self._linear3('image_att_W_re3', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer            

           self.att_w_re1, self.att_b_re1 = self._linear3('att_W_re1', self.ctx_shape[1], 1, transferweight = 0) # (batch_size*196, 1)
           self.att_w_re2, self.att_b_re2 = self._linear3('att_W_re2', self.ctx_shape[1], 1, transferweight = 0) # (batch_size*196, 1)           
           self.att_w_re3, self.att_b_re3 = self._linear3('att_W_re3', self.ctx_shape[1], 1, transferweight = 0) # (batch_size*196, 1)           
           ####################################################################################################
           
           t_re,_,output_re, alpha_list_re1, alpha_list_re2, alpha_list_re3 = tf.while_loop(cond_re, body_re, [ind_re, h_re, initial_output_re, initial_alpha_list_re1, initial_alpha_list_re2, initial_alpha_list_re3], swap_memory=True)  # (max_seq,batch,1,1000)
           output_final_re = output_re.stack()
           output_final_re = tf.reshape(output_final_re,[-1, self.batch_size, self._num_hidden])  # (max_seq,batch,200)
           output_final_re = tf.transpose(output_final_re, [1, 0, 2]) #batch,max_seq,200
        
           alpha_list_final_re1 = alpha_list_re1.stack()    # (max_seq,batch,1,196)
           alpha_list_final_re1 = tf.reshape(alpha_list_final_re1,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list_final_re1 = tf.transpose(alpha_list_final_re1, [1, 0, 2]) #batch,max_seq,196
        

           alpha_list_final_re2 = alpha_list_re2.stack()    # (max_seq,batch,1,196)
           alpha_list_final_re2 = tf.reshape(alpha_list_final_re2,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list_final_re2 = tf.transpose(alpha_list_final_re2, [1, 0, 2]) #batch,max_seq,196
           
           alpha_list_final_re3 = alpha_list_re3.stack()    # (max_seq,batch,1,196)
           alpha_list_final_re3 = tf.reshape(alpha_list_final_re3,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list_final_re3 = tf.transpose(alpha_list_final_re3, [1, 0, 2]) #batch,max_seq,196
        
        

           output_final_re2 = array_ops.reverse_sequence(
               input=output_final_re, seq_lengths=self.length,
               seq_dim=1, batch_dim=0)
          
           alpha_list_final_re21 = array_ops.reverse_sequence(
               input=alpha_list_final_re1, seq_lengths=self.length,
               seq_dim=1, batch_dim=0)
               
           alpha_list_final_re22 = array_ops.reverse_sequence(
               input=alpha_list_final_re2, seq_lengths=self.length,
               seq_dim=1, batch_dim=0)
               
           alpha_list_final_re23 = array_ops.reverse_sequence(
               input=alpha_list_final_re3, seq_lengths=self.length,
               seq_dim=1, batch_dim=0)
          
           self.alpha_list1_re = alpha_list_final_re21
           self.alpha_list2_re = alpha_list_final_re22
           self.alpha_list3_re = alpha_list_final_re23
        
        
        output = tf.concat(axis=2, values=[output_final, output_final_re2])
        output = tf.reshape(output, [-1, 2*self._num_hidden])
        _temp, self.decode_class_W, self.decode_class_b  = self._linear2('decode_class_W', output, 1000, transferweight = 0) # (batch, 1000)

        prediction = tf.nn.softmax(_temp) + 1e-10
        prediction = tf.reshape(prediction, [-1, max_length_com, num_classes])

        return prediction


        
    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def alpha_list_com(self):
        pred = self.prediction
        alpha_forward1 = self.alpha_list1
        alpha_forward2 = self.alpha_list2
        alpha_forward3 = self.alpha_list3
        
        alpha_backward1 = self.alpha_list1_re
        alpha_backward2 = self.alpha_list2_re
        alpha_backward3 = self.alpha_list3_re

        
        return pred, alpha_forward1, alpha_forward2, alpha_forward3, alpha_backward1, alpha_backward2, alpha_backward3
        
    @lazy_property
    def cost(self):

        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction)  #shape = (batch_size , max_seg , hiddensize)
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)  #shape = (batch_size , max_seg)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1) #shape = (batch_size , 1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        cross_entropy = tf.reduce_mean(cross_entropy) #shape = (1 , 1)
        tf.losses.add_loss(cross_entropy)
        tf.summary.scalar("cost", cross_entropy)

        # Add L2 regularisation
        for var in tf.trainable_variables():
            with tf.name_scope("L2_regularisation/%s" % var.op.name):
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     self._regulariser(var))
        return cross_entropy 
        
        
    @lazy_property
    def optimize(self):
        learning_rate = 0.0001 
        beta = 0.0001

        with tf.control_dependencies([self.cost]):
            loss = tf.add_n(tf.losses.get_losses())
            reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            total_loss = loss + reg_losses
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("regularisation_loss", reg_losses)
        tf.summary.scalar("total_loss", total_loss)        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)           
        return optimizer.minimize(total_loss)




    @lazy_property
    def error(self):

        mistakes = tf.equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)  # true -> 1, false -> 0
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))  #shape = (batch_size, max_seg)
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, axis=1) # shape = (batch_size,1)
        mistakes /= tf.cast(self.length, tf.float32)
        accuracy = tf.reduce_mean(mistakes)  # shape = (1,1)
        tf.summary.scalar("Accuracy", accuracy)

        return accuracy


        
        
        
    @staticmethod
    def _weight_and_bias(in_size, out_size):

        weight = tf.truncated_normal([in_size, out_size], stddev=1.0/math.sqrt(float(in_size*2)))
        bias = tf.zeros([out_size])
      
        return tf.Variable(weight), tf.Variable(bias)
 

    
    
    def _regulariser(self, var):
        """A `Tensor` -> `Tensor` function that applies L2 weight loss."""
        weight_decay = tf.multiply(tf.nn.l2_loss(var),
                                   0.0001,
                                   name="weight_loss")
        return weight_decay


    def _relu(self, tensor):
        """Helper to perform leaky / regular ReLU operation."""
        return tf.nn.relu(tensor)
    
    def _tanh(self, tensor):
        """Helper to perform leaky / regular ReLU operation."""
        return tf.nn.tanh(tensor)
        
    def _linear(self, name, input_, output_dim, 
                relu=False, bias_init=0.0):
        """
        Helper to perform linear map with optional ReLU activatioself.decode_gru_W, self.decode_gru_bn.
    
        A weight decay is added to the weights variable only if one is specified.
        """
        with tf.variable_scope(name):
            input_dim = input_.get_shape()[1]
            weight = tf.get_variable(name="weight",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer=None,
                                     trainable=True)
            if bias_init is None:
                output = tf.matmul(input_, weight)
            else:
                bias = tf.get_variable(
                                name="bias",
                                shape=output_dim,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_init),
                                trainable=True)
                output = tf.matmul(input_, weight) + bias
            if relu: 
                output = self._relu(output)
            return output
        
        # x = np.random.randn(2,200).astype('f')

    def _linear2(self, name, input_, output_dim, bias_init = 0, weight_init= 0, relu=False, transferweight = 0, tantanh = False,  trainable_choice=True):
        """
        Helper to perform linear map with optional ReLU activation.
    
        A weight decay is added to the weights variable only if one is specified.
        """
        input_dim = input_.get_shape()[1]
        with tf.variable_scope(name):
            
            if transferweight == 0:
               weight_init = tf.truncated_normal_initializer(stddev=0.01)
               bias_init = 0.01 # 0.0 
            else:
               weight_init = tf.constant_initializer(weight_init) 
              
  
            weight = tf.get_variable(name="weight",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer= weight_init,
                                     trainable=trainable_choice)
                         
            bias = tf.get_variable(
                         name="bias",
                         shape=output_dim,
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(bias_init),
                         trainable=trainable_choice)          
                         
            output = tf.matmul(input_, weight) + bias 

            if relu: 
                output = self._relu(output)
                
            if tantanh: 
                output = self._tanh(output)                     
            return output, weight, bias 
          
    def _linear3(self, name, input_dim, output_dim, bias_init = 0, weight_init= 0, relu=False, transferweight = 0, tantanh = False,  trainable_choice=True):
        """
        Helper to perform linear map with optional ReLU activation.
    
        A weight decay is added to the weights variable only if one is specified.
        """
        with tf.variable_scope(name):
            
            if transferweight == 0:
               weight_init = tf.truncated_normal_initializer(stddev=0.01)
               bias_init = 0.01 # 0.0 
            else:
               weight_init = tf.constant_initializer(weight_init) 
              
  
            weight = tf.get_variable(name="weight",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer= weight_init,
                                     trainable=trainable_choice)
                         
            bias = tf.get_variable(
                         name="bias",
                         shape=output_dim,
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(bias_init),
                         trainable=trainable_choice)          
                         

            return weight, bias
    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1] #int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
        

