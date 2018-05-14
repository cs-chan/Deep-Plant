# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:01:02 2017

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

    def __init__(self, data, target,  dropout, num_hidden=200, num_layers=2):  # default = 1.0
      
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._dropout = dropout
        self.prediction
        self.error
        self.optimize


    @lazy_property
    def max_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        max_length_num = tf.reduce_max(length)
        return max_length_num
        
    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length


    @lazy_property
    def prediction(self):
        
        max_length_com = tf.shape(self.target)[1]
        num_classes = int(self.target.get_shape()[2])
    
    
        
        with tf.variable_scope("bidirectional_rnn"):
            gru_cell_fw = GRUCell(self._num_hidden)
            gru_cell_fw = DropoutWrapper(gru_cell_fw, output_keep_prob=self._dropout)
            output_fw, _ = rnn.dynamic_rnn(
                gru_cell_fw,
                self.data,
                dtype=tf.float32,
                sequence_length=self.length,
            )
            

            
            tf.get_variable_scope().reuse_variables()
            data_reverse =array_ops.reverse_sequence(
              input=self.data, seq_lengths=self.length,
              seq_dim=1, batch_dim=0)
    
            # for reverse direction
            gru_cell_re = GRUCell(self._num_hidden)
            gru_cell_re = DropoutWrapper(gru_cell_re, output_keep_prob=self._dropout)
            tmp, _ = rnn.dynamic_rnn(
                gru_cell_re,
                data_reverse,
                dtype=tf.float32,
                sequence_length=self.length,
            )
            
            output_re = array_ops.reverse_sequence(
               input=tmp, seq_lengths=self.length,
               seq_dim=1, batch_dim=0)
      

        output = tf.concat(axis=2, values=[output_fw, output_re])
        

        
        weight, bias = self._weight_and_bias(
            2*self._num_hidden, num_classes)
        output = tf.reshape(output, [-1, 2*self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.regularizer = tf.nn.l2_loss(weight)

        
        prediction = tf.reshape(prediction, [-1, max_length_com, num_classes])


        return prediction

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

        tf.summary.scalar("cost", cross_entropy)
        return cross_entropy 
        
        
    @lazy_property
    def optimize(self):
        learning_rate = 0.0001#0.001
        beta = 0.0001

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4) 
        loss =  tf.reduce_mean(self.cost + beta * self.regularizer) 
        return optimizer.minimize(loss)




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
                                   0.00004, # 0.0001
                                   name="weight_loss")
        return weight_decay


    def _relu(self, tensor):
        """Helper to perform leaky / regular ReLU operation."""
        return tf.nn.relu(tensor)
    
        
    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1] #int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
        
   
