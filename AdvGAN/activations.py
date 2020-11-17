#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 08:39:05 2019

@author: vicari
"""
import tensorflow as tf
import numpy as np

def MaxMin():
    def f(x):
        shape = x.get_shape().as_list()
        shape[0] = -1
        group = tf.reshape(x,[-1, 2])
        a,b = tf.split(group, 2, 1)
        c = tf.concat((tf.minimum(a,b), tf.maximum(b,a)), axis=1)
        res = tf.reshape(c,shape) 
        return res
    
    return lambda x : f(x)

def FullSort():
    def f(x):
        shape = x.get_shape().as_list()
        res = -tf.nn.top_k(-x, k=shape[-1])[0]
        return res
    
    return lambda x : f(x)

def lrelu(l=0.2):
    return lambda x : tf.nn.leaky_relu(x,l)

def relu():
    return lambda x : tf.nn.relu(x)