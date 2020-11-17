#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:40:23 2019

@author: vicari
"""

from AdvGAN import utils, dataset, cyclegan, models, print_functions, activations, layers

import tensorflow as tf
from tensorflow.layers import Flatten

from tensorflow import keras

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

def keras_conv_net(input_shape, num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model

def make_simple_conv_net(nb_class=10):
    def simple_conv_net(X, scope_name='classifier_conv_net', reuse=False):
        
        with tf.variable_scope(scope_name, reuse) as scope:
        
            if reuse:
                scope.reuse_variables()
                
            net = layers.conv2D(X, 3, 64, 1, name='conv0_1', batch_normed=True)
            net = tf.nn.relu(net)
            
            net = layers.conv2D(net, 4, 64, 2, name='conv0_2', batch_normed=True)
            net = tf.nn.relu(net)
                           
            net = layers.conv2D(X, 3, 128, 1, name='conv1_1', batch_normed=True)
            net = tf.nn.relu(net)
            
            net = layers.conv2D(net, 4, 128, 2, name='conv1_2', batch_normed=True)
            net = tf.nn.relu(net)
                     
            net = layers.conv2D(X, 3, 256, 1, name='conv2_1', batch_normed=True)
            net = tf.nn.relu(net)
            
            net = layers.conv2D(net, 4, 256, 2, name='conv2_2', batch_normed=True)
            net = tf.nn.relu(net)     
            net = Flatten()(net)
            
            net = layers.linear(net, nb_class, name='dense_layer')
            
            net = tf.nn.softmax(net)
            
            return net
    
    return simple_conv_net