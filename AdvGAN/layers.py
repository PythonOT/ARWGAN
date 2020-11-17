#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:39:34 2019

@author: various artists
"""

# Imports
#%%
from AdvGAN.utils import scope_has_variables, spectral_norm
from AdvGAN.activations import relu, lrelu

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from AdvGAN.normalization import WeightNorm, instance_norm

#  dense Layers
#%%
def linear(input_, output_size, use_bias=False, bias_start=0.0, spectral_normed=False, batch_normed=False, batch_renormed=False, weight_normed=False, lr_normed=False,
           layer_normed=False, instance_normed=False, is_training=False , name="linear"):
    
    shape = input_.get_shape().as_list()
   
    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
            
        if not weight_normed: 
            weight = tf.get_variable("w", [shape[1], output_size], tf.float32, tf.contrib.layers.variance_scaling_initializer()) 
            
            if spectral_normed:
                weight = spectral_norm(weight)
        
        #print(name) #debug use only

        if weight_normed:
            mul = WeightNorm(tf.keras.layers.Dense(output_size, use_bias=use_bias))(input_)
        else:
            mul = tf.matmul(input_, weight)
        
        if use_bias and not (weight_normed):
            bias = tf.get_variable(name="bias", shape=[output_size],
                                   initializer=tf.constant_initializer(bias_start))
            mul += bias
            
        if batch_renormed:
            clip = {'rmax' : tf.constant(3, dtype=tf.float32), 'rmin': tf.constant(1/3, dtype=tf.float32), 'dmax': tf.constant(5, dtype=tf.float32) }
            mul = tf.layers.batch_normalization(mul, training=is_training, renorm=True, renorm_clipping=clip)            
        if batch_normed:
            mul = tf.layers.batch_normalization(mul, training=is_training)
        #if layer_normed:
            #mul = tf.contrib.layers.layer_norm(mul)
        #if instance_normed:
            #mul = instance_norm(mul)


        return mul

#  Conv2D Layers
#%%

def conv2D(inputs, kernel, output_channel, stride, use_bias=False, name='conv_1', spectral_normed=True, weight_normed=False, batch_normed=False, depthwise=False,
           batch_renormed=False, layer_normed=False, instance_normed=False, lr_normed=False, is_training=False, stddev=0.02, padding="SAME"):

    with tf.variable_scope(name) as scope:
        
        if scope_has_variables(scope):
            scope.reuse_variables()
            
        if not weight_normed: 
            w = tf.get_variable("w", [kernel, kernel, inputs.get_shape()[-1], output_channel],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
                        
            if spectral_normed:
                w = spectral_norm(w)

        if depthwise:
            conv = tf.keras.layers.SeparableConv2D(output_channel, kernel, use_bias=use_bias, strides=stride, padding=padding)(inputs)
        if weight_normed:
            conv = WeightNorm(tf.keras.layers.Conv2D(output_channel, kernel, use_bias=use_bias, strides=stride, padding=padding))(inputs)
        else:
            conv = tf.nn.conv2d(inputs, w, strides=[1, stride, stride, 1], padding=padding)
            
        if batch_renormed:
            clip = {'rmax' : tf.constant(3, dtype=tf.float32), 'rmin': tf.constant(1/3, dtype=tf.float32), 'dmax': tf.constant(5, dtype=tf.float32) }
            conv = tf.layers.batch_normalization(conv, training=is_training, renorm=True, renorm_clipping=clip)
            #conv = tf.layers.batch_normalization(conv)
        
        if batch_normed:
            conv = tf.layers.batch_normalization(conv, training=is_training)
            
        if layer_normed:
            conv = tf.contrib.layers.layer_norm(conv)
        if instance_normed:
            conv = instance_norm(conv)          
        if lr_normed:
            conv = tf.nn.lrn(conv, bias=0.00005)
            
        return conv


def deconv2D(inputs, kernel, output_channel,strides, shape, spectral_normed=False, weight_normed=False, batch_normed=False, 
             batch_renormed=False, lr_normed=False, layer_normed=False, use_bias=False, instance_normed=False, is_training= False, name='conv'):
    
    s = [1, strides, strides, 1]
    with tf.variable_scope(name) as scope:
    
        if scope_has_variables(scope):
            scope.reuse_variables()
      
        if not weight_normed:  
            w = tf.get_variable("w", [kernel, kernel, output_channel, inputs.get_shape()[-1]],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
            
            if spectral_normed:
                w = spectral_norm(w)

        if weight_normed:
            deconv = WeightNorm(tf.keras.layers.Conv2DTranspose(output_channel, kernel, use_bias=use_bias, strides=strides, padding='SAME'))(inputs)
        else:
            deconv = tf.nn.conv2d_transpose(inputs, w, shape, strides=s, padding='SAME')
           
        if batch_renormed:
            clip = {'rmax' : tf.constant(3, dtype=tf.float32), 'rmin': tf.constant(1/3, dtype=tf.float32), 'dmax': tf.constant(5, dtype=tf.float32) }
            deconv = tf.layers.batch_normalization(deconv, training=is_training, renorm=True, renorm_clipping=clip)
        if batch_normed:
            deconv = tf.layers.batch_normalization(deconv, training=is_training)
        if layer_normed:
            deconv = tf.contrib.layers.layer_norm(deconv)
        if instance_normed:
            deconv = instance_norm(deconv)
        if lr_normed:
            deconv = tf.nn.lrn(deconv, bias=0.00005)
            
    return deconv

#  Conv1D Layers
#%%

def conv1D(inputs, kernel, output_channel, stride, use_bias=False, name='conv_1', stddev=0.02, padding="SAME", spectral_normed=True, batch_normed=False, depthwise=False,
           batch_renormed=False, layer_normed=False, instance_normed=False, lr_normed=False, is_training=False):

    with tf.variable_scope(name) as scope:
        
        if scope_has_variables(scope):
            scope.reuse_variables()
            
        w = tf.get_variable("w", [kernel, inputs.get_shape()[-1], output_channel],
                    initializer=tf.contrib.layers.xavier_initializer())
                            
        if spectral_normed:
            w = spectral_norm(w)
        
        conv = tf.nn.conv1d(inputs, w, stride=stride, padding=padding)
        
        if batch_renormed:
            clip = {'rmax' : tf.constant(3, dtype=tf.float32), 'rmin': tf.constant(1/3, dtype=tf.float32), 'dmax': tf.constant(5, dtype=tf.float32) }
            conv = tf.layers.batch_normalization(conv, training=is_training, renorm=True, renorm_clipping=clip)
            #conv = tf.layers.batch_normalization(conv)
        if batch_normed:
            conv = tf.layers.batch_normalization(conv, training=is_training)        
        if layer_normed:
            conv = tf.contrib.layers.layer_norm(conv)
        if instance_normed:
            conv = instance_norm(conv)          
        if lr_normed:
            conv = tf.nn.lrn(conv, bias=0.00005)

        return conv  
    
def deconv1D(inputs, kernel, output_channel,stride, shape, name='conv', spectral_normed=False, weight_normed=False, batch_normed=False, 
             batch_renormed=False, lr_normed=False, layer_normed=False, use_bias=False, instance_normed=False, is_training= False):
    with tf.variable_scope(name) as scope:
    
        if scope_has_variables(scope):
            scope.reuse_variables()
      
        w = tf.get_variable("w", [kernel, output_channel, inputs.get_shape()[-1]],
                    initializer=tf.contrib.layers.xavier_initializer())
                                 
        if spectral_normed:
            w = spectral_norm(w)

        deconv = tf.contrib.nn.conv1d_transpose(inputs, w, shape, stride=stride, padding='SAME')
        
        if batch_renormed:
            clip = {'rmax' : tf.constant(3, dtype=tf.float32), 'rmin': tf.constant(1/3, dtype=tf.float32), 'dmax': tf.constant(5, dtype=tf.float32) }
            deconv = tf.layers.batch_normalization(deconv, training=is_training, renorm=True, renorm_clipping=clip)
        if batch_normed:
            deconv = tf.layers.batch_normalization(deconv, training=is_training)
        if layer_normed:
            deconv = tf.contrib.layers.layer_norm(deconv)
        if instance_normed:
            deconv = instance_norm(deconv)
        if lr_normed:
            deconv = tf.nn.lrn(deconv, bias=0.00005)
            
    return deconv

#  Conv3D Layers
#%%
def conv3_sn(inputs, kernel, output_channel, stride, use_bias=False, name='conv_1', spectral_normed=True, stddev=0.02, padding="SAME"):

    with tf.variable_scope(name) as scope:
        
        if scope_has_variables(scope):
            scope.reuse_variables()
            
            
        w = tf.get_variable("w", [kernel, kernel, kernel, inputs.get_shape()[-1], output_channel],
                    initializer=tf.contrib.layers.xavier_initializer())
        
        if spectral_normed:
            conv = tf.nn.conv3d(inputs, spectral_norm(w), strides=[1, stride, stride, stride, 1], padding=padding)
        else:
             conv = tf.nn.conv3d(inputs, w, strides=[1, stride, stride, stride, 1], padding=padding)
             
        return conv

def deconv3d_layer(inputs, kernel, output_channel,strides, shape, name='conv'):
    s = [1, strides, strides, strides, 1]
    with tf.variable_scope(name) as scope:
    
        if scope_has_variables(scope):
            scope.reuse_variables()
      
        w = tf.get_variable("w", [kernel, kernel, kernel, output_channel, inputs.get_shape()[-1]],
                    initializer=tf.contrib.layers.xavier_initializer())

        deconv = tf.nn.conv3d_transpose(inputs, w, shape, strides=s, padding='SAME')
    return deconv
    
# https://github.com/brade31919/SRGAN-tensorflow
#%%
def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [size[0], h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [size[0], h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def UpsampleConv(input, output_dim, filter_size, name, spectral_normed=False, update_collection=None, he_init=True):
    output = input
    output = tf.concat([output, output, output, output], axis=3)
    output = tf.depth_to_space(output, 2)
    return conv2D(output, filter_size, output_dim, spectral_normed=spectral_normed, name=name)
    
def residual_block(inputs, output_channel, stride, scope, train = True, h=relu(), use_bn=True, spectral_normed=True):

    with tf.variable_scope(scope):
        net = conv2D(inputs, 3, output_channel, stride, use_bias=False, name='conv1', spectral_normed=spectral_normed)
        if use_bn:
            net = tf.layers.batch_normalization(net, training=train)
        net = h(net)
        net = conv2D(net, 3, output_channel, stride, use_bias=False, name='conv2', spectral_normed=spectral_normed)
        if use_bn:
            net = tf.layers.batch_normalization(net, training=train)
        net = net + inputs
        net = h(net)
        
    return net

def disc_block(inputs, output_channel, stride, scope, train=True, h=lrelu(0.2), use_bn=False, spectral_normed=True):

    with tf.variable_scope(scope):
        net = conv2D(inputs, 3, output_channel, stride, use_bias=False, name='conv1', spectral_normed=spectral_normed)
        if use_bn:
            net = tf.layers.batch_normalization(net, training=train)
        net = h(net)
        net = conv2D(net, 3, output_channel, stride, use_bias=False, name='conv2', spectral_normed=spectral_normed)
        if use_bn:
            net = tf.layers.batch_normalization(net, training=train)
        net = h(net)
        
    return net

def discriminator_block(inputs, output_channel, kernel, stride, scope, h=lrelu(0.2), spectral_normed=False):

    with tf.variable_scope(scope):
        net = conv2D(inputs, kernel, output_channel, stride, use_bias=False, name='conv1', spectral_normed=spectral_normed)
        net = h(net)
        net = conv2D(net, kernel, output_channel, stride, use_bias=False, name='conv2', spectral_normed=spectral_normed)
        net = h(net)
        
    return net

# Lightweight gated
#%%

@add_arg_scope
def light_gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation='leaky_relu', use_lrn=True,training=True):
    assert padding in ['SAME']
    x = tf.keras.layers.SeparableConv2D(cnum, ksize, strides=stride, padding=padding, dilation_rate=rate, activation=None, name=name)(x_in) 
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation=='leaky_relu':
        x = tf.nn.leaky_relu(x)
    
    g = tf.keras.layers.SeparableConv2D(cnum, ksize, strides=stride, padding=padding, dilation_rate=rate, activation=tf.nn.sigmoid, name=name+'_g')(x_in)

    x = tf.multiply(x,g)
    return x, g

@add_arg_scope
def light_gate_deconv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv", training=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])
        b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        g = tf.nn.sigmoid(deconv)

        deconv = tf.multiply(g,deconv)

    return deconv, g

# 
#%%
    
@add_arg_scope
def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation='leaky_relu', use_lrn=True,training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x_in, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)    
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation=='leaky_relu':
        x = tf.nn.leaky_relu(x)

    g = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+'_g')

    x = tf.multiply(x,g)
    return x, g

@add_arg_scope
def gate_deconv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv", training=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])
        b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        g = tf.nn.sigmoid(deconv)

        deconv = tf.multiply(g,deconv)

    return deconv, g

@add_arg_scope
def gate_conv_3d(x_in, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation='leaky_relu', use_lrn=True,training=True):

    x = tf.layers.conv3d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)    
    if use_lrn:
        x = tf.layers.batch_normalization(x, training=training)
    if activation=='leaky_relu':
        x = tf.nn.leaky_relu(x)

    g = tf.layers.conv3d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+'_g')

    x = tf.multiply(x,g)
    return x, g

@add_arg_scope
def gate_deconv_3d(input_, output_shape, k_h=5, k_w=5, k_d=5, d_h=2, d_w=2, d_d=2, stddev=0.02,
       name="deconv", training=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w,  d_d, 1])

        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w,  d_d, 1])
        b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        g = tf.nn.sigmoid(deconv)

        deconv = tf.multiply(g,deconv)

    return deconv, g