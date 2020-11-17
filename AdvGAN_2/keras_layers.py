#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:22:50 2019

@author: vicari
"""

# Imports
# %%
import tensorflow as tf

from AdvGAN.normalization import WeightNorm_todel, instance_norm
from AdvGAN.utils import scope_has_variables, spectral_norm


#  dense Layers
# %%
def linear(input_, output_size, use_bias=False, bias_start=0.0, spectral_normed=False, batch_normed=False,
           weight_normed=False, lr_normed=False, layer_normed=False, instance_normed=False, name="linear"):
    shape = input_.get_shape().as_list()

    with tf.compat.v1.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()

        if not weight_normed:
            weight = tf.compat.v1.get_variable("w", [shape[1], output_size], tf.float32,
                                     tf.contrib.layers.variance_scaling_initializer())

            if spectral_normed:
                weight = spectral_norm(weight)

        # print(name) #debug use only

        if weight_normed:
            mul = WeightNorm_todel(tf.keras.layers.Dense(output_size, use_bias=use_bias))(input_)
        elif batch_normed:
            mul = tf.keras.layers.BatchNormalization(tf.keras.layers.Dense(output_size, use_bias=use_bias))(input_)
        else:
            mul = tf.matmul(input_, weight)

        if use_bias and not weight_normed:
            bias = tf.compat.v1.get_variable(name="bias", shape=[output_size],
                                   initializer=tf.constant_initializer(bias_start))
            mul += bias

        # if layer_normed:
        # mul = tf.contrib.layers.layer_norm(mul)
        if batch_normed:
            mul = tf.compat.v1.layers.batch_normalization(mul)
        # if instance_normed:
        # mul = instance_norm(mul)

        return mul


#  Conv2D Layers
# %%
def conv2d(inputs, kernel, output_channel, stride, use_bias=False, name='conv_1', spectral_normed=True,
           weight_normed=False, batch_normed=False,
           layer_normed=False, instance_normed=False, lr_normed=False, stddev=0.02, padding="SAME"):
    with tf.compat.v1.variable_scope(name) as scope:

        if scope_has_variables(scope):
            scope.reuse_variables()

        if not weight_normed:
            w = tf.compat.v1.get_variable("w", [kernel, kernel, inputs.get_shape()[-1], output_channel],
                                initializer=tf.contrib.layers.variance_scaling_initializer())

            if spectral_normed:
                w = spectral_norm(w)

        if weight_normed:
            conv = WeightNorm_todel(
                tf.keras.layers.Conv2D(output_channel, kernel, use_bias=use_bias, strides=stride, padding=padding))(
                inputs)
        elif batch_normed:
            conv = tf.keras.layers.BatchNormalization(
                tf.keras.layers.Conv2D(output_channel, kernel, use_bias=use_bias, strides=stride, padding=padding))(
                inputs)
        else:
            conv = tf.nn.conv2d(inputs, w, strides=[1, stride, stride, 1], padding=padding)

        if layer_normed:
            conv = tf.contrib.layers.layer_norm(conv)
        if batch_normed:
            conv = tf.compat.v1.layers.batch_normalization(conv)
        if instance_normed:
            conv = instance_norm(conv)
        if lr_normed:
            conv = tf.nn.lrn(conv, bias=0.00005)

        return conv


def deconv2d(inputs, kernel, output_channel, strides, shape, spectral_normed=False, weight_normed=False,
             batch_normed=False,
             lr_normed=False, layer_normed=False, use_bias=False, instance_normed=False, name='conv'):
    s = [1, strides, strides, 1]
    with tf.compat.v1.variable_scope(name) as scope:

        if scope_has_variables(scope):
            scope.reuse_variables()

        if not weight_normed:
            w = tf.compat.v1.get_variable("w", [kernel, kernel, output_channel, inputs.get_shape()[-1]],
                                initializer=tf.contrib.layers.variance_scaling_initializer())

            if spectral_normed:
                w = spectral_norm(w)

        if weight_normed:
            deconv = WeightNorm_todel(
                tf.keras.layers.Conv2DTranspose(output_channel, kernel, use_bias=use_bias, strides=strides,
                                                padding='SAME'))(inputs)
        elif batch_normed:
            deconv = tf.keras.layers.BatchNormalization(
                tf.keras.layers.Conv2DTranspose(output_channel, kernel, use_bias=use_bias, strides=strides,
                                                padding='SAME'))(inputs)
        else:
            deconv = tf.nn.conv2d_transpose(inputs, w, shape, strides=s, padding='SAME')

        if layer_normed:
            deconv = tf.contrib.layers.layer_norm(deconv)
        if batch_normed:
            deconv = tf.compat.v1.layers.batch_normalization(deconv)
        if instance_normed:
            deconv = instance_norm(deconv)
        if lr_normed:
            deconv = tf.nn.lrn(deconv, bias=0.00005)

    return deconv
