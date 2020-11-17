#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 09:04:49 2019

@author: vicari
"""
import tensorflow as tf

from tensorflow import keras

import numpy as np


# from layers import discriminator_block


###############################################################################
###############################################################################
###############################################################################

# Simple
# %%

class Dense2D(keras.layers.Layer):

    def __init__(self, h, w, k, activation=None):
        super(Dense2D, self).__init__()

        self.h = h
        self.w = w
        self.k = k

        self.dense = keras.layers.Dense(self.h * self.w * self.k, activation=activation)

    def call(self, inputs, training=None):
        x = tf.reshape(self.dense(inputs), shape=[-1, self.h, self.w, self.k])
        return x


class GeneratorConv2D(keras.Model):

    def __init__(self, hiddens_dims=(512, 256, 128, 64), img_size=32, nb_channels=1, activation='relu', output='tanh',
                 use_bias=False):
        super(GeneratorConv2D, self).__init__()

        p = 2 ** (len(hiddens_dims) - 1)
        s = int(img_size / p)

        # Dense
        self.dense = Dense2D(s, s, hiddens_dims[0], activation)

        # Convs
        self.convs = keras.models.Sequential(name='dynamic-convs-gen')

        for conv_id in range(1, len(hiddens_dims)):
            up = keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')

            conv1 = keras.layers.Conv2D(filters=hiddens_dims[conv_id], kernel_size=(3, 3), strides=(1, 1),
                                        padding='same', activation=activation, use_bias=use_bias)

            conv2 = keras.layers.Conv2D(filters=hiddens_dims[conv_id], kernel_size=(3, 3), strides=(1, 1),
                                        padding='same', activation=activation, use_bias=use_bias)

            self.convs.add(up)
            self.convs.add(conv1)
            self.convs.add(conv2)

        self.last_conv = keras.layers.Conv2D(filters=nb_channels, kernel_size=(5, 5), strides=(1, 1),
                                             padding='same', activation=output, use_bias=use_bias)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.dense(inputs)
        x = self.convs(x)
        x = self.last_conv(x)
        return x


class DiscriminatorConv2D(keras.Model):

    def __init__(self, hiddens_dims=(64, 128, 256, 512), activation='relu', output=None, use_bias=False):
        super(DiscriminatorConv2D, self).__init__()

        # Convs
        self.convs = keras.models.Sequential(name='dynamic-convs-disc')

        for conv_id in range(1, len(hiddens_dims)):
            conv1 = keras.layers.Conv2D(filters=hiddens_dims[conv_id], kernel_size=(3, 3), strides=(1, 1),
                                        padding='same', activation=activation, use_bias=use_bias)

            conv2 = keras.layers.Conv2D(filters=hiddens_dims[conv_id], kernel_size=(3, 3), strides=(2, 2),
                                        padding='same', activation=activation, use_bias=use_bias)

            self.convs.add(conv1)
            self.convs.add(conv2)

        self.flatten = keras.layers.Flatten()

        self.dense = keras.layers.Dense(1, activation=output, use_bias=use_bias)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.convs(inputs)
        x = self.dense(self.flatten(x))
        return x


###############################################################################
###############################################################################

class FullyConnected(keras.Model):

    def __init__(self, hiddens_dims=(64, 128, 256, 512), use_bias=False, activation='relu', output=None, ouput_dims=1):
        super(FullyConnected, self).__init__()

        # fcs
        self.fcs = keras.models.Sequential(name='dynamic-convs')

        for fc_id in range(1, len(hiddens_dims)):
            fc = keras.layers.Dense(hiddens_dims[fc_id], activation=activation, use_bias=use_bias)
            self.fcs.add(fc)

        self.dense = keras.layers.Dense(ouput_dims, activation=output, use_bias=use_bias)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.fcs(inputs)
        x = self.dense(x)
        return x


###############################################################################
###############################################################################


class Dense1D(keras.layers.Layer):

    def __init__(self, w, k, activation=None):
        super(Dense1D, self).__init__()

        self.w = w
        self.k = k

        self.dense = keras.layers.Dense(self.w * self.k, activation=activation)

    def call(self, inputs, training=None):
        x = tf.reshape(self.dense(inputs), shape=[-1, self.w, self.k])
        return x


class GeneratorConv1D(keras.Model):

    def __init__(self, hiddens_dims=(512, 256, 128), img_size=32, nb_channels=1, activation='relu', output='tanh'):
        super(GeneratorConv1D, self).__init__()

        p = 2 ** (len(hiddens_dims) - 1)
        s = int(img_size / p) + img_size%2

        # Dense
        self.dense = Dense1D(s, hiddens_dims[0], activation)

        # Convs
        self.convs = keras.models.Sequential(name='dynamic-convs-gen')

        for conv_id in range(1, len(hiddens_dims)):
            up = keras.layers.UpSampling1D(size=2)

            conv1 = keras.layers.Conv1D(filters=hiddens_dims[conv_id], kernel_size=3, strides=1,
                                        padding='same', activation=activation)

            conv2 = keras.layers.Conv1D(filters=hiddens_dims[conv_id], kernel_size=3, strides=1,
                                        padding='same', activation=activation)

            self.convs.add(up)
            self.convs.add(conv1)
            self.convs.add(conv2)

        self.last_conv = keras.layers.Conv1D(filters=nb_channels, kernel_size=5, strides=1,
                                             padding='same', activation=output)
        
        self.crop = keras.layers.Cropping1D(cropping=(0,img_size%2))

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.dense(inputs)
        x = self.convs(x)
        x = self.last_conv(x)
        x = self.crop(x)
        return x


class DiscriminatorConv1D(keras.Model):

    def __init__(self, hiddens_dims=(64, 128, 256, 512), activation='relu', output=None, im_size=4):
        super(DiscriminatorConv1D, self).__init__()

        # Convs
        #self.pad = keras.layers.ZeroPadding1D(padding=(0,im_size%2))
        
        self.convs = keras.models.Sequential(name='dynamic-convs-disc')

        for conv_id in range(1, len(hiddens_dims)):
            conv1 = keras.layers.Conv1D(filters=hiddens_dims[conv_id], kernel_size=3, strides=1,
                                        padding='same', activation=activation)

            conv2 = keras.layers.Conv1D(filters=hiddens_dims[conv_id], kernel_size=3, strides=2,
                                        padding='same', activation=activation)

            self.convs.add(conv1)
            self.convs.add(conv2)

        self.flatten = keras.layers.Flatten()

        self.dense = keras.layers.Dense(1, activation=output)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.convs(inputs)
        x = self.dense(self.flatten(x))
        return x


class GatedConv(keras.layers.Layer):

    def __init__(self, cnum, k, stride=1, rate=1, activation=None, padding='SAME', name='gated_conv'):
        super(GatedConv, self).__init__()

        self.conv = keras.layers.Conv2D(cnum, k, strides=stride, dilation_rate=rate,
                                        activation=activation, padding=padding, name=name)

        self.gated_mask = keras.layers.Conv2D(cnum, k, strides=stride, dilation_rate=rate,
                                              activation=tf.nn.sigmoid, padding=padding, name=name + "_mask")

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        gated_mask = self.gated_mask(inputs)
        return x * gated_mask


class GatedDeconv(keras.layers.Layer):

    def __init__(self, cnum, k, stride=1, rate=1, activation=None, padding='SAME', name='gated_deconv'):
        super(GatedDeconv, self).__init__()

        self.up = keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')

        self.conv = keras.layers.Conv2D(cnum, k, strides=stride, dilation_rate=rate,
                                        activation=activation, padding=padding, name=name)

        self.gated_mask = keras.layers.Conv2D(cnum, k, strides=stride, dilation_rate=rate,
                                              activation=tf.nn.sigmoid, padding=padding, name=name + "_mask")

    def call(self, inputs, training=None):
        x_in = self.up(inputs)
        x = self.conv(x_in)
        gated_mask = self.gated_mask(x_in)
        return x * gated_mask


class GeneratorUnet(keras.Model):

    def __init__(self, nb_channels=3, activation='relu', output='tanh'):
        super(GeneratorUnet, self).__init__()

        cnum = 32

        # Encoder
        self.enc1 = GatedConv(cnum, 5, 2, activation=activation)
        self.enc2 = GatedConv(2 * cnum, 3, 2, activation=activation)
        self.enc3 = GatedConv(4 * cnum, 3, 2, activation=activation)
        self.enc4 = GatedConv(8 * cnum, 3, 2, activation=activation)

        # Dilat
        self.dilat1 = GatedConv(8 * cnum, 3, 1, rate=2, activation=activation)
        self.dilat2 = GatedConv(8 * cnum, 3, 1, rate=2, activation=activation)
        self.dilat3 = GatedConv(8 * cnum, 3, 1, rate=4, activation=activation)
        self.dilat4 = GatedConv(8 * cnum, 3, 1, rate=4, activation=activation)

        # Decoder
        self.dec1 = GatedDeconv(4 * cnum, 3, activation=activation)
        self.gate_conv_1 = GatedConv(4 * cnum, 3, 1, activation=activation)
        self.dec2 = GatedDeconv(2 * cnum, 3, activation=activation)
        self.gate_conv_2 = GatedConv(2 * cnum, 3, 1, activation=activation)
        self.dec3 = GatedDeconv(cnum, 3, activation=activation)
        self.gate_conv_3 = GatedConv(cnum, 3, 1, activation=activation)
        self.dec4 = GatedDeconv(nb_channels, 3, activation=activation)
        self.gate_conv_4 = GatedConv(nb_channels, 5, 1, activation=None)

        return

    def call(self, inputs, mask, z, training=None, **kwargs):

        if mask is not None:
            m = (2 * mask) - 1
            x_in = tf.concat([inputs, z, m], axis=-1)
        else:
            x_in = inputs

        # Encoder
        x_1 = self.enc1(x_in)
        x_2 = self.enc2(x_1)
        x_3 = self.enc3(x_2)
        x_4 = self.enc4(x_3)

        # Dilat
        x_dilat1 = self.dilat1(x_4)
        x_dilat2 = self.dilat2(x_dilat1)
        x_dilat3 = self.dilat3(x_dilat2)
        x_dilat4 = self.dilat4(x_dilat3)

        # decoder
        x = self.dec1(x_dilat4)
        x = tf.concat([x_3, x], axis=-1)
        x = self.gate_conv_1(x)

        x = self.dec2(x)
        x = tf.concat([x_2, x], axis=-1)
        x = self.gate_conv_2(x)

        x = self.dec3(x)
        x = tf.concat([x_1, x], axis=-1)
        x = self.gate_conv_3(x)

        x = self.dec4(x)
        x = tf.concat([inputs, x], axis=-1)
        x = self.gate_conv_4(x)

        return tf.nn.tanh(x)


class DiscriminatorBlock(keras.layers.Layer):

    def __init__(self, dims, activation='relu'):
        super(DiscriminatorBlock, self).__init__()

        self.conv1 = keras.layers.Conv2D(filters=dims, kernel_size=(3, 3), strides=2,
                                         padding='same', activation=activation)

        self.conv2 = keras.layers.Conv2D(filters=dims, kernel_size=(3, 3), strides=1,
                                         padding='same', activation=activation)

        self.conv3 = keras.layers.Conv2D(filters=dims, kernel_size=(3, 3), strides=1,
                                         padding='same', activation=activation)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DiscriminatorUnet(keras.Model):

    def __init__(self, activation='relu', output=None):
        super(DiscriminatorUnet, self).__init__()

        # Convs
        self.block1 = DiscriminatorBlock(32, activation=activation)
        self.block2 = DiscriminatorBlock(64, activation=activation)
        self.block3 = DiscriminatorBlock(128, activation=activation)
        self.block4 = DiscriminatorBlock(256, activation=activation)

        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(1, activation=output)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.dense(self.flatten(x))
        return x


class TransformBlock(keras.layers.Layer):

    def __init__(self, n_points, activation='relu', last_dense=9,
                 last_weight=(np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)),
                 shape=(3, 3)):
        super(TransformBlock, self).__init__()

        self.conv1 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1,
                                         padding='same', activation=activation)

        self.conv2 = keras.layers.Conv1D(filters=1024, kernel_size=1, strides=1,
                                         padding='same', activation=activation)

        self.max_pool = keras.layers.MaxPool1D(n_points)

        self.dense1 = keras.layers.Dense(512, activation=activation)

        self.dense2 = keras.layers.Dense(256, activation=activation)

        self.dense_last = keras.layers.Dense(last_dense, activation=activation, weights=last_weight)

        self.reshape = keras.layers.Reshape(shape)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_last(x)
        x = self.reshape(x)
        return x


class FeaturePointNet(keras.layers.Layer):

    def __init__(self, n_points=2048, activation='relu'):
        super(FeaturePointNet, self).__init__()

        cnum = 32

        self.input_transformation = TransformBlock(n_points, activation=activation,
                                                   last_dense=9,
                                                   last_weight=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)],
                                                   shape=(3, 3))

        self.conv1 = keras.layers.Convolution1D(64, 1, activation=activation)
        self.conv2 = keras.layers.Convolution1D(64, 1, activation=activation)

        self.feature_transformation = TransformBlock(n_points, activation=activation,
                                                     last_dense=64 * 64,
                                                     last_weight=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)],
                                                     shape=(64, 64))

        self.conv3 = keras.layers.Convolution1D(64, 1, activation=activation)
        self.conv4 = keras.layers.Convolution1D(128, 1, activation=activation)
        self.conv5 = keras.layers.Convolution1D(1024, 1, activation=activation)

        self.global_feature = keras.layers.MaxPooling1D(pool_size=n_points)

        return

    def call(self, inputs, training=None, **kwargs):
        input_transformed = self.input_transformation(inputs)
        x = tf.matmul(inputs, input_transformed)
        x = self.conv1(x)
        x = self.conv2(x)

        feature_transformed = self.feature_transformation(x)

        x = tf.matmul(x, feature_transformed)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.global_feature(x)

        return x


class DiscriminatorPointNet(keras.Model):

    def __init__(self, n_points=2048, activation='relu', output=None):
        super(DiscriminatorPointNet, self).__init__()

        # Convs
        self.features = FeaturePointNet(n_points=n_points, activation=activation)

        self.dense1 = keras.layers.Dense(512, activation=output)
        self.dense2 = keras.layers.Dense(256, activation=output)
        self.dense3 = keras.layers.Dense(1, activation=output)

        return

    def call(self, inputs, training=None, **kwargs):
        x = self.features(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x


class GeneratorPointNet(keras.Model):

    def __init__(self, n_points=2048, activation='relu', output='tanh'):
        super(GeneratorPointNet, self).__init__()

        # Convs

        self.dense1 = keras.layers.Dense(128, activation=activation)
        self.dense2 = keras.layers.Dense(256, activation=activation)
        self.dense3 = keras.layers.Dense(512, activation=activation)

        

        self.dense4 = keras.layers.Dense(n_points, activation=activation)
        self.dense5 = keras.layers.Dense(3*n_points, activation=output)

        self.reshape = keras.layers.Reshape((n_points, 3))

        return

    def call(self, z, mask=None, training=None, **kwargs):
        x = self.dense1(z)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        x = self.reshape(x)

        return x
