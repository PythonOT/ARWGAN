#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 09:04:49 2019

@author: vicari
"""
import tensorflow as tf
from tensorflow.layers import Flatten
#from layers import discriminator_block

from AdvGAN import utils, dataset, cyclegan, models, print_functions, activations, layers

###############################################################################
###############################################################################
###############################################################################

# Simple 
#%%  
def make_g_conv(img_size=32, hiddens_dims=[512,256,128,64], nb_channels=1, scope_name="generator", use_sn=False, use_bn=False, use_brn=False, use_wn=False, use_ln=False, use_in=False, use_bias=False,
            use_lrn=False, h=activations.relu(), o=None, default_reuse=False, default_train=True, depthwise=False):
    
    s = img_size
    s_ = []
    for i in range(len(hiddens_dims)):
        p = 2**i
        s_.append(int(s/p))
    
    def network(X, reuse=default_reuse, train=default_train, scope_name=scope_name):
            
        with tf.variable_scope(scope_name, reuse) as scope:
            
            bs = tf.shape(X)[0]
            net = layers.linear(X, s_[-1]*s_[-1]*hiddens_dims[0], name='dense_layer', spectral_normed=use_sn, instance_normed=use_in, 
                                                         weight_normed=use_wn, layer_normed=use_ln, use_bias=use_bias, is_training=train)
            net = tf.reshape(net, [-1, s_[-1], s_[-1], hiddens_dims[0]])
            if use_bn:
                net = tf.layers.batch_normalization(net, training=train)
            net = h(net)
    
            for i in range(1,len(hiddens_dims)):
                                
                net = layers.conv2D(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i), lr_normed=use_lrn, weight_normed=use_wn, depthwise=depthwise,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
                net = h(net)
                
                net = layers.deconv2D(net, 3, hiddens_dims[i], 2, shape=[bs, s_[-i-1], s_[-i-1], hiddens_dims[i]], name='deconv'+str(i), batch_normed=use_bn,
                                                         batch_renormed=use_brn, lr_normed=use_lrn, instance_normed=use_in, weight_normed=use_wn, layer_normed=use_ln, spectral_normed=use_sn, is_training=train)
                net = h(net)
                
            net = layers.conv2D(net, 3, nb_channels, 1, use_bias=use_bias, name='conv_last', spectral_normed=use_sn, is_training=train, depthwise=depthwise)
    
            if(o != None):
                net = o(net)
        
            return net
        
    def build_net(X, y=None, z=None, name=scope_name, reuse=False, train=True):

        res = network(X, reuse=reuse, train=train)
        return res 

    return build_net

def make_d_conv(hiddens_dims=[64,128,256,512], kernel_size=3, scope_name="discriminator", stride=2, use_sn=False, use_ln=False, use_bn=False, use_in=False, use_bias=False,
            use_brn=False, use_lrn=False, use_wn=False, h=activations.relu(), o=None, default_reuse=True, default_train=True, n_class=10, keep_prob=0.5):
    
    def network(X, y=None, reuse=default_reuse, train=default_train, ct=False, dropout=False, scope_name=scope_name):
        
        with tf.variable_scope(scope_name, reuse) as scope:
    
            if reuse:
                scope.reuse_variables()
                
            net = layers.conv2D(X, 3, hiddens_dims[0], 1, use_bias=use_bias, name='conv0_1', lr_normed=use_lrn, weight_normed=use_wn,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
            net = h(net)
            net = layers.conv2D(net, 3, hiddens_dims[0], 2, use_bias=use_bias, name='conv0_2', lr_normed=use_lrn, weight_normed=use_wn,
                                   batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
            net = h(net)           
            if dropout:
                net = tf.nn.dropout(net, keep_prob=keep_prob) 
                
            for i in range(1,len(hiddens_dims)-1):                                                      
                net = layers.conv2D(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i)+'_1', lr_normed=use_lrn, weight_normed=use_wn,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
                net = h(net)
                
                net = layers.conv2D(net, 3, hiddens_dims[i], 2, use_bias=use_bias, name='conv'+str(i)+'_2', lr_normed=use_lrn, weight_normed=use_wn,
                                           batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, layer_normed=use_ln, spectral_normed=use_sn, is_training=train)
                net = h(net)
                if dropout:
                    net = tf.nn.dropout(net, keep_prob=keep_prob)
            
            net = Flatten()(net)
            
            if y is not None:
                net = tf.concat([net, y],axis=1)
            
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias, is_training=train)
            
            return net
           
    def build_net(x, y=None, name=scope_name,reuse=False, train=True, ct=False, dropout=False):
        
        res = network(x, y, reuse=reuse, train=train, ct=ct, dropout=dropout, scope_name=name)
        return res

    return build_net

###############################################################################
###############################################################################
###############################################################################
    
# Classical 
#%%  
def make_dc_gen_conv(img_size=32, hiddens_dims=[512,256,128,64], nb_channels=1, scope_name="generator", use_sn=False, use_bn=True, use_bias=False,
            h=activations.relu(), o=None, default_reuse=False, default_train=True, conditional=False):
    
    s = img_size
    s_ = []
    for i in range(len(hiddens_dims)):
        p = 2**i
        s_.append(int(s/p))
    
    def network(X, reuse=default_reuse, train=default_train):
            
        with tf.variable_scope(scope_name, reuse) as scope:
            
            bs = tf.shape(X)[0]
    
            net = layers.linear(X, s_[-1]*s_[-1]*hiddens_dims[0], name='dense_layer', spectral_normed=False, use_bias=use_bias)
            net = tf.reshape(net, [-1, s_[-1], s_[-1], hiddens_dims[0]])
            if use_bn:
                net = tf.layers.batch_normalization(net, training=train)
            net = h(net)
    
            for i in range(1,len(hiddens_dims)):
                net = layers.deconv2D(net, 4, hiddens_dims[i], 2, shape=[bs, s_[-i-1], s_[-i-1], hiddens_dims[i]], name='deconv'+str(i))
                if use_bn:
                    net = tf.layers.batch_normalization(net, training=train)
                net = h(net)
                
            #print(net.get_shape().as_list())
            net = layers.conv2D(net, 3, nb_channels, 1, use_bias=use_bias, name='conv_last', spectral_normed=use_sn)
    
            if(o != None):
                net = o(net)
            
            return net
        
    def build_net(X, y=None, z=None, name=scope_name, reuse=False, train=True):
        
        if conditional:
            batch_data = tf.concat([z, y],axis=1)
            res = network(batch_data, reuse=reuse, train=train)
            return res
        else:
            res = network(X, reuse=reuse, train=train)
            return res 

    return build_net

def make_dc_disc_conv(hiddens_dims=[64,128,256,512],kernel_size=3, stride=2, scope_name="discriminator", use_sn=False, use_bn=False, use_bias=False,
            h=activations.relu(), o=None, default_reuse=True, default_train=True, conditional=False, n_class=10):
    
    def network(X, y=None, reuse=default_reuse, train=default_train):
        
        with tf.variable_scope(scope_name, reuse) as scope:
    
            if reuse:
                scope.reuse_variables()
                
            net = layers.conv2D(X, 3, hiddens_dims[0], 1, use_bias=use_bias, name='conv0_1', spectral_normed=use_sn)
            net = h(net)
            net = layers.conv2D(net, 4, hiddens_dims[0], 2, use_bias=use_bias, name='conv0_2', spectral_normed=use_sn)
            net = h(net)
            
            for i in range(1,len(hiddens_dims)-1):
                net = layers.conv2D(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i)+'_1', spectral_normed=use_sn)
                net = h(net)
                net = layers.conv2D(net, 4, hiddens_dims[i], 2, use_bias=use_bias, name='conv'+str(i)+'_2', spectral_normed=use_sn)
                net = h(net)
                
            net = layers.conv2D(net, 3, hiddens_dims[-1], 1, use_bias=use_bias, name='conv_last', spectral_normed=use_sn)
            net = h(net)
            
            net = Flatten()(net)
            
            if y is not None:
                net = tf.concat([net, y],axis=1)
            
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=False, use_bias=use_bias)
            
            return net
           
    def build_net(x, y=None, name=scope_name, reuse=False, train=True):
        
        if conditional:
            #s = tf.shape(x)
            #y = tf.reshape(y, (s[0], 1, 1, n_class))
            #a = tf.ones((s[0], s[1], s[2], n_class)) * y
            #batch_data = tf.concat([x, a],axis=-1)
            res = network(x, y, reuse=reuse, train=train)
            return res
        else:
            res = network(x, reuse=reuse, train=train)
            return res 

    return build_net

# Fully Connected
#%%   
def make_fc(hiddens_dims=[256,256,256], ouput_dims=2, scope_name="net", use_sn=False, use_bn=False, use_bias=True, use_in=False,
            use_brn=False, h=activations.relu(), o=None, default_reuse=True, default_train=True):
    
    def network(X, reuse=default_reuse, train=default_train):
        with tf.variable_scope(scope_name) as scope:
    
            if reuse:
                scope.reuse_variables()
                
            net = layers.linear(X, hiddens_dims[0], name='dense_layer_0', spectral_normed=use_sn, batch_normed=use_bn, instance_normed=use_in, use_bias=use_bias, is_training=train)
            net = h(net)
            
            for i in range(1,len(hiddens_dims)):
                net = layers.linear(net, hiddens_dims[i], name='dense_layer_'+str(i), spectral_normed=use_sn, batch_normed=use_bn, instance_normed=use_in, use_bias=use_bias, is_training=train)
                net = h(net)
            
            net = layers.linear(net, ouput_dims, name='dense_layer_'+str(len(hiddens_dims)), spectral_normed=use_sn, instance_normed=use_in, use_bias=use_bias, is_training=train)
            
            if o != None:
                net = o(net)
                
            return net
        
    return network

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def make_g_conv_1d(img_size=48, hiddens_dims=[512,256,128,64], nb_channels=1, scope_name="generator", use_sn=False, use_bn=False, use_brn=False, use_ln=False, use_in=False, use_bias=False,
            use_lrn=False, h=activations.relu(), o=None, default_reuse=False, default_train=True, depthwise=False):
    
    s = img_size
    s_ = []
    for i in range(len(hiddens_dims)):
        p = 2**i
        s_.append(int(s/p))
    
    def network(X, reuse=default_reuse, train=default_train, scope_name=scope_name):
            
        with tf.variable_scope(scope_name, reuse) as scope:
            
            bs = tf.shape(X)[0]
            net = layers.linear(X, s_[-1]*hiddens_dims[0], name='dense_layer', spectral_normed=use_sn, instance_normed=use_in, 
                                layer_normed=use_ln, use_bias=use_bias, is_training=train)
            net = tf.reshape(net, [-1, s_[-1], hiddens_dims[0]])
            if use_bn:
                net = tf.layers.batch_normalization(net, training=train)
            net = h(net)
    
            for i in range(1,len(hiddens_dims)):
                                
                net = layers.conv1D(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i), lr_normed=use_lrn,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
                net = h(net)
                
                net = layers.deconv1D(net, 3, hiddens_dims[i], 2, shape=[bs, s_[-i-1], hiddens_dims[i]], name='deconv'+str(i), batch_normed=use_bn,
                                                         batch_renormed=use_brn, lr_normed=use_lrn, instance_normed=use_in, layer_normed=use_ln, spectral_normed=use_sn, is_training=train)
                net = h(net)
                
            net = layers.conv1D(net, 3, nb_channels, 1, use_bias=use_bias, name='conv_last', spectral_normed=use_sn, is_training=train, depthwise=depthwise)
    
            if(o != None):
                net = o(net)
            
            return net
        
    def build_net(X, y=None, z=None, name=scope_name, reuse=False, train=True):

        res = network(X, reuse=reuse, train=train)
        return res 

    return build_net

def make_d_conv_1d(hiddens_dims=[64,128,256,512], kernel_size=3, scope_name="discriminator", stride=2, use_sn=False, use_ln=False, use_bn=False, use_in=False, use_bias=False,
            use_brn=False, use_lrn=False, h=activations.relu(), o=None, default_reuse=True, default_train=True, n_class=10, keep_prob=0.5):
    
    def network(X, y=None, reuse=default_reuse, train=default_train, ct=False, dropout=False, scope_name=scope_name):
        
        with tf.variable_scope(scope_name, reuse) as scope:
    
            if reuse:
                scope.reuse_variables()
                
            net = layers.conv1D(X, 3, hiddens_dims[0], 1, use_bias=use_bias, name='conv0_1', lr_normed=use_lrn,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
            net = h(net)
            net = layers.conv1D(net, 3, hiddens_dims[0], 2, use_bias=use_bias, name='conv0_2', lr_normed=use_lrn,
                                   batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
            net = h(net)           
            if dropout:
                net = tf.nn.dropout(net, keep_prob=keep_prob) 
                
            for i in range(1,len(hiddens_dims)-1):                                                      
                net = layers.conv1D(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i)+'_1', lr_normed=use_lrn,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
                net = h(net)
                
                net = layers.conv1D(net, 3, hiddens_dims[i], 2, use_bias=use_bias, name='conv'+str(i)+'_2', lr_normed=use_lrn,
                                           batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, layer_normed=use_ln, spectral_normed=use_sn, is_training=train)
                net = h(net)
                if dropout:
                    net = tf.nn.dropout(net, keep_prob=keep_prob)
            
            net = Flatten()(net)
            
            if y is not None:
                net = tf.concat([net, y],axis=1)
            
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias, is_training=train)
            
            return net
           
    def build_net(x, y=None, name=scope_name,reuse=False, train=True, ct=False, dropout=False):
        
        res = network(x, y, reuse=reuse, train=train, ct=ct, dropout=dropout, scope_name=name)
        return res

    return build_net

    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
# SRGAN
#%%   
    
def make_gen_sr(img_size=32, nb_channels=1, scope_name="net", use_sn=False, use_bn=True, use_bias=False,
            h=activations.relu(), o=None, default_reuse=False, default_train=True):
    
    def network(X, train=default_train, reuse=default_reuse):
        s = img_size 
        s2, s4, s8 = int(s/2), int(s/4), int(s/8)
        gf_dim = 64
        
        with tf.variable_scope(scope_name) as scope:
    
            if reuse:
                scope.reuse_variables()
    
            net_h0 = layers.linear(X, s4*s4*gf_dim, name='dense_layer_gen', spectral_normed=use_sn, use_bias=use_bias)
            if use_bn:
                net_h0 = tf.layers.batch_normalization(net_h0, training=train)
            net_h0 = tf.reshape(net_h0, [-1, s4, s4, gf_dim])
            net = h(net_h0)
    
            input_stage = net
    
            for i in range(1, 5, 1):
                name_scope = 'resblock_%d'%(i)
                net = layers.residual_block(net, 64, 1, name_scope, train=train, h=h, use_bn=use_bn, spectral_normed=use_sn)
            if use_bn:
                net = tf.layers.batch_normalization(net, training=train) 
            net = h(net)
            
            net = input_stage + net
            
            
            net = layers.conv2D(net, 3, 256, 1, use_bias=use_bias, name='conv1', spectral_normed=use_sn)
            net = layers.pixelShuffler(net, scale=2)
            if use_bn:
                net =  tf.layers.batch_normalization(net, training=train)
            net = h(net)
            
            net = layers.conv2D(net, 3, 256, 1, use_bias=use_bias, name='conv2', spectral_normed=use_sn)
            net = layers.pixelShuffler(net, scale=2)
            if use_bn:
                net =  tf.layers.batch_normalization(net, training=train)
            net = h(net)
            
            net = layers.conv2D(net, 3, nb_channels, 1, use_bias=use_bias, name='conv4', spectral_normed=use_sn)
    
            if(o != None):
                net = o(net)
    
            return net
    
    return network

#unert
#%%
'''
def generator_unet(image, options, reuse=False, name="generator"):

    def network(X, y, train=default_train, reuse=default_reuse):
        dropout_rate = 0.5 if options.is_training else 1.0
        with tf.variable_scope(name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
    
            # image is (256 x 256 x input_c_dim)
            
            e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
            # e1 is (128 x 128 x self.gf_dim)
            e2 = instance_norm(conv2d(activations.lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = instance_norm(conv2d(activations.lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = instance_norm(conv2d(activations.lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = instance_norm(conv2d(activations.lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = instance_norm(conv2d(activations.lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = instance_norm(conv2d(activations.lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = instance_norm(conv2d(activations.lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
            # e8 is (1 x 1 x self.gf_dim*8)
    
            d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
            d1 = tf.nn.dropout(d1, dropout_rate)
            d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)
    
            d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
            d2 = tf.nn.dropout(d2, dropout_rate)
            d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)
    
            d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
            d3 = tf.nn.dropout(d3, dropout_rate)
            d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)
    
            d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
            d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)
    
            d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
            d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)
    
            d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
            d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)
    
            d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
            d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)
    
            d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
            # d8 is (256 x 256 x output_c_dim)

    return tf.nn.tanh(d8)
'''
###############################################################################
###############################################################################
###############################################################################

# unet with mask
#%% 
def d_resnet(use_bias=False, use_sn=False, h=activations.lrelu(0.2), default_reuse=True):
    
    def network(X, reuse=default_reuse):
    
        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()
     
            #16
            net = layers.conv2D(X, 3, 32, 2, use_bias=use_bias, name='dis_conv_0', spectral_normed=use_sn)
            net = h(net)
               
            net = layers.discriminator_block(net, 32, 3, 1, 'disblock_1_1')
            net = layers.discriminator_block(net, 32, 3, 1, 'disblock_1_2')
    
            #8
            net = layers.conv2D(net, 3, 64, 2, use_bias=use_bias, name='dis_conv_1', spectral_normed=use_sn)
            net = h(net)
    
       
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_1')
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_2')
    
            #4
            net = layers.conv2D(net, 3, 128, 2, use_bias=use_bias, name='dis_conv_2', spectral_normed=use_sn)
            net = h(net)
    
    
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_1')
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_2')
    
            # 2
            net = layers.conv2D(net, 3, 256, 2, use_bias=use_bias, name='dis_conv_3', spectral_normed=use_sn)
            net = h(net)
    
            
            net = layers.discriminator_block(net, 256, 3, 1, 'disblock_4_1')
            net = layers.discriminator_block(net, 256, 3, 1, 'disblock_4_2')
    
            # 1
            net = layers.conv2D(net, 3, 512, 2, use_bias=use_bias, name='dis_conv_4', spectral_normed=use_sn)
            net = h(net)
            
            
            net = tf.reshape(net, [-1, 1 * 1 * 512])
    
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
                
            return net
    
    return network

# Unet : default settings do not use Spectral Norm and use LeakyReLU, ReLU could do it too
def unet_gen(img_size=32, bs=64, nb_channels=3, scope_name="generator", use_bias=False, use_sn=False, h=activations.lrelu(0.2), default_reuse=False, o=tf.tanh, ignore_mask=False):
    
    def network(x, name='generator',reuse=False, trainig=True):

        cnum = 32
        s_h, s_w = img_size, img_size
        s_h2, s_w2 = int(img_size/2), int(img_size/2)
        s_h4, s_w4 = int(img_size/4), int(img_size/4)

        with tf.variable_scope(name, reuse=reuse):

            # encoder
            x_now = x
            x1, _ = layers.gate_conv(x,cnum,5,2,use_lrn=False,name='gconv1_ds')
            x2, _ = layers.gate_conv(x1,2*cnum,3,2,name='gconv2_ds')
            x3, _ = layers.gate_conv(x2,4*cnum,3,2,name='gconv3_ds')

            # dilated conv
            x4,_ = layers.gate_conv(x3, 4*cnum, 3, 1, rate=2, name='co_conv1_dlt')
            x4,_ = layers.gate_conv(x4, 4*cnum, 3, 1, rate=4, name='co_conv2_dlt')

            # decoder_x
            x5, _ = layers.gate_deconv(x4,[bs, s_h4, s_w4, 2*cnum], name='deconv5')
            x5 = tf.concat([x2,x5],axis=-1)
            x5, _ = layers.gate_conv(x5, 2*cnum,3,1,name='gconv5')

            x6, _ = layers.gate_deconv(x5,[bs, s_h2, s_w2, cnum], name='deconv6')
            x6 = tf.concat([x1,x6],axis=-1)
            x6, _ = layers.gate_conv(x6, cnum,3,1,name='gconv6')

            x7, _ = layers.gate_deconv(x6,[bs, s_h, s_w, nb_channels], name='deconv7')
            x7 = tf.concat([x_now,x7],axis=-1)
            x7, _ = layers.gate_conv(x7, nb_channels,5,1,activation=None,use_lrn=False,name='gconv7')
            
            output = o(x7)

        return output, None
    
    def build_net(x, mask, z, name=scope_name, reuse=default_reuse, train=True):
        if mask is not None:

            m = (2*mask)-1
            batch_data = tf.concat([x, z, m],axis=-1)
                
            gen_img, _ = network(batch_data, name=name, reuse=reuse, trainig=train)
            
            if ignore_mask:
                return gen_img
            
            res = gen_img*mask + incom_imgs
            return res
        else:
            gen_img, _ = network(x, name=name, reuse=reuse, trainig=train)
            return gen_img

    return build_net

# Unet : default settings do not use Spectral Norm and use LeakyReLU, ReLU could do it too
def unet_gen_deep(img_size=32, bs=64, nb_channels=3, scope_name="generator", use_bias=False, use_sn=False, h=activations.lrelu(0.2), default_reuse=False, o=tf.tanh, ignore_mask=False):
    
    def network(x, name='generator',reuse=False, trainig=True):

        cnum = 32
        s_h, s_w = img_size, img_size
        s_h2, s_w2 = int(img_size/2), int(img_size/2)
        s_h4, s_w4 = int(img_size/4), int(img_size/4)
        s_h8, s_w8 = int(img_size/8), int(img_size/8)

        with tf.variable_scope(name, reuse=reuse):

            # encoder
            x_now = x
            enc_1, _ = layers.gate_conv(x,cnum,5,2,use_lrn=False,name='gconv1_ds')
            enc_2, _ = layers.gate_conv(enc_1,2*cnum,3,2,name='gconv2_ds')
            enc_3, _ = layers.gate_conv(enc_2,4*cnum,3,2,name='gconv3_ds')
            enc_4, _ = layers.gate_conv(enc_3,8*cnum,3,2,name='gconv4_ds')
            

            # dilated conv
            dilat_1,_ = layers.gate_conv(enc_4, 8*cnum, 3, 1, rate=2, name='co_conv1_dlt')
            dilat_2,_ = layers.gate_conv(dilat_1, 8*cnum, 3, 1, rate=4, name='co_conv2_dlt')

            # decoder_x
            dec_4 , _ = layers.gate_deconv(dilat_2,[bs, s_h8, s_w8, 4*cnum], name='deconv4')
            dec_4 = tf.concat([enc_3,dec_4],axis=-1)
            dec_4, _ = layers.gate_conv(dec_4, 4*cnum,3,1,name='gconv4')
            
            dec_3, _ = layers.gate_deconv(dec_4,[bs, s_h4, s_w4, 2*cnum], name='deconv3')
            dec_3 = tf.concat([enc_2,dec_3],axis=-1)
            dec_3, _ = layers.gate_conv(dec_3, 2*cnum,3,1,name='gconv3')

            dec_2, _ = layers.gate_deconv(dec_3,[bs, s_h2, s_w2, cnum], name='deconv2')
            dec_2 = tf.concat([enc_1,dec_2],axis=-1)
            dec_2, _ = layers.gate_conv(dec_2, cnum,3,1,name='gconv2')

            dec_1, _ = layers.gate_deconv(dec_2,[bs, s_h, s_w, nb_channels], name='deconv1')
            dec_1 = tf.concat([x_now,dec_1],axis=-1)
            dec_1, _ = layers.gate_conv(dec_1, nb_channels,5,1,activation=None,use_lrn=False,name='gconv1')
            
            output = o(dec_1)

        return output, None
    
    def build_net(x, mask, z, name=scope_name, reuse=default_reuse, train=True):
        if mask is not None:

            m = (2*mask)-1
            batch_data = tf.concat([x, z, m],axis=-1)
                
            gen_img, _ = network(batch_data, name=name, reuse=reuse, trainig=train)
            
            if ignore_mask:
                return gen_img
            
            res = gen_img*mask + incom_imgs
            return res
        else:
            gen_img, _ = network(x, name=name, reuse=reuse, trainig=train)
            return gen_img

    return build_net


def mask_disc(hiddens_dims=[64,128,256,512], name="discriminator", default_reuse=True, default_train=False,use_bias=False, use_sn=True, mask=None):

    def network(X, reuse=default_reuse, use_bias=False, use_sn=True, h=activations.lrelu()):
    
        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            #4
            net = layers.conv2D(X, 3, 64, 2, use_bias=use_bias, name='dis_conv_1', spectral_normed=use_sn)
            net = h(net)
    
       
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_1')
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_2')
    
            #2
            net = layers.conv2D(net, 3, 128, 2, use_bias=use_bias, name='dis_conv_2', spectral_normed=use_sn)
            net = h(net)
    
    
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_1')
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_2')
    
            # 1
            net = layers.conv2D(net, 3, 256, 2, use_bias=use_bias, name='dis_conv_3', spectral_normed=use_sn)
            net = h(net)
            
            net = Flatten()(net)

    
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
                
            return net

    def build_net(x, mask, reuse=True, train=False, use_sn=True):
        if mask is not None:
            batch_data = tf.concat([x, mask],axis=-1)
            res = network(batch_data, reuse)
            return res
        else:
            res = network(x, reuse)
            return res
    
    
    return build_net  

def mask_disc_deep(hiddens_dims=[64,128,256,512], name="discriminator", default_reuse=True, default_train=False,use_bias=False, use_sn=True, mask=None):

    def network(X, reuse=default_reuse, use_bias=False, use_sn=True, h=activations.lrelu()):
    
        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            #8
            net = layers.conv2D(X, 3, 32, 2, use_bias=use_bias, name='dis_conv_0', spectral_normed=use_sn)
            net = h(net)
    
       
            net = layers.discriminator_block(net, 32, 3, 1, 'disblock_1_1')
            net = layers.discriminator_block(net, 32, 3, 1, 'disblock_1_2')
            
            #4
            net = layers.conv2D(net, 3, 64, 2, use_bias=use_bias, name='dis_conv_1', spectral_normed=use_sn)
            net = h(net)
    
       
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_1')
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_2')
    
            #2
            net = layers.conv2D(net, 3, 128, 2, use_bias=use_bias, name='dis_conv_2', spectral_normed=use_sn)
            net = h(net)
    
    
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_1')
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_2')
    
            # 1
            net = layers.conv2D(net, 3, 256, 2, use_bias=use_bias, name='dis_conv_3', spectral_normed=use_sn)
            net = h(net)
            
            net = Flatten()(net)

    
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
                
            return net

    def build_net(x, mask, reuse=True, train=False, use_sn=True):
        if mask is not None:
            batch_data = tf.concat([x, mask],axis=-1)
            res = network(batch_data, reuse)
            return res
        else:
            res = network(x, reuse)
            return res
    
    
    return build_net  

###############################################################################
###############################################################################
###############################################################################
    
# Unet : default settings do not use Spectral Norm and use LeakyReLU, ReLU could do it too
def unet_gen_labels(img_size=32, bs=64, nb_classes=10, nb_channels=3, scope_name="generator", use_bias=False, use_sn=False, h=activations.lrelu(0.2), default_reuse=False, o=tf.tanh,label_is_in=False, label_is_out=False):
    
    def network(x, y=None, name='generator',reuse=False, trainig=True):

        cnum = 32
        s_h, s_w = img_size, img_size
        s_h2, s_w2 = int(img_size/2), int(img_size/2)
        s_h4, s_w4 = int(img_size/4), int(img_size/4)

        with tf.variable_scope(name, reuse=reuse):

            # encoder
            x_now = x
            
            if label_is_in:
                x_ = tf.ones_like(x)           
                x_ = x_ * tf.reshape(y, (-1, 1, 1, nb_classes))
                x = tf.concat([x, x_],axis=-1)
                
            x1, _ = layers.gate_conv(x,cnum,5,2,use_lrn=False,name='gconv1_ds')
            x2, _ = layers.gate_conv(x1,2*cnum,3,2,name='gconv2_ds')
            x3, _ = layers.gate_conv(x2,4*cnum,3,2,name='gconv3_ds')

            # dilated conv
            x4,_ = layers.gate_conv(x3, 4*cnum, 3, 1, rate=2, name='co_conv1_dlt')
            x4,_ = layers.gate_conv(x4, 4*cnum, 3, 1, rate=4, name='co_conv2_dlt')

            # decoder_x
            x5, _ = layers.gate_deconv(x4,[bs, s_h4, s_w4, 2*cnum], name='deconv5')
            x5 = tf.concat([x2,x5],axis=-1)
            x5, _ = layers.gate_conv(x5, 2*cnum,3,1,name='gconv5')

            x6, _ = layers.gate_deconv(x5,[bs, s_h2, s_w2, cnum], name='deconv6')
            x6 = tf.concat([x1,x6],axis=-1)
            x6, _ = layers.gate_conv(x6, cnum,3,1,name='gconv6')

            x7, _ = layers.gate_deconv(x6,[bs, s_h, s_w, nb_channels], name='deconv7')
            x7 = tf.concat([x_now,x7],axis=-1)
            x7, _ = layers.gate_conv(x7, nb_channels,5,1,activation=None,use_lrn=False,name='gconv7')
            
            output = o(x7)
            
            if label_is_out:
                # decoder_y
                y = Flatten()(x4)
    
        
                y = layers.linear(y, nb_classes, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
                    
                y = tf.nn.softmax(y)
            
                return output, y
            else:
                return output
    
    def build_net(x, y=None, name=scope_name, reuse=default_reuse, train=True):
        if label_is_out:
            gen_img, y = network(x, name=scope_name, reuse=reuse, trainig=train)
            return gen_img , y      
        if label_is_in:
            gen_img = network(x, y, name=scope_name, reuse=reuse, trainig=train)
            return gen_img

    return build_net 

def mask_disc_labels(hiddens_dims=[64,128,256,512], nb_classes=10, name="discriminator", default_reuse=True, default_train=False,use_bias=False, use_sn=True, h=activations.lrelu()):

    def network(X, y, reuse=default_reuse, use_bias=False, use_sn=True):
    
        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            #4
            x_ = tf.ones_like(X)
            x_ = x_ * tf.reshape(y, (-1, 1, 1, nb_classes))
            X = tf.concat([X, x_],axis=-1)

            net = layers.conv2D(X, 3, 64, 2, use_bias=use_bias, name='dis_conv_1', spectral_normed=use_sn)
            net = h(net)
    
       
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_1')
            net = layers.discriminator_block(net, 64, 3, 1, 'disblock_2_2')
    
            #2
            net = layers.conv2D(net, 3, 128, 2, use_bias=use_bias, name='dis_conv_2', spectral_normed=use_sn)
            net = h(net)
    
    
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_1')
            net = layers.discriminator_block(net, 128, 3, 1, 'disblock_3_2')
    
            # 1
            net = layers.conv2D(net, 3, 256, 2, use_bias=use_bias, name='dis_conv_3', spectral_normed=use_sn)
            net = h(net)
            
            net = Flatten()(net)

            net = tf.concat([net, y], axis=-1)
    
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
                
            return net

    def build_net(x, y, reuse=True, train=False, use_sn=True):
        y_norm = y*2 - 1
        #batch_data = tf.concat([x, y_norm],axis=-1)
        res = network(x, y_norm, reuse)
        return res
    
    
    return build_net       
 
###############################################################################
###############################################################################
###############################################################################
# unet with mask
#%% 
           
def unet_gen_3d(img_size=128, batch_size=128, nb_channels=4, use_bias=False, use_sn=False, h=activations.lrelu(0.2), default_reuse=False):
    
    def network(x, mask, name='generator',reuse=False, trainig=True):
        
        bs = batch_size
        cnum = 64
        s_h, s_w, s_d = img_size, img_size, nb_channels
        s_h2, s_w2, s_d2 = int(img_size/2), int(img_size/2), int(nb_channels/2)
        s_h4, s_w4, s_d4 = int(img_size/4), int(img_size/4), int(nb_channels/4)

        with tf.variable_scope(name, reuse=reuse):

            # encoder
            x_now = x
            x1, mask1 = layers.gate_conv_3d(x,cnum,5,2,use_lrn=False,name='gconv1_ds')
            x2, mask2 = layers.gate_conv_3d(x1,2*cnum,3,2,name='gconv2_ds')
            x3, mask3 = layers.gate_conv_3d(x2,4*cnum,3,2,name='gconv3_ds')

            # dilated conv
            x4,_ = layers.gate_conv_3d(x3, 4*cnum, 3, 1, rate=2, name='co_conv1_dlt')
            x4,_ = layers.gate_conv_3d(x4, 4*cnum, 3, 1, rate=4, name='co_conv2_dlt')

            # decoder

            x12, _ = layers.gate_deconv_3d(x4,[bs, s_d4, s_h4, s_w4, 2*cnum], name='deconv5')
            x12 = tf.concat([x2,x12],axis=-1)
            x12, mask12 = layers.gate_conv_3d(x12,2*cnum,3,1,name='gconv12')

            x13, _ = layers.gate_deconv_3d(x12,[bs, s_d2 , s_h2, s_w2, cnum], name='deconv6')
            x13 = tf.concat([x1,x13],axis=-1)
            x13, mask13 = layers.gate_conv_3d(x13,cnum,3,1,name='gconv13')

            x14, _ = layers.gate_deconv_3d(x13,[bs, s_d, s_h, s_w, 1], name='deconv7')
            x14 = tf.concat([x_now,x14],axis=-1)
            x14, mask14 = layers.gate_conv_3d(x14,1,3,1,activation=None,use_lrn=False,name='gconv14')
            
            output = tf.tanh(x14)

        return output, mask14
    
    def build_net(x, mask, z, name='generator',reuse=False, train=True):
        incom_imgs = x*(1-mask)
        a = tf.ones_like(x)
        a_mask = a*mask
        a_z = a*z
        s = tf.shape(x)
        a_incom_imgs = tf.reshape(incom_imgs, [s[0], s[3], s[1], s[2], 1])
        a_mask = tf.reshape(a_mask, [s[0], s[3], s[1], s[2], 1])
        a_z = tf.reshape(a_z, [s[0], s[3], s[1], s[2], 1])
        batch_data = tf.concat([a_incom_imgs, a_mask, a_z],axis=-1)
        gen_img, _ = network(batch_data, mask, name=name, reuse=reuse, trainig=train)
        gen_img = tf.reshape(gen_img, [s[0], s[1], s[2], s[3]])
        res = gen_img*mask + incom_imgs
        return res

    return build_net

def mask_disc_3d(hiddens_dims=[64,128,256], name="discriminator", default_reuse=True, default_train=False,use_bias=False, use_sn=True, h=activations.lrelu()):

    def network(X, reuse=True, train=default_train):
        
        with tf.variable_scope(name, reuse=reuse):
     
            net = layers.conv3_sn(X, 3, hiddens_dims[0], 1, use_bias=use_bias, name='conv0_1', spectral_normed=use_sn)
            net = h(net)
            net = layers.conv3_sn(net, 4, hiddens_dims[0], 2, use_bias=use_bias, name='conv0_2', spectral_normed=use_sn)
            net = h(net)
            
            for i in range(1,len(hiddens_dims)-1):
                net = layers.conv3_sn(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i)+'_1', spectral_normed=use_sn)
                net = h(net)
                net = layers.conv3_sn(net, 4, hiddens_dims[i], 2, use_bias=use_bias, name='conv'+str(i)+'_2', spectral_normed=use_sn)
                net = h(net)
                
            net = layers.conv3_sn(net, 3, hiddens_dims[-1], 1, use_bias=use_bias, name='conv_last', spectral_normed=use_sn)
            net = h(net)
    
            net = tf.reshape(net, [-1, hiddens_dims[-1]])
            
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
            
            return net
    
        return network

    def build_net(x, mask, reuse=True, train=False, use_sn=True):
        #batch_data = tf.concat([x, mask],axis=-1)
        #print(batch_data.get_shape().as_list())
        s = tf.shape(x)
        x = tf.reshape(x, [s[0], s[3], s[1], s[2], 1])
        res = network(x, reuse)
        return res
    
    
    return build_net 

###############################################################################
###############################################################################
###############################################################################
# unet without mask
#%%            
def unet_gen_3d_all(img_size=8, batch_size=8, nb_channels=48, use_bias=False, use_sn=False, h=activations.lrelu(0.2), default_reuse=False):
    
    def network(x, mask, name='generator',reuse=False, trainig=True):
        
        bs = batch_size
        cnum = 64
        s_h, s_w, s_d = img_size, img_size, nb_channels
        s_h2, s_w2, s_d2 = int(img_size/2), int(img_size/2), int(nb_channels/2)
        s_h4, s_w4, s_d4 = int(img_size/4), int(img_size/4), int(nb_channels/4)

        with tf.variable_scope(name, reuse=reuse):

            # encoder
            x_now = x
            x1, mask1 = layers.gate_conv_3d(x,cnum,5,2,use_lrn=False,name='gconv1_ds')
            x2, mask2 = layers.gate_conv_3d(x1,2*cnum,3,2,name='gconv2_ds')
            x3, mask3 = layers.gate_conv_3d(x2,4*cnum,3,2,name='gconv3_ds')

            # dilated conv
            x4,_ = layers.gate_conv_3d(x3, 4*cnum, 3, 1, rate=2, name='co_conv1_dlt')
            x4,_ = layers.gate_conv_3d(x4, 4*cnum, 3, 1, rate=4, name='co_conv2_dlt')

            # decoder

            x12, _ = layers.gate_deconv_3d(x4,[bs, s_d4, s_h4, s_w4, 2*cnum], name='deconv5')
            x12 = tf.concat([x2,x12],axis=-1)
            x12, mask12 = layers.gate_conv_3d(x12,2*cnum,3,1,name='gconv12')

            x13, _ = layers.gate_deconv_3d(x12,[bs, s_d2 , s_h2, s_w2, cnum], name='deconv6')
            x13 = tf.concat([x1,x13],axis=-1)
            x13, mask13 = layers.gate_conv_3d(x13,cnum,3,1,name='gconv13')

            x14, _ = layers.gate_deconv_3d(x13,[bs, s_d, s_h, s_w, 1], name='deconv7')
            x14 = tf.concat([x_now,x14],axis=-1)
            x14, mask14 = layers.gate_conv_3d(x14,1,3,1,activation=None,use_lrn=False,name='gconv14')
            
            output = tf.tanh(x14)

        return output, mask14
    
    def build_net(x, mask, z, name='generator',reuse=False, train=True):
        a = tf.ones_like(x)
        a_mask = a*mask
        a_z = a*z
        s = tf.shape(x)
        a_incom_imgs = tf.reshape(x, [s[0], s[3], s[1], s[2], 1])
        a_mask = tf.reshape(a_mask, [s[0], s[3], s[1], s[2], 1])
        a_z = tf.reshape(a_z, [s[0], s[3], s[1], s[2], 1])
        batch_data = tf.concat([a_incom_imgs, a_mask, a_z],axis=-1)
        gen_img, _ = network(batch_data, mask, name=name, reuse=reuse, trainig=train)
        gen_img = tf.reshape(gen_img, [s[0], s[1], s[2], s[3]])
        mask = tf.clip_by_value(mask, 0, 1)
        res = gen_img*mask + x*(1-mask)
        return res

    return build_net

def mask_disc_3d_all(hiddens_dims=[64,128,256,512], name="discriminator", default_reuse=True, default_train=False,use_bias=False, use_sn=True, h=activations.lrelu()):

    def network(X, reuse=True, train=default_train):
        
        with tf.variable_scope(name, reuse=reuse):
     
            net = layers.conv3_sn(X, 3, hiddens_dims[0], 1, use_bias=use_bias, name='conv0_1', spectral_normed=use_sn)
            net = h(net)
            net = layers.conv3_sn(net, 4, hiddens_dims[0], 2, use_bias=use_bias, name='conv0_2', spectral_normed=use_sn)
            net = h(net)
            
            for i in range(1,len(hiddens_dims)-1):
                net = layers.conv3_sn(net, 3, hiddens_dims[i], 1, use_bias=use_bias, name='conv'+str(i)+'_1', spectral_normed=use_sn)
                net = h(net)
                net = layers.conv3_sn(net, 4, hiddens_dims[i], 2, use_bias=use_bias, name='conv'+str(i)+'_2', spectral_normed=use_sn)
                net = h(net)
                
            net = layers.conv3_sn(net, 3, hiddens_dims[-1], 1, use_bias=use_bias, name='conv_last', spectral_normed=use_sn)
            net = h(net)
    
            net = tf.reshape(net, [-1, 6*512])
            
            net = layers.linear(net, 1, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias)
            
            return net
    
        return network

    def build_net(x, mask, reuse=True, train=False, use_sn=True):
        #a = tf.ones_like(x)
        #a_mask = a*mask
        s = tf.shape(x)
        x = tf.reshape(x, [s[0], s[3], s[1], s[2], 1])
        #a_mask = tf.reshape(a_mask, [s[0], s[3], s[1], s[2], 1])
        #batch_data = tf.concat([x, a_mask],axis=-1)
        #print(batch_data.get_shape().as_list())
        #s = tf.shape(batch_data)
        res = network(x, reuse)
        return res
    
    
    return build_net 

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
# Cycle 
#%%  

def make_cycle_g_t_conv(img_size=32, hiddens_dims=[512,256,128,64], nb_channels=1, scope_name="generator_b", use_sn=False, use_bn=False, use_wn=False, use_ln=False, use_in=False, use_bias=False,
            use_lrn=False, h=activations.relu(), o=None, default_reuse=False, default_train=True, conditional=False, output_dims=100):
    
    def network(X, y=None, reuse=default_reuse, train=default_train, ct=False, dropout=False, scope_name="discriminator"):
        
        with tf.variable_scope(scope_name, reuse) as scope:
    
            if reuse:
                scope.reuse_variables()
                
            net = layers.conv2D(X, 3, hiddens_dims[0], 2, use_bias=use_bias, name='conv0_1', lr_normed=use_lrn, weight_normed=use_wn,
                                       batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, spectral_normed=use_sn, layer_normed=use_ln, is_training=train)
            net = h(net)
            if dropout:
                net = tf.nn.dropout(net, keep_prob=keep_prob) 
            for i in range(1,len(hiddens_dims)-1):
                net = layers.conv2D(net, 3, hiddens_dims[i], 2, use_bias=use_bias, name='conv'+str(i)+'_2', lr_normed=use_lrn, weight_normed=use_wn,
                                           batch_normed=use_bn, batch_renormed=use_brn, instance_normed=use_in, layer_normed=use_ln, spectral_normed=use_sn, is_training=train)
                net = h(net)
                if dropout:
                    net = tf.nn.dropout(net, keep_prob=keep_prob)
            
            net_ = Flatten()(net)
            
            if y is not None:
                net_ = tf.concat([net_, y],axis=1)
            
            net = layers.linear(net_, output_dims, name='dense_layer', spectral_normed=use_sn, use_bias=use_bias, is_training=train)
            
            if ct:
                return net, net_
            
            return net
        
    def build_net(X, y=None, z=None, name='generator',reuse=False, train=True):
        
        if conditional:
            batch_data = tf.concat([z, y],axis=1)
            res = network(batch_data, reuse=reuse, train=train)
            return res
        else:
            res = network(X, reuse=reuse, train=train)
            return res 

    return build_net