#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:40:03 2019

@author: vicari
"""

# Imports
#%%
import matplotlib.pyplot as plt
import numpy as np 

from ipywidgets import FloatProgress
from IPython.display import display

import random
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K
from keras.backend.tensorflow_backend import set_session
import os
import math
import scipy.misc
from sklearn.utils import shuffle

from AdvGAN import layers, utils, print_functions, dataset, models, activations, losses, classifiers

import configparser

# Test block, will be removed
#%%
print('AdvGAN library')

# Classes
#%%     
        

###########################
#
# Class AdvGAN 
#
###########################  
class CycleWGAN:
    
    def set_name(self, name):
        self.name = name
        
    def set_generator_s(self, gen):
        self.generator_t2s = gen
        
    def set_discriminator_s(self, disc):
        self.discriminator_s = disc
        
    def set_generator_t(self, gen):
        self.generator_s2t = gen
        
    def set_discriminator_t(self, disc):
        self.discriminator_t = disc
        
    def set_batch_size(self, bs=None, mini_batch_max=None):       
        if bs is None:
            bs = self.batch_size 
        if (mini_batch_max is not None) and (mini_batch_max < bs):
            self.batch_size = bs
            self.accumulate_grd = True
            self.batch_size = mini_batch_max
            self.n_minibatches = int(bs / mini_batch_max)      
        else:
            self.batch_size = bs
            self.accumulate_grd = False
            self.n_minibatches = 1

    def set_loss(self, loss):
        possible_losses = ['gp', 'lp', 'clip', 'sn', 'ct']
        if loss['name'].lower() in possible_losses:
            self.loss = loss['name'].lower()
            
            if loss['name'].lower() in ['gp', 'lp', 'ct']:
                self.gp_lambda = float(loss['gp_lambda'])
                
            if loss['name'].lower() in ['ct']:
                self.ct_lambda = float(loss['ct_lambda'])
                
            if loss['name'].lower() in ['clip']:
                self.clip_value = float(loss['clip_value'])
        else:
            raise Exception('the loss {} is not valid'.format(loss))
            
    def set_optimizer_D(self, optimizer_D):
        if optimizer_D['name'].lower() == 'adam':
            
            if optimizer_D['learning_rate'] is None:
                optimizer_D['learning_rate'] = 1e-4
                
            if optimizer_D['beta1'] is None :
                optimizer_D['beta1'] = 0.9
                
            if optimizer_D['beta2'] is None :
                optimizer_D['beta2'] = 0.99
                
            self.optimizer_D = tf.compat.v1.train.AdamOptimizer(learning_rate=float(optimizer_D['learning_rate']), beta1=float(optimizer_D['beta1']), beta2=float(optimizer_D['beta2']))
            
        elif optimizer_D['name'].lower() == 'rmsprop':
            
            if optimizer_D['learning_rate'] is None:
                optimizer_D['learning_rate'] = 1e-4
                
            self.optimizer_D = tf.compat.v1.train.RMSPropOptimizer(learning_rate=float(optimizer_D['learning_rate']))
            
        else:
            raise Exception('the optimizer {} is not valid'.format(optimizer_D['name']))
            
    
    def set_optimizer_G(self, optimizer_G):
        if optimizer_G['name'].lower() == 'adam':
            
            if optimizer_G['learning_rate'] is None:
                optimizer_G['learning_rate'] = 1e-4
                
            if optimizer_G['beta1'] is None :
                optimizer_G['beta1'] = 0.9
                
            if optimizer_G['beta2'] is None :
                optimizer_G['beta2'] = 0.99
                
            self.optimizer_G = tf.compat.v1.train.AdamOptimizer(learning_rate=float(optimizer_G['learning_rate']), beta1=float(optimizer_G['beta1']), beta2=float(optimizer_G['beta2']))
            
        elif optimizer_G['name'].lower() == 'rmsprop':
            
            if optimizer_G['learning_rate'] is None:
                optimizer_G['learning_rate'] = 1e-4
                
            self.optimizer_G = tf.compat.v1.train.RMSPropOptimizer(learning_rate=float(optimizer_G['learning_rate']))
            
        else:
            raise Exception('the optimizer {} is not valid'.format(optimizer_G['name']))
            
    def set_dataset(self, data):
        
        if not isinstance(data, (dataset.Dataset)):
            raise Exception('the dataset must be a Dataset object')
        else:
            self.data = data
            
    def set_training_options(self, nb_iter=None, init_step=None, n_c_iters_start=None, n_c_iters=None):
        
        if init_step is not None:
            self.init_step = init_step
            
        if n_c_iters_start is not None:
            self.n_c_iters_start = n_c_iters_start
            
        if n_c_iters is not None:
            self.n_c_iters = n_c_iters
            
        if nb_iter is not None:
            self.nb_iter = nb_iter   
        
    def set_classifier_s(self, classifier=None, nb_class=None):
        
        if classifier is not None:
            self.classifier_s = classifier
        
        if nb_class is not None:
            self.nb_class = nb_class 
            
    def set_classifier_t(self, classifier=None, nb_class=None, classif_treshold=None, classif_iter=None):
        
        if classifier is not None:
            self.classifier_t = classifier
        
        if nb_class is not None:
            self.nb_class = nb_class 
         
        if classif_treshold is not None:
            self.classif_treshold = classif_treshold
            
        if classif_iter is not None:
            self.classif_iter = classif_iter
        
    def set_logs(self, print_method=None, frequency_print=None, frequency_logs=None, logs_path=None):
        
        if print_method is not None:
            self.print_method = print_method 
            
        if frequency_print is not None:
            self.frequency_print = frequency_print 
        
        if frequency_logs is not None:
            self.frequency_logs = frequency_logs
            
        if logs_path is not None:
            self.logs_path = logs_path 
            
    def set_test_preds(self, make_preds=None, data_test=None):
        
        if make_preds is not None:
            self.make_preds = make_preds 
        if data_test is not None:
            self.data_test = data_test 
        
    def set_uses(self, use_labels=None):

        if use_labels is not None:
            self.use_labels = use_labels

        
    def set_hypers(self, clf_s_alpha=None, clf_t_alpha=None, recon_beta=None):

        if clf_s_alpha is not None:
            self.clf_s_alpha = clf_s_alpha
        if clf_t_alpha is not None:
            self.clf_t_alpha = clf_t_alpha
        if recon_beta is not None:
            self.recon_beta = recon_beta
        
    def __init__(self):                      
        
        self.generator_s2t = models.make_g_conv(
                                    img_size=32,
                                    hiddens_dims=[256,128,64],
                                    nb_channels=1,
                                    scope_name="generator_s2t",
                                    use_sn=False,
                                    use_bn=True,
                                    h=tf.nn.relu,
                                    o=tf.nn.tanh,
                                    use_wn=False,
                                    use_in=False
                                    )
        
        self.generator_t2s = models.make_cycle_g_t_conv(
                                    hiddens_dims=[128,128,128],
                                    h=tf.nn.relu,
                                    scope_name="generator_t2s",
                                    o=tf.nn.tanh
                                    )
        
        self.discriminator_s = models.make_fc(
                                hiddens_dims=[128,128,128],
                                ouput_dims=1,
                                scope_name="discriminator_s",
                                h=activations.lrelu(0.2),
                                use_sn = False,
                                use_bias = False,
                                default_reuse = True
                                )   
        
        self.discriminator_t = models.make_d_conv(
                                hiddens_dims=[64,128,256],
                                scope_name="discriminator_t",
                                h=activations.lrelu(0.2),
                                use_sn=False
                                )
 
        self.accumulate_grd = False
        self.n_minibatches = 1
        self.batch_size = 64   
        self.name = "CycleGAN"
        
        self.use_adaptative_adv = False

        self.z_dim = [16]      
        self.gp_lambda = 10  
        self.ct_lambda = 2
        
        self.use_MA = True
        self.ma_gamma = 0.01
        
        self.one_way = False
        
        self.optimizer_D = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
        self.optimizer_G = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
        
        self.optimizer_C = tf.compat.v1.train.RMSPropOptimizer(learning_rate=5e-5)
        
        
        self.init_step = 0
        self.n_c_iters_start = 1
        self.n_c_iters = 5
        self.nb_iter = 5000
        
        self.make_preds = False
        self.data_test = None
        
        
        self.data = None
        self.classifier_s = None
        self.classifier_t = classifiers.make_simpler_conv_net(8)
        
        self.nb_classes = 10
        
        self.print_method = print_functions.show_it_label('advGAN')
        self.frequency_print = 100     
        self.frequency_logs = None
        self.logs_path = './logs/'
        
        self.loss = 'lp'
        
        self.temperature = 20
        self.clip_value = 0.1
        
        self.adv = False      
        self.use_proba = True
        self.use_ceil = False    
        self.use_labels = False
        
        
        self.g_loss = None
        self.d_loss  = None
        
        self.builded = False
        
    def noise(self, batchsize, z_size):
        """
        Function to generate gaussian noise

        Parameters
        ----------
        batchsize: int
        z_size: int
        
        Returns
        -------
        np.array
            array of noise with the shape [batchsize, z_size]
        """
        '''
        size = [batchsize] + z_size
        return np.random.normal(-1., 1., size=size)
        '''
        size = [batchsize] + z_size
        rand = np.ones(len(size), dtype=np.int)
        rand[-1] = size[-1]
        rand[0] = size[0]
        ones = np.ones(size, dtype=np.float)
        return ones*np.random.normal(-1., 1., size=rand)
    
                
    def build_model(self, reset_graph=False):
        """
        Function to build the graph use for AdvGAN
        """
    
        if self.data is None:
            raise Exception('a dataset must be specified')
            
        if (self.adv and (self.classifier_s is None)):
            raise Exception('A classifier must be specified to train in adversarial mode')
            
        if(reset_graph):
            tf.compat.v1.reset_default_graph()
            
        self.dataset_is_tf  = self.data.is_tf
        
        if self.classifier_s is not None:
            self.classifier_s.load_model()
            
        shape_s, shape_t, shape_s_y, shape_t_y = self.data.shape()
        
        shape_s[0] = None
        shape_t[0] = None
        shape_s_y[0] = None
        shape_t_y[0] = None
        
        if self.dataset_is_tf:
            self.X_s , self.X_t, self.y_s, self.y_t, self.mask_t_y = self.data.next_batch()    
            
            self.X_s.set_shape(shape_s)
            self.X_t.set_shape(shape_t)
            self.y_s.set_shape(shape_s_y)
            self.y_t.set_shape(shape_t_y)
            
        else:                  
            self.X_s = tf.compat.v1.placeholder("float", shape_s, name="X_s")
            self.X_t = tf.compat.v1.placeholder("float", shape_t, name="X_t")
            
            if self.use_labels:   
                self.y_s = tf.compat.v1.placeholder("float", shape_s_y, name="y_s")
        
        if self.use_labels:            
            if not self.one_way:   
                self.use_Classifs = tf.compat.v1.placeholder(tf.bool, shape=())                
                #self.y_t = tf.placeholder("float", shape_t_y, name="y_t")
            
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        # Discriminate Real
        if self.use_labels and self.one_way:      
            self.pred_real_s = self.discriminator_s(self.X_s, self.y_s, reuse=False)   
            self.pred_real_t = self.discriminator_t(self.X_t, reuse=False)
        else:
            self.pred_real_t = self.discriminator_t(self.X_t, reuse=False)           
            self.pred_real_s = self.discriminator_s(self.X_s, reuse=False)          
        
        # source -> target-ish
        if self.use_labels and self.one_way:      
            self.X_fake_t = self.generator_s2t(self.X_s, self.y_s, train=self.is_training)
            self.X_fake_s, self.y_fake_s = self.generator_t2s(self.X_t, train=self.is_training)
        else:
            self.X_fake_t = self.generator_s2t(self.X_s, train=self.is_training)      
            self.X_fake_s = self.generator_t2s(self.X_t, train=self.is_training)
        
        # target-ish -> source-ish
        if self.use_labels and self.one_way:      
            self.X_fake_s_, self.y_fake_s_ = self.generator_t2s(self.X_fake_t, train=self.is_training, reuse=True)
            self.X_fake_t_ = self.generator_s2t(self.X_fake_s, self.y_fake_s, train=self.is_training, reuse=True)
        else:
            self.X_fake_s_ = self.generator_t2s(self.X_fake_t, train=self.is_training, reuse=True)  
            self.X_fake_t_ = self.generator_s2t(self.X_fake_s, train=self.is_training, reuse=True)

        # Discriminate target-ish
        if self.use_labels and self.one_way:      
             self.pred_fake_s = self.discriminator_s(self.X_fake_s, self.y_fake_s)
             self.pred_fake_t = self.discriminator_t(self.X_fake_t)
        else:
            self.pred_fake_t = self.discriminator_t(self.X_fake_t)
            self.pred_fake_s = self.discriminator_s(self.X_fake_s)       
        
        
        # Reconstruction loss
        if self.use_labels and self.one_way:
            reconstruction_loss = self.recon_beta  * ( tf.reduce_mean(input_tensor=tf.abs(self.X_fake_s_ - self.X_s)) + tf.reduce_mean(input_tensor=tf.abs(self.X_fake_t_ - self.X_t)) )
            self.g_loss_s2t = - tf.reduce_mean(input_tensor=self.pred_fake_t) + reconstruction_loss/2 
            self.g_loss_t2s = - tf.reduce_mean(input_tensor=self.pred_fake_s) + reconstruction_loss/2         
            self.g_loss_t2s += tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=tf.one_hot(tf.argmax(input=self.y_s, axis=1), shape_s_y[-1]) * tf.math.log(tf.clip_by_value(self.y_fake_s ,1e-10,1.0)), axis=[1]))
            
        else:
            reconstruction_loss = self.recon_beta  * ( tf.reduce_mean(input_tensor=tf.abs(self.X_fake_s_ - self.X_s)) + tf.reduce_mean(input_tensor=tf.abs(self.X_fake_t_ - self.X_t)) )
            self.g_loss_s2t = - tf.reduce_mean(input_tensor=self.pred_fake_t) + reconstruction_loss/2 
            self.g_loss_t2s = - tf.reduce_mean(input_tensor=self.pred_fake_s) + reconstruction_loss/2
        
        #Use classifier
        if self.use_labels and not self.one_way:
            self.classif_real_s = self.classifier_s.model(self.X_s)
            self.classif_fake_s = self.classifier_s.model(self.X_fake_s)
            
            self.classif_real_t = self.classifier_t(self.X_t)
            self.classif_fake_t = self.classifier_t(self.X_fake_t)
   
            self.loss_classif_s = self.clf_s_alpha*((1/tf.maximum(1.0,tf.reduce_sum(input_tensor=self.mask_t_y)))*tf.reduce_sum(input_tensor=-tf.reduce_sum(input_tensor=(self.mask_t_y*self.y_t) * tf.math.log(tf.clip_by_value(self.classif_fake_s, 1e-10, 1.0)), axis=[1])))
            self.loss_classif_t = self.clf_t_alpha*tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=self.y_s * tf.math.log(tf.clip_by_value(self.classif_fake_t ,1e-10,1.0)), axis=[1]))
            
            self.g_loss_t2s += self.loss_classif_s
            self.g_loss_s2t += self.loss_classif_t
            
        if self.use_MA :
            ma = 0
            for i in range(self.batch_size-1):
                for j in range(i+1, self.batch_size):
                    ma += (tf.abs(tf.reduce_sum(input_tensor=tf.square(self.X_s[i] - self.X_s[j]), axis=np.arange(0, len(self.X_s.shape)-1)) - tf.reduce_sum(input_tensor=tf.square(self.X_fake_t[i] - self.X_fake_t[j]), axis=np.arange(0, len(self.X_fake_t.shape)-1)))) / 2
                    ma += (tf.abs(tf.reduce_sum(input_tensor=tf.square(self.X_t[i] - self.X_t[j]), axis=np.arange(0, len(self.X_t.shape)-1)) - tf.reduce_sum(input_tensor=tf.square(self.X_fake_s[i] - self.X_fake_s[j]), axis=np.arange(0, len(self.X_fake_s.shape)-1)))) / 2
                    
            self.g_loss_s2t += self.ma_gamma*tf.reduce_mean(input_tensor=ma)/2
            self.g_loss_t2s += self.ma_gamma*tf.reduce_mean(input_tensor=ma)/2
            
        
        self.d_loss_t = - (tf.reduce_mean(input_tensor=self.pred_real_t) - tf.reduce_mean(input_tensor=self.pred_fake_t))
        self.d_loss_s = - (tf.reduce_mean(input_tensor=self.pred_real_s) - tf.reduce_mean(input_tensor=self.pred_fake_s))
        
        self.neg_loss_t = - self.d_loss_t
        self.neg_loss_s = - self.d_loss_s
        
        if(self.use_labels and self.one_way):
            labels = self.y_s
        else:
            labels = None
            
        if self.loss == 'gp' :
            self.d_loss_s += self.gp_lambda * losses.gp_loss(self.X_s, self.X_fake_s, self.discriminator_s, shape_s, labels=labels)
            self.d_loss_t += self.gp_lambda * losses.gp_loss(self.X_t, self.X_fake_t, self.discriminator_t, shape_t, labels=None)   
        if self.loss == 'lp' :
            self.d_loss_s += self.gp_lambda * losses.lp_loss(self.X_s, self.X_fake_s, self.discriminator_s, shape_s, labels=labels)
            self.d_loss_t += self.gp_lambda * losses.lp_loss(self.X_t, self.X_fake_t, self.discriminator_t, shape_t, labels=None)        
        if self.loss ==  'ct':
            self.d_loss_s += self.gp_lambda * losses.lp_loss(self.X_s, self.X_fake_s, self.discriminator_s, shape_s, labels=labels)
            self.d_loss_t += self.gp_lambda * losses.lp_loss(self.X_t, self.X_fake_t, self.discriminator_t, shape_t, labels=None)
            
            self.d_loss_s += self.ct_lambda * losses.ct_loss(self.X_s, self.X_fake_s, self.discriminator_s, shape_s, labels=labels)
            self.d_loss_t += self.ct_lambda * losses.ct_loss(self.X_t, self.X_fake_t, self.discriminator_t, shape_t, labels=None)
                
        self.d_loss = self.d_loss_s + self.d_loss_t
        self.g_loss = self.g_loss_s2t + self.g_loss_t2s
        self.neg_loss = self.neg_loss_s + self.neg_loss_t
        
        D_vars = [var for var in tf.compat.v1.trainable_variables() if "discriminator" in var.name]
        G_vars = [var for var in tf.compat.v1.trainable_variables() if "generator" in var.name]  
        
        if self.use_labels and not self.one_way:
            C_var = [var for var in tf.compat.v1.trainable_variables() if "classifier" in var.name]
        
        if self.accumulate_grd:
            # initialized with 0s
            
            self.accum_vars_D = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False ) for tv in D_vars]         
            self.accumulation_counter_D = tf.Variable(0.0, trainable=False)
            
            self.accum_vars_G = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in G_vars]         
            self.accumulation_counter_G = tf.Variable(0.0, trainable=False)
            
        self.D_sum = tf.compat.v1.summary.scalar('D_loss', self.d_loss)
        self.G_sum = tf.compat.v1.summary.scalar('G_loss', self.g_loss)

        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            
            if self.accumulate_grd:

                gvs_G = (self.optimizer_G).compute_gradients(self.g_loss, G_vars)
                gvs_D = (self.optimizer_D).compute_gradients(self.d_loss, D_vars)
                
                self.accum_ops_G = [accumulator.assign_add(grad) for (accumulator, (grad, var)) in zip(self.accum_vars_G, gvs_G)] 
                self.accum_ops_G.append(self.accumulation_counter_G.assign_add(1.0))            
                
                self.accum_ops_D = [accumulator.assign_add(grad) for (accumulator, (grad, var)) in zip(self.accum_vars_D, gvs_D)]  
                self.accum_ops_D.append(self.accumulation_counter_D.assign_add(1.0))    
                
                self.G_step = (self.optimizer_G).apply_gradients([(accumulator / self.accumulation_counter_G, var) for (accumulator, (grad, var)) in zip(self.accum_vars_G, gvs_G)])

                self.D_step = (self.optimizer_D).apply_gradients([(accumulator / self.accumulation_counter_D, var) for (accumulator, (grad, var)) in zip(self.accum_vars_D, gvs_D)])
                
                self.zero_ops_G = [accumulator.assign(tf.zeros_like(tv)) for (accumulator, tv) in zip(self.accum_vars_G, G_vars)]
                self.zero_ops_D = [accumulator.assign(tf.zeros_like(tv)) for (accumulator, tv) in zip(self.accum_vars_D, D_vars)]
            
                self.zero_ops_G.append(self.accumulation_counter_G.assign(0.0))
                self.zero_ops_D.append(self.accumulation_counter_D.assign(0.0))
            
            else:
                self.G_step = (self.optimizer_G).minimize(self.g_loss, var_list=G_vars)
                self.D_step = (self.optimizer_D).minimize(self.d_loss, var_list=D_vars)
                if self.use_labels and not self.one_way:
                    self.C_step = (self.optimizer_C).minimize(self.loss_classif_t, var_list=C_var)
                
            if self.loss == 'clip' :
                self.D_clipping = [w.assign(tf.clip_by_value(w, -self.clip_value, self.clip_value)) for w in D_vars]

        
        self.saver = tf.compat.v1.train.Saver()
        self.builded = True
            
    def train(self, init_step=None, n_c_iters_start=None, n_c_iters=None, nb_iter=None, print_method=None, 
                  frequency_print = None, frequency_logs=None, use_ceil=None, restore=False, restore_meta=False, data_pretrain=None):
        """
        Function to train AdvGAN

        Parameters
        ----------
        init_step: int
            number of step for the initialisation phase
        n_c_iters_start: int
            number of discriminator update in the initialisation phase
        n_c_iters: int
            ratio of discriminator update per generator update
        nb_iter: int
            number of generation for the training
        print_method: function
            function call every X generation
        frequency_print: int
            the number of generation beetwen each print_method d
        frequency_logs: int
            the number of generation beetwen each log update
        use_ceil: boolean
                ceil the probabilty depending of the amount of adversarial data in the batch
        """
        if not self.builded and not restore_meta :
            raise Exception('Build the model first or use the restore_meta option')
        
        if init_step is not None:
            self.init_step = init_step
        if n_c_iters_start is not None:
            self.n_c_iters_start = n_c_iters_start
        if n_c_iters is not None:
            self.n_c_iters = n_c_iters
        if nb_iter is not None:
            self.nb_iter = nb_iter
        if print_method is not None:
            self.print_method = print_method
        if frequency_print is not None:
            self.frequency_print = frequency_print
        if frequency_logs is not None:
            self.frequency_logs = frequency_logs
        if use_ceil is not None:
            self.use_ceil = use_ceil
            
        self.critic_losses = []
        self.neg_losses = []
        self.generator_losses = []
        self.clf_losses = []
        self.acc_t_logs =[]
        if(self.make_preds):
            self.clf_s_test_losses = []
            self.clf_t_test_losses = []
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.compat.v1.Session(config=config) as sess:
            set_session(sess)
            sess.run(tf.compat.v1.global_variables_initializer())
            it = 0
            if(restore):
                if restore_meta:
                    self.saver = tf.compat.v1.train.import_meta_graph("data/"+self.name+'.ckpt.meta')
                exists = os.path.isfile("data/"+self.name+'.ckpt.index')
                if exists:
                    self.saver.restore(sess, "data/"+self.name+'.ckpt')
                    np.save('logs', [self.critic_losses, self.neg_losses, self.generator_losses, self.clf_losses])
                else:
                    print('no weights found, training with new weights')
                exists = os.path.isfile("data/logs.npy")
                if exists:
                    loader = np.load('data/logs.npy')
                    self.critic_losses = loader[0]
                    self.neg_losses = loader[1]
                    self.generator_losses = loader[2]
                    self.clf_losses = loader[3]
                else:
                    print('BEWARE !!!! no logs found, creating new logs')
                exists = os.path.isfile("data/"+self.name+'.saved')
                if exists:
                    config = configparser.ConfigParser()
                    config.read("data/"+self.name+'.saved')
                    ckpt = config['ckpt']
                    it = ckpt.getint('it') 
                    print("Model load at it=", it)
                    it=it+1
                
            self.f_prog = FloatProgress(min=it, max=self.nb_iter)
            display(self.f_prog)
                
            if self.classifier_s is not None:
                self.classifier_s.load_weights()
            
            #GAN_vars = [var for var in tf.global_variables() if "GAN" in var.name]
            #sess.run(tf.variables_initializer(GAN_vars))
            
            d_sum, g_sum = None, None
            self.writer = tf.summary.FileWriter(self.logs_path, graph=tf.compat.v1.get_default_graph())


            ######################
            
            Data_X = tf.compat.v1.placeholder("float", (None, 64, 64, 13), name="Data_X")  
            Data_y = tf.compat.v1.placeholder("float", (None, 8), name="Data_y")
            
            y_pred = self.classifier_t(Data_X)
            
            C_var = [var for var in tf.compat.v1.trainable_variables() if "classifier" in var.name]
            
            ce = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=Data_y * tf.math.log(tf.clip_by_value(y_pred ,1e-10,1.0)), axis=[1]))
            
            step = (self.optimizer_C).minimize(ce, var_list=C_var)      
            
            acc = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.cast(tf.argmax(input=Data_y, axis=1), tf.float32), tf.cast(tf.argmax(input=y_pred, axis=1), tf.float32)), tf.float32))

            
            data_pretrain.shuffle()
            shape_x = data_pretrain.shape()[0]
            for e in range(10):   
                for i in range(0, shape_x[0], self.batch_size):
                    X, y = data_pretrain.next_batch(self.batch_size)
                    feed_dict_batch = {Data_X: X, Data_y: y}
                    _, a = sess.run([step, acc], feed_dict=feed_dict_batch)
                print(a)
                   
            
            
            ##################################################
            # Train phase
            #####
            if self.n_c_iters_start != None:
                n_c_iters = self.n_c_iters_start
            
            for self.it in range(it,self.nb_iter+1):
                if self.it > self.init_step :
                    n_c_iters = self.n_c_iters
                    
                #loss_critic = [] # To be sure there is convergence
                
                ####################################################################################################################
                # Train the discriminator
                #####         
                for _ in range(n_c_iters):                   

                    # Classic GAN
                    if self.dataset_is_tf:
                        fd = {self.is_training : True}  
                    else:
                        X_s, X_t = self.data.next_batch(self.batch_size)
                        fd = {self.X_s: X_s, self.X_t: X_t,
                                   self.is_training : True}
                    
                    _, d_loss, neg_loss = sess.run(
                                [self.D_step, self.d_loss, self.neg_loss],
                                feed_dict=fd) 
 
                ####################################################################################################################
                # Train the generator
                ##### 
                
                if self.use_labels and not self.one_way:
                    if (self.it > self.classif_treshold+1):
                        use_clf = True
                    else:                     
                        use_clf = False   
                        
                    if self.dataset_is_tf:
                        fd = {self.is_training : True, self.use_Classifs: use_clf}
                    else:
                        X_s, X_t, y_s = self.data.next_batch(self.batch_size, with_labels=True)
                        fd = {self.X_s: X_s, self.X_t: X_t, self.y_s : y_s, self.is_training : True, self.use_Classifs: use_clf}
                        
                elif self.use_labels and self.one_way:
                    if self.dataset_is_tf:
                        fd = {self.is_training : True}
                    else:
                        X_s, X_t, y_s = self.data.next_batch(self.batch_size, with_labels=True)
                        fd = {self.X_s: X_s, self.X_t: X_t, self.y_s : y_s, self.is_training : True}
                        
                else:
                    if self.dataset_is_tf:
                        fd = {self.is_training : True}
                    else:
                        X_s, X_t = self.data.next_batch(self.batch_size)
                        fd = {self.X_s: X_s, self.X_t: X_t, self.is_training : True}
                    
                if not self.one_way:
                    _, g_loss = sess.run(
                        [self.G_step, self.g_loss],
                        feed_dict=fd)
                else:
                    _, g_loss = sess.run(
                        [self.G_step, self.g_loss],
                        feed_dict=fd)
                # Train Classifier
 
                acc_t = 0
                clf_loss = 0
                '''
                if (self.use_labels and not self.one_way and (self.it > self.classif_treshold)):
                    for _ in range(self.classif_iter):
                        
                        if self.dataset_is_tf:
                            fd = {self.is_training : True, self.use_Classifs: True}
                        else:
                            X_s, X_t, y_s = self.data.next_batch(self.batch_size, with_labels=True)            
                            fd = {self.X_s: X_s, self.X_t: X_t, self.y_s : y_s, self.is_training : True, self.use_Classifs: True}
                            
                        _, c_loss, acc_t = sess.run(
                            [self.C_step, self.loss_classif_t, self.accuracy_target],
                            feed_dict=fd)
                '''       
                    
                #TODO :
                '''
                    Classif s entre g_t2s et y_t
                    Classif t entre g_s2t et y_s
                '''
                
                if(self.make_preds and use_clf and not self.one_way):
                    if self.data_test is not None:
                        X_s, y_s, X_t, y_t = self.data_test.next_batch(self.batch_size)
                
                        fd = {self.X_s: X_s, self.X_t: X_t, self.y_s : y_s, self.is_training : False}
                        
                        classif_fake_s, classif_real_t = sess.run(
                            [self.classif_fake_s, self.classif_real_t],
                            feed_dict=fd)
                        
                        self.ce_s = np.mean(-np.sum(y_s * np.log(np.clip(classif_fake_s ,1e-10, 1.0-1e-10)), axis=1))
                        
                        self.ce_t = np.mean(-np.sum(y_t * np.log(np.clip(classif_real_t ,1e-10, 1.0-1e-10)), axis=1))
                        
                        self.clf_s_test_losses.append(self.ce_s)
                        self.clf_t_test_losses.append(self.ce_t)

                
                ####################################################################################################################
                # Logscd @Claire Voreiter 
                #####
                    
                
                self.critic_losses.append(d_loss)
                self.neg_losses.append(neg_loss)
                self.generator_losses.append(g_loss)
                if not self.one_way:
                    self.clf_losses.append(clf_loss)
                self.acc_t_logs.append(acc_t)
                self.f_prog.value += 1

                ####################################################################################################################
                # print_method and save
                #####
                if self.it % self.frequency_print == 0:
                    if self.print_method != None:
                        if self.dataset_is_tf:
                            fd = {self.is_training : False}
                        else:
                            X_s, X_t = self.data.next_batch(self.batch_size)                   
                            fd = {self.X_t: X_t, self.X_s: X_s, self.is_training : False}
                            
                        X_s, X_t, fake_xt, fake_xs, reconstruct_xs, reconstruct_xt = sess.run(
                                [self.X_s, self.X_t, self.X_fake_t, self.X_fake_s, self.X_fake_s_, self.X_fake_t_],
                                feed_dict=fd)

                        self.print_method(X_s, X_t, fake_xt, fake_xs, reconstruct_xs, reconstruct_xt, gan=self)
                    # To save the model 
                    w_meta_graphe = False
                    if self.it == 0:
                        w_meta_graphe = True
                    keep = 1
                    if keep > 1:
                        self.saver.save(sess, "data/"+self.name+".ckpt", write_meta_graph=w_meta_graphe, global_step=self.it, max_to_keep=keep)
                    else:
                        self.saver.save(sess, "data/"+self.name+".ckpt", write_meta_graph=w_meta_graphe)
   
                    config = configparser.ConfigParser()
                    config['ckpt'] = {'it': str(self.it)}
                    with open("data/"+self.name+".saved", 'w') as cfg:
                        config.write(cfg)

                    np.save('data/logs.npy', [self.critic_losses, self.neg_losses, self.generator_losses, self.clf_losses, self.acc_t_logs])
                                        
   
    #############################################
    
    def load_and_generate(self, path, batch_size=None, Xs=None, Xt=None):
        """
        Function to generate a batch of data
    
        Parameters
        ----------
        path: string
                path to file ckpt from ./data, without the extension
                
        noise: array of shape [batch, z_dim]
            The noise for the generator, optional
        
        Returns
        -------
        array
            batch of generated data
        """
        nb = self.batch_size
        with tf.compat.v1.Session() as sess:
            self.saver.restore(sess, "data/"+path+".ckpt")
            if batch_size != None:
                nb = batch_size
                
            res_Xs = None
            res_Xt = None
            
            if Xs is not None:
                res_Xs = self.X_fake_t.eval(feed_dict={self.X_s: Xs, self.is_training : False})
                
            if Xt is not None:
                res_Xt = self.X_fake_s.eval(feed_dict={self.X_t: Xt, self.is_training : False})
                
            return res_Xs, res_Xt
        
    def load_and_classify(self, path, batch_size=None, Xs=None, Xt=None):
        nb = self.batch_size
        with tf.compat.v1.Session() as sess:
            self.saver.restore(sess, "data/"+path+".ckpt")
            if batch_size != None:
                nb = batch_size
                
            res_Xs = None
            res_Xt = None
            
            if Xs is not None:
                res_clf_s = self.classif_real_s.eval(feed_dict={self.X_s: Xs, self.is_training : False})
                res_clf_t = self.classif_fake_t.eval(feed_dict={self.X_s: Xs, self.is_training : False})

                
            if Xt is not None:
                res_clf_t = self.classif_real_t.eval(feed_dict={self.X_t: Xt, self.is_training : False})
                res_clf_s = self.classif_fake_s.eval(feed_dict={self.X_t: Xt, self.is_training : False})

                
            return res_clf_s, res_clf_t
    #############################################
       
    def load_and_discrimininate(self, path, data):
        with tf.compat.v1.Session() as sess:
            self.saver.restore(sess, "data/"+path+".ckpt")
            return self.pred_fake.eval(feed_dict={self.X_fake: data})
    
    #############################################
    
    def save_loss(self, path='./loss'):
        """
        Function to save the critic loss
        """
        np.save(path, self.critic_losses)
        
    def estimate(self):
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
 
##########################################################################################
##########################################################################################
##########################################################################################
        
def make_gan(file):
    config = configparser.ConfigParser()
    config.read(file)
    
    gan = CycleWGAN()
    
    # MISC
    misc = config['MISC']
    
    gan.set_name(misc.get('name'))
    gan.set_batch_size(misc.getint('batch_size'))
    gan.set_batch_size(misc.getint('mini_batch_max'))
    
    # USES
    uses = config['USES']
    gan.set_uses(uses.getboolean('use_labels')) 
        
    # REGULARIZATION
    reg = dict(config['REGULARIZATION'])
    
    if len(reg.keys()) > 0:
        gan.set_loss(reg)
    
    # OPTIMIZERS
    opti_D = dict(config['D_OPTIMIZER'])
    
    if len(opti_D.keys()) > 0:
        gan.set_optimizer_D(opti_D)
        
    opti_G = dict(config['G_OPTIMIZER'])
    
    if len(opti_G.keys()) > 0:
        gan.set_optimizer_G(opti_G)
    
    # TRAIN
    train = config['TRAIN']
    
    nb_iter = train.getint('nb_iter')
    init_step = train.getint('init_step')
    n_c_iters_start = train.getint('n_c_iters_start')
    n_c_iter = train.getint('n_c_iter')
        
    gan.set_training_options(nb_iter, init_step, n_c_iters_start, n_c_iter)
    
    # LOGS 
    log = config['LOGS']
        
    print_method = log.get('print_method')
    
    if print_method == 'show_it_label':
        print_method = print_functions.show_it_label(log.get('label', 'label'))
        
        
    frequency_print = log.getint('frequency_print')
    frequency_logs = log.getint('frequency_logs')
    logs_path = log.get('logs_path')  
        
    gan.set_logs(print_method, frequency_print, frequency_logs, logs_path)
    
    #Other variable
    hypers = config['HYPERS']
    
    clf_s_alpha = hypers.getfloat('clf_s_alpha')
    clf_t_alpha = hypers.getfloat('clf_t_alpha')
    recon_beta = hypers.getfloat('recon_beta')
    classif_treshold = hypers.getint('classif_treshold')
    classif_iter = hypers.getint('classif_iter')
    
    gan.set_hypers(clf_s_alpha, clf_t_alpha, recon_beta)
    gan.set_classifier_t(classif_treshold=classif_treshold, classif_iter=classif_iter)
        
    return gan