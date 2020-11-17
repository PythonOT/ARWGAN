#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:55:28 2019

@author: vicari
"""
import tensorflow as tf
import numpy as np

def gp_loss(Xr, Xf, D, shape, labels=None):  
    """
    apply gradient penalty the discriminator loss
    """
    shape[0] = -1
    rand = list(np.ones(len(shape)))
    rand[0] = tf.shape(Xr)[0]
    epsilon = tf.random_uniform(shape=rand, minval=0., maxval=1.)
    X_hat = epsilon*Xr + (1.0 - epsilon)*Xf
    if labels is None:
        D_X_hat= D(X_hat, reuse=False)
    else:
        D_X_hat= D(X_hat, labels, reuse=False)
        
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    L2_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_D_X_hat), axis=np.arange(1, len(grad_D_X_hat.shape)))) # Norme euclidienne / norme L2
    gp = tf.reduce_mean((L2_norm - 1.0)**2)
    return gp
    
def lp_loss(Xr, Xf, D, shape, sampling='classic', labels=None):
    """
    apply lipschitz penalty the discriminator loss
    """
    
    shape[0] = -1
    
    if sampling == 'dragan_both':
        Xr = tf.reshape(Xr, shape)
        Xf = tf.reshape(Xf, shape)
        
        samples = tf.concat((Xr, Xf), axis=0)
        
        rand = list(np.ones(len(shape)))
        rand[0] = tf.shape(samples)[0]
        u = tf.random_uniform(shape=rand, minval=0., maxval=1.)
        _, batch_std = tf.nn.moments(tf.reshape(samples, [-1]), axes=[0])
        
        delta = 0.5 * batch_std * u
        
        alpha = tf.random_uniform(shape=rand, minval=0., maxval=1.)
        X_hat = samples + (1 - alpha) * delta
        
        if labels is None:
            D_X_hat= D(X_hat, reuse=False)
        else:
            D_X_hat= D(X_hat, labels, reuse=False)
            
        gradients = tf.gradients(D_X_hat, [X_hat])[0]
        gradients_sqr = tf.square(gradients)
        slopes = tf.sqrt(tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))))
        gp = tf.reduce_mean((tf.maximum(0., slopes - 1)) ** 2)
        
        return gp

    elif sampling == 'classic' :
        rand = list(np.ones(len(shape)))
        rand[0] = tf.shape(Xr)[0]
        epsilon = tf.random_uniform(shape=rand, minval=0., maxval=1.)
        X_hat = epsilon*Xr + (1.0 - epsilon)*Xf
        if labels is None:
            D_X_hat= D(X_hat, reuse=False)
        else:
            D_X_hat= D(X_hat, labels, reuse=False)
            
        grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
        L2_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_D_X_hat), axis=np.arange(1, len(grad_D_X_hat.shape)))) # Norme euclidienne / norme L2
        gp = tf.reduce_mean(tf.maximum(0., L2_norm - 1)**2)
        return gp
        
def ct_loss(Xr, Xf, D, shape, labels=None):
    
    if labels is not None:
        x = D(Xr, labels, dropout=True, reuse=False)
        x_ = D(Xr, labels, dropout=True, reuse=False)
    else:
        x = D(Xr, dropout=True, reuse=False) 
        x_ = D(Xr, dropout=True, reuse=False) 
        
    CT = tf.square(x-x_)
    CT_ = tf.maximum(CT,0.0)
    return tf.reduce_mean(CT_)