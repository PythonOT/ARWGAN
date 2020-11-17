#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:55:28 2019

@author: vicari
"""
import numpy as np
import tensorflow as tf


def gp_loss(xr, xf, d, shape, labels=None):

    """
    apply gradient penalty the discriminator loss
    """
    shape[0] = -1
    rand = list(np.ones(len(shape)))
    rand[0] = tf.shape(input=xr)[0]
    epsilon = tf.random.uniform(shape=rand, minval=0., maxval=1.)
    x_hat = epsilon * xr + (1.0 - epsilon) * xf
    if labels is None:
        d_x_hat = d(x_hat, reuse=False)
    else:
        d_x_hat = d(x_hat, labels, reuse=False)

    grad_d_x_hat = tf.gradients(d_x_hat, [x_hat])[0]
    l2_norm = tf.sqrt(1e-8 + tf.reduce_sum(input_tensor=tf.square(grad_d_x_hat),
                                           axis=np.arange(1, len(grad_d_x_hat.shape))))  # Norme euclidienne / norme L2
    gp = tf.reduce_mean(input_tensor=(l2_norm - 1.0) ** 2)
    return gp


def lp_loss(xr, xf, d, shape, sampling='classic', labels=None):
    """
    apply lipschitz penalty the discriminator loss
    """

    shape[0] = -1

    if sampling == 'dragan_both':
        xr = tf.reshape(xr, shape)
        xf = tf.reshape(xf, shape)

        samples = tf.concat((xr, xf), axis=0)

        rand = list(np.ones(len(shape)))
        rand[0] = tf.shape(input=samples)[0]
        u = tf.random.uniform(shape=rand, minval=0., maxval=1.)
        _, batch_std = tf.nn.moments(x=tf.reshape(samples, [-1]), axes=[0])

        delta = 0.5 * batch_std * u

        alpha = tf.random.uniform(shape=rand, minval=0., maxval=1.)
        x_hat = samples + (1 - alpha) * delta

        if labels is None:
            d_x_hat = d(x_hat, reuse=False)
        else:
            d_x_hat = d(x_hat, labels, reuse=False)

        gradients = tf.gradients(d_x_hat, [x_hat])[0]
        gradients_sqr = tf.square(gradients)
        slopes = tf.sqrt(tf.reduce_sum(input_tensor=gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))))
        gp = tf.reduce_mean(input_tensor=(tf.maximum(0., slopes - 1)) ** 2)

        return gp

    elif sampling == 'classic':
        rand = list(np.ones(len(shape)))
        rand[0] = tf.shape(input=xr)[0]
        epsilon = tf.random.uniform(shape=rand, minval=0., maxval=1.)
        x_hat = epsilon * xr + (1.0 - epsilon) * xf
        if labels is None:
            d_x_hat = d(x_hat, reuse=False)
        else:
            d_x_hat = d(x_hat, labels, reuse=False)

        grad_d_x_hat = tf.gradients(d_x_hat, [x_hat])[0]
        l2_norm = tf.sqrt(1e-8 + tf.reduce_sum(input_tensor=tf.square(grad_d_x_hat), axis=np.arange(1, len(
            grad_d_x_hat.shape))))  # Norme euclidienne / norme L2
        gp = tf.reduce_mean(input_tensor=tf.maximum(0., l2_norm - 1) ** 2)
        return gp


def ct_loss(xr, d, labels=None):
    if labels is not None:
        x = d(xr, labels, dropout=True, reuse=False)
        x_ = d(xr, labels, dropout=True, reuse=False)
    else:
        x = d(xr, dropout=True, reuse=False)
        x_ = d(xr, dropout=True, reuse=False)

    ct = tf.square(x - x_)
    ct_ = tf.maximum(ct, 0.0)
    return tf.reduce_mean(input_tensor=ct_)
