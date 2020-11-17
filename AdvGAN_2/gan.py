#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:40:03 2019

@author: vicari
"""

import configparser

# Imports
# %%
import numpy as np
import tensorflow as tf
import datetime
from tqdm import tqdm

from AdvGAN_2 import print_functions, dataset, models, utils

# Test block, will be removed
# %%
print('AdvGAN library')

# Classes
# %%


###########################
#
# Class AdvGAN
#
###########################
class AdvWGAN:

    def set_name(self, name):
        """

        :param name:
        """
        self.name = name

    def set_generator(self, gen):
        """

        :param gen:
        """
        self.generator = gen

    def set_discriminator(self, disc):
        """

        :param disc:
        """
        self.discriminator = disc

    def set_batch_size(self, bs=None, mini_batch_max=None):
        """

        :param bs:
        :param mini_batch_max:
        """
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
            
        self.set_z_dim(self.z_dim)

    def set_z_dim(self, z_dim):
        """

        :param z_dim:
        """
        if not isinstance(z_dim, (list, tuple, np.ndarray)):
            z_dim = [z_dim]
        self.z_dim = z_dim
        self.z_size = [self.batch_size] + z_dim
        self.z_rand = np.ones(len(self.z_size), dtype=np.int)
        self.z_rand[-1] = self.z_size[-1]
        self.z_rand[0] = self.z_size[0]

    def set_loss(self, loss):
        """

        :param loss:
        """
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

    def set_optimizer_d(self, optimizer_d):
        """

        :param optimizer_d:
        """
        if optimizer_d['name'].lower() == 'adam':

            if optimizer_d['learning_rate'] is None:
                optimizer_d['learning_rate'] = 1e-4

            if optimizer_d['beta1'] is None:
                optimizer_d['beta1'] = 0.9

            if optimizer_d['beta2'] is None:
                optimizer_d['beta2'] = 0.99

            self.optimizer_D = tf.compat.v1.train.AdamOptimizer(learning_rate=float(optimizer_d['learning_rate']),
                                                                beta1=float(optimizer_d['beta1']),
                                                                beta2=float(optimizer_d['beta2']))

        elif optimizer_d['name'].lower() == 'rmsprop':

            if optimizer_d['learning_rate'] is None:
                optimizer_d['learning_rate'] = 1e-4

            self.optimizer_D = tf.keras.optimizers.RMSprop(learning_rate=float(optimizer_d['learning_rate']))

        else:
            raise Exception('the optimizer {} is not valid'.format(optimizer_d['name']))

    def set_optimizer_g(self, optimizer_g):
        """

        :param optimizer_g:
        """
        if optimizer_g['name'].lower() == 'adam':

            if optimizer_g['learning_rate'] is None:
                optimizer_g['learning_rate'] = 1e-4

            if optimizer_g['beta1'] is None:
                optimizer_g['beta1'] = 0.9

            if optimizer_g['beta2'] is None:
                optimizer_g['beta2'] = 0.99

            self.optimizer_G = tf.compat.v1.train.AdamOptimizer(learning_rate=float(optimizer_g['learning_rate']),
                                                                beta1=float(optimizer_g['beta1']),
                                                                beta2=float(optimizer_g['beta2']))

        elif optimizer_g['name'].lower() == 'rmsprop':

            if optimizer_g['learning_rate'] is None:
                optimizer_g['learning_rate'] = 1e-4

            self.optimizer_G = tf.keras.optimizers.RMSprop(learning_rate=float(optimizer_g['learning_rate']))

        else:
            raise Exception('the optimizer {} is not valid'.format(optimizer_g['name']))

    def set_dataset(self, data):
        """

        :param data:
        """
        if not isinstance(data, dataset.Dataset):
            raise Exception('the dataset must be a Dataset object')
        else:
            self.data = data
            if not self.use_softmax:
                self.number_of_samples = self.data.get_number_of_samples()

    def set_training_options(self, nb_iter=None, init_step=None, n_c_iters_start=None, n_c_iters=None,
                             g_iter_start=None, g_iter=None):
        """

        :param nb_iter:
        :param init_step:
        :param n_c_iters_start:
        :param n_c_iters:
        :param g_iter_start:
        :param g_iter:
        """
        if init_step is not None:
            self.init_step = init_step

        if n_c_iters_start is not None:
            self.n_c_iters_start = n_c_iters_start

        if n_c_iters is not None:
            self.n_c_iters = n_c_iters

        if nb_iter is not None:
            self.nb_iter = nb_iter

        if g_iter_start is not None:
            self.g_iter_start = g_iter_start

        if g_iter is not None:
            self.g_iter = g_iter

    def set_classifier(self, classifier=None, nb_class=None):
        """

        :param classifier:
        :param nb_class:
        """
        if classifier is not None:
            self.classifier = classifier

        if nb_class is not None:
            self.nb_class = nb_class

    def set_logs(self, print_method=None, frequency_print=None, frequency_logs=None, logs_path=None):
        """

        :param print_method:
        :param frequency_print:
        :param frequency_logs:
        :param logs_path:
        """
        if print_method is not None:
            self.print_method = print_method

        if frequency_print is not None:
            self.frequency_print = frequency_print

        if frequency_logs is not None:
            self.frequency_logs = frequency_logs

        if logs_path is not None:
            self.logs_path = logs_path

    def set_temperature(self, temperature):
        """

        :param temperature:
        """
        self.temperature = temperature

    def set_adv(self, adv=True):
        """

        :param adv:
        """
        self.adv = adv

    def set_uses(self, use_proba=None, use_ceil=None, ceil_value=None, use_labels=None, use_mask=None,
                 use_softmax=None, use_recon=None, recon_beta=None, reg_losses=None, reg_term=None,
                 use_clf=None, label_used=None, clf_alpha=None):
        """

        :param use_proba:
        :param use_ceil:
        :param ceil_value:
        :param use_labels:
        :param use_mask:
        :param use_softmax:
        :param use_recon:
        :param recon_beta:
        :param reg_losses:
        :param reg_term:
        :param use_clf:
        :param label_used:
        :param clf_alpha:
        """
        if use_proba is not None:
            self.use_proba = use_proba
        if use_ceil is not None:
            self.use_ceil = use_ceil
        if ceil_value is not None:
            self.ceil_value = ceil_value
        if use_labels is not None:
            self.use_labels = use_labels
        if use_mask is not None:
            self.use_mask = use_mask
        if use_softmax is not None:
            self.use_softmax = use_softmax
        if recon_beta is not None:
            self.recon_beta = recon_beta
        if use_recon is not None:
            self.use_recon = use_recon
        if reg_losses is not None:
            self.reg_losses = reg_losses
        if reg_term is not None:
            self.reg_term = reg_term
        if use_clf is not None:
            self.use_clf = use_clf
        if label_used is not None:
            self.label_used = label_used
        if clf_alpha is not None:
            self.clf_alpha = clf_alpha

    def __init__(self):

        self.generator = models.FullyConnected(
            hiddens_dims=[256, 256, 256],
            ouput_dims=2,
            activation=tf.nn.relu,
            output=tf.nn.sigmoid
        )

        self.discriminator = models.FullyConnected(
            hiddens_dims=[128, 128, 128, 128, 128, 128],
            ouput_dims=1,
            activation=tf.nn.relu
        )

        self.accumulate_grd = False
        self.n_minibatches = 1
        self.batch_size = 64
        self.name = "AdvGAN"

        self.use_adaptive_adv = False

        self.z_dim = [None]
        self.gp_lambda = 10.0
        self.ct_lambda = 2.0

        self.optimizer_D = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
        self.optimizer_G = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

        self.nb_class = 10
        self.init_step = 1
        self.n_c_iters_start = 5
        self.n_c_iters = 5
        self.g_iter_start = 1
        self.g_iter = 1
        self.nb_iter = 5000
        self.ceil_value = 0.75

        self.saver = None

        self.reg_losses = False

        self.data = None
        self.classifier = None

        self.classifier_reshape = None
        self.classifier_norm = None

        self.dataset_is_tf = False

        self.nb_classes = 10

        self.print_method = print_functions.show_it_label('advGAN')
        self.frequency_print = 100
        self.frequency_logs = None
        self.logs_path = './logs/'

        self.loss = 'lp'

        self.temperature = 20
        self.clip_value = 0.1
        self.m_value = 0.2

        self.adv = True
        self.use_proba = True
        self.use_ceil = False
        self.use_labels = False
        self.use_mask = False
        self.mask = None
        self.use_recon = False
        self.recon_beta = 1.0
        self.use_softmax = True

        self.reg_losses = False
        self.reg_term = 1e-3
        self.use_clf = False
        self.label_used = [1, 0, 0, 0, 0, 0]
        self.clf_alpha = 1.0

        self.g_loss = None
        self.d_loss = None

        self.batch_reg = False

        self.X_real = None
        self.labels = None
        self.classifier_proba = None
        self.adv_map = None
        self.z = None
        self.is_training = None
        self.pred_real = None
        self.X_fake = None
        self.pred_fake = None
        self.loss_clf = None
        self.neg_loss = None
        self.accum_vars_D = None
        self.accumulation_counter_D = None
        self.accum_vars_G = None
        self.accumulation_counter_G = None
        self.D_sum = None
        self.G_sum = None
        self.accum_ops_G = None
        self.accum_ops_D = None
        self.zero_ops_G = None
        self.zero_ops_D = None
        self.G_step = None
        self.D_step = None
        self.D_clipping = None

        self.f_prog = None
        self.critic_losses = None
        self.neg_losses = None
        self.generator_losses = None
        self.writer = None
        self.sample_labels = None

        self.built = False

    def summary(self):
        print("AdvGan Options :")

        print()
        print("accumulate_grd :", self.accumulate_grd)
        print("n_minibatches :", self.n_minibatches)
        print("batch_size :", self.batch_size)
        print("name :", self.name)

        print("use_adaptative_adv :", self.use_adaptive_adv)

        print("z_dim :", self.z_dim)
        print("gp_lambda :", self.gp_lambda)
        print("ct_lambda :", self.ct_lambda)

        print("optimizer_D :", self.optimizer_D)
        print("optimizer_G :", self.optimizer_G)

        print("init_step :", self.init_step)
        print("n_c_iters_start :", self.n_c_iters_start)
        print("n_c_iters :", self.n_c_iters)
        print("g_iter :", self.g_iter)
        print("nb_iter :", self.nb_iter)

        print("reg_losses :", self.reg_losses)

        print("frequency_print :", self.frequency_print)
        print("frequency_logs :", self.frequency_logs)
        print("logs_path :", self.logs_path)

        print("loss :", self.loss)

        print("temperature :", self.temperature)
        print("clip_value :", self.clip_value)

        print("adv :", self.adv)
        print("use_proba :", self.use_proba)
        print("use_ceil :", self.use_ceil)
        print("use_labels :", self.use_labels)
        print("use_recon :", self.use_recon)
        print("recon_beta :", self.recon_beta)
        print("use_clf :", self.use_clf)

    #@tf.function
    def noise(self, batch_size=None):
        """

        :rtype: object
        :param batch_size:
        :param z_size:
        :return:
        """
        if batch_size is not None:
            z_size = self.z_size
            z_rand = self.z_rand
            z_size[0] = batch_size
            z_rand[0] = batch_size
        else:
            z_size = self.z_size
            z_rand = self.z_rand

        ones = tf.ones(z_size)
        return ones * tf.random.normal(shape=z_rand, mean=0., stddev=1.)
    
    def replacenan(self, t):
        return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

    #@tf.function
    def get_data(self):

        z = self.noise()

        if self.use_mask:
            x_real, classifier_proba, mask, labels = self.data.next_batch()
            if self.use_labels:
                mask = labels
            if self.use_proba:
                probs = self.classifier.predict(x_real)[:, :, :, 4]
                labels_masked = tf.cast(labels * mask, tf.float32)
                pm = tf.cast(tf.expand_dims(probs, -1) * labels_masked, tf.float32)
                classifier_proba = tf.cast(
                    1.0 - (tf.reduce_sum(input_tensor=pm, axis=[1, 2, 3]) / tf.reduce_sum(input_tensor=labels_masked,
                                                                                          axis=[1, 2, 3])), tf.float32)
        else:
            x_real, classifier_proba, labels = self.data.next_batch()
            mask = None

        classifier_proba = self.replacenan(classifier_proba)
        
        if self.adv and self.use_softmax:
            if self.use_ceil and self.ceil_value is not None:
                classifier_proba = tf.minimum(classifier_proba, self.ceil_value)
            classifier_proba = tf.nn.softmax(classifier_proba * self.temperature)
            classifier_proba = tf.expand_dims(classifier_proba, -1)

        return x_real, labels, classifier_proba, mask, z

    #@tf.function
    def generate(self, z=None, x=None, mask=None, training=False):

        if z is None:
            z = self.noise()

        if self.use_mask:
            x_fake = self.generator(x, mask, z, training=training)
        else:
            x_fake = self.generator(z, training=training)

        return x_fake

    #@tf.function
    def gradient_penalty(self, f_, x_real_, x_fake_):
        def _gradient_penalty(f, x_real, x_fake=None, lp=True, epsilon=1e-10):
            def _interpolate(a, b=None):
                shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
                alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.shape)
                return inter

            x = _interpolate(x_real, x_fake)

            with tf.GradientTape() as tape:
                tape.watch([x])
                pred = f(x)
            gradients = tape.gradient(pred, x)
            gradients = tf.reshape(gradients, [gradients.shape[0], -1])
            gp_ = tf.norm(gradients, axis=1)  # [b]
            gp_ = tf.maximum(0., (gp_ - 1.0)) if lp else (gp_-1.0)
            gp_ = tf.reduce_mean(gp_ ** 2)

            return gp_

        if self.loss == 'lp':
            gp = _gradient_penalty(f_, x_real_, x_fake=x_fake_, lp=True)
        elif self.loss == 'gp':
            gp = _gradient_penalty(f_, x_real_, x_fake=x_fake_, lp=False)
        elif self.loss == 'ct':
            gp = _gradient_penalty(f_, x_real_, x_fake=x_fake_, lp=True)
        else:
            gp = tf.constant(0.0)

        return gp

    #@tf.function
    def reconstruction_loss(self, x_real, x_fake):
        loss = tf.reduce_mean(tf.abs(x_real - x_fake))
        return loss

    #@tf.function
    def loss_d(self, x_real, x_fake, mask=None, classifier_proba=None, training=False):
        if self.use_mask:
            pred_real = self.discriminator(x_real, training=training)
        else:
            pred_real = self.discriminator(x_real, training=training)

        if self.adv:
            pred_real *= classifier_proba
            if not self.use_softmax:
                pred_real *= self.number_of_samples / self.batch_size

        if self.use_mask:
            pred_fake = self.discriminator(x_fake, training=training)
        else:
            pred_fake = self.discriminator(x_fake, training=training)
        
        if self.adv and not self.use_softmax:
            d_loss = - (tf.reduce_sum(input_tensor=pred_real) - tf.reduce_sum(input_tensor=pred_fake))
        elif self.adv:
            d_loss = - (tf.reduce_sum(input_tensor=pred_real) - tf.reduce_mean(input_tensor=pred_fake))
        else:
            d_loss = -(tf.reduce_mean(input_tensor=pred_real) - tf.reduce_mean(input_tensor=pred_fake))

        return d_loss

    #@tf.function
    def loss_g(self, x_fake, mask=None, training=False):

        if self.use_mask:
            pred_fake = self.discriminator(x_fake, training=training)
        else:
            pred_fake = self.discriminator(x_fake, training=training)
            
        g_loss = -tf.reduce_mean(input_tensor=pred_fake)

        return g_loss

    #@tf.function
    def d_update(self):
        # Discriminator udpdate
        x_real, labels, classifier_proba, mask, z = self.get_data()
        x_fake = self.generate(z, x_real, mask, True)
        with tf.GradientTape() as d_tape:
            d_loss = self.loss_d(x_real, x_fake, mask, classifier_proba, True)
            neg_loss = -d_loss
            d_loss = d_loss + self.gp_lambda * self.gradient_penalty(self.discriminator, x_real, x_fake)
        grads_d = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_D.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))

        return d_loss, neg_loss

    #@tf.function
    def g_update(self):

        x_real, labels, classifier_proba, mask, z = self.get_data()

        with tf.GradientTape() as g_tape:
            x_fake = self.generate(z, x_real, mask, True)
            g_loss = self.loss_g(x_fake, mask)
            
            if self.use_recon:
                if mask is not None:
                    a = ((1-labels)*mask + (1-mask))
                    x_fake_masked = a * x_fake
                    x_real_masked = a * x_real
                    
                    recon = self.reconstruction_loss(x_real_masked, x_fake_masked)
                    g_loss += self.recon_beta*recon

            if self.use_clf:
                y_t = mask * self.label_used
                preds = self.classifier.predict(x_fake)
                preds = preds * mask
                preds = 1 - preds
                loss_clf = self.clf_alpha * tf.reduce_mean(input_tensor=-tf.reduce_sum(
                    input_tensor=y_t * tf.math.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=[1]))
                g_loss += loss_clf

        grads_g = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.optimizer_G.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        return g_loss

    def save_weights(self, name='last'):
        self.discriminator.save_weights('./checkpoints/discriminator_'+name)
        self.generator.save_weights('./checkpoints/generator_'+name)

    def load_weights(self, name='last'):
        self.discriminator.load_weights('./checkpoints/discriminator_'+name)
        self.generator.load_weights('./checkpoints/generator_'+name)

    def save_model(self, name=''):
        self.discriminator.save('./models/discriminator_'+name)
        self.generator.save('./models/generator_'+name)

    def load_model(self, name=''):
        self.discriminator = tf.keras.models.load_model('./models/discriminator_'+name)
        self.generator = tf.keras.models.load_model('./models/generator_'+name)

    def train(self, init_step=None, n_c_iters_start=None, n_c_iters=None, nb_iter=None, print_method=None,
              frequency_print=None, frequency_logs=None, use_ceil=None, restore_weight=False):
        """
        Function to train AdvGAN

        Parameters
        ----------
        """

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

        self.data.build(run_adv_proba=self.adv, classifier=self.classifier, use_softmax=not self.use_softmax, temp=self.temperature)

        if restore_weight:
            self.load_weights()

        it = 0
        d_loss = 0
        neg_loss = 0

        n_c_iters = self.n_c_iters_start

        self.neg_losses = []
        self.critic_losses = []
        self.generator_losses = []
        #pbar = tqdm(total = self.frequency_print)
        for self.it in range(it, self.nb_iter + 1):
            if self.it > self.init_step:
                n_c_iters = self.n_c_iters

            for _ in range(n_c_iters):
                d_loss, neg_loss = self.d_update()

            # Generator udpdate
            g_loss = self.g_update()
            #pbar.update(1)
            self.neg_losses.append(neg_loss)
            self.critic_losses.append(d_loss)
            self.generator_losses.append(g_loss)

            if self.it % self.frequency_print == 0:
                if self.print_method is not None:
                    x_real, labels, classifier_proba, mask, z = self.get_data()
                    samples = self.generate(z, x_real, mask, False)

                    self.print_method(samples=samples, gan=self)
                    print(self.it, 'd loss:', float(d_loss), 'g loss:', float(g_loss))
                    self.save_weights()
                #pbar.n = 0
                #pbar.last_print_n = 0
                #pbar.update()

#############################################################

def make_gan(file):
    config = configparser.ConfigParser()
    config.read(file)

    adv_gan = AdvWGAN()

    # MISC
    misc = config['MISC']

    adv_gan.set_name(misc.get('name'))
    adv_gan.set_adv(misc.getboolean('adv'))
    adv_gan.set_z_dim(eval(misc.get('z_dim')))
    adv_gan.set_batch_size(misc.getint('batch_size'))
    adv_gan.set_batch_size(misc.getint('mini_batch_max'))
    adv_gan.set_temperature(misc.getfloat('temperature'))

    # USES
    uses = config['USES']
    adv_gan.set_uses(uses.getboolean('use_proba'), uses.getboolean('use_ceil'), uses.getfloat('ceil_value'),
                     uses.getboolean('use_labels'), uses.getboolean('use_mask'), uses.getboolean('use_softmax'),
                     uses.getboolean('use_recon'), uses.getfloat('recon_beta'), uses.getboolean('reg_losses'),
                     uses.getfloat('reg_term'),
                     uses.getboolean('use_clf'), eval(str(uses.get('label_used'))), uses.getfloat('clf_alpha'))

    # REGULARIZATION
    reg = dict(config['REGULARIZATION'])

    if len(reg.keys()) > 0:
        adv_gan.set_loss(reg)

    # OPTIMIZERS
    opti_d = dict(config['D_OPTIMIZER'])

    if len(opti_d.keys()) > 0:
        adv_gan.set_optimizer_d(opti_d)

    opti_g = dict(config['G_OPTIMIZER'])

    if len(opti_g.keys()) > 0:
        adv_gan.set_optimizer_g(opti_g)

    # TRAIN
    train = config['TRAIN']

    nb_iter = train.getint('nb_iter')
    init_step = train.getint('init_step')
    n_c_iters_start = train.getint('n_c_iters_start')
    n_c_iter = train.getint('n_c_iter')
    g_iter_start = train.getint('g_iter_start')
    g_iter = train.getint('g_iter')

    adv_gan.set_training_options(nb_iter, init_step, n_c_iters_start, n_c_iter, g_iter_start, g_iter)

    # LOGS

    log = config['LOGS']

    print_method = log.get('print_method')

    if print_method == 'show_it_label':
        print_method = print_functions.show_it_label(log.get('label', 'label'))

    frequency_print = log.getint('frequency_print')
    frequency_logs = log.getint('frequency_logs')
    logs_path = log.get('logs_path')

    adv_gan.set_logs(print_method, frequency_print, frequency_logs, logs_path)

    return adv_gan
