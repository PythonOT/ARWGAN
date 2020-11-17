#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 08:44:43 2019

@author: vicari
"""
import numpy as np
from sklearn.utils import shuffle
import scipy.io as sio
from scipy import sparse
import tensorflow as tf
import os
import random
import skimage
import imageio

from AdvGAN import utils

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Dataset:
    def __init__(self, X, y=None, augmentation=None): # 'flip'
        """
        Class to build and use a dataset

        Parameters
        ----------
        X: Array
            any type of data
        y: Array
            OneHot of labels, must have the same len as X
        """
        self.X = np.array(X[:])
        print("Number of data : ",len(self.X))
        if(y is None):
            y=np.zeros(len(self.X))
        self.y = y
        self.cursor = 0
        self.proba = None
        self.use_softmax =False
        self.augmentation = augmentation
        self.temp = 15
        self.is_tf = False
        
    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        return list(np.shape(self.X)), list(np.shape(self.y))
    
    def adv_proba(self, classifier, ceil_value=0.75):   
        """
        Function to set proba with the following rule
            proba = 1-P where P is the probability of the gt class
        """
        if self.proba is not None:
            return
        x = self.X[:]
        y_pred = classifier.predict(x)
        self.proba = 1-np.sum(self.y*y_pred, axis=-1)
        
        self.proba = utils.ceil(self.proba, ceil_value)

        self.proba = utils.softmax(self.proba , self.temp)
    
    def next_batch(self, size, with_labels=False, with_proba=False):
        """
        Function to get the next_batch of datas

        Parameters
        ----------
        size: int
            size of the batch
        with_labels: bool
            True to have the labels, ignored if with_proba is True
        with_proba: bool
            True to have the probabilty alongside X
        
        Returns
        -------
        np.array
            array of data
        [np.array]
            array of label or proba
        """
        if( (self.cursor + size) > len(self.X)):
            self.cursor = 0
            self.shuffle()
            
        flip = 0
        
        if self.augmentation == 'flip':
            if random.random() > 0.5:
                flip = 1
            
        if(with_proba and with_labels):
            res_x = self.X[self.cursor:self.cursor+size]
            res_y = self.y[self.cursor:self.cursor+size]
            res_c = self.proba[self.cursor:self.cursor+size]
            self.cursor+= size
            
            if flip:
                res_x = np.flip(res_x, 2)
            return res_x, res_y, res_c
        
        if(with_proba):
            res_x = self.X[self.cursor:self.cursor+size]
            res_y = self.proba[self.cursor:self.cursor+size]
            self.cursor+= size
            
            if flip:
                res_x = np.flip(res_x, 2)
            return res_x, res_y
        if(with_labels):
            res_x = self.X[self.cursor:self.cursor+size]
            res_y = self.y[self.cursor:self.cursor+size]
            self.cursor+= size
            
            if flip:
                res_x = np.flip(res_x, 2)
            return res_x, res_y
        else:
            res = self.X[self.cursor:self.cursor+size]
            self.cursor+= size
            if flip:
                res = np.flip(res, 2)
            return res
        
    def shuffle(self):
        """
        Function to shuffle the dataset
        """
        if(self.proba is None):
            self.X, self.y = shuffle(self.X, self.y)
        else:
            self.X, self.y, self.proba = shuffle(self.X, self.y, self.proba )

#########################################################
            
class Dataset_from_files(Dataset):
    def __init__(self, path, one_label=None, with_maps=False, adv=True):
        """
        Class to build and use a dataset

        Parameters
        ----------
        path: String
            path to the repositorie
        """
        path, dirs, files = next(os.walk(path))
        self.liste = None
        self.y = None
        self.adv_map = None
        self.cursor = 0
        self.proba = None
        self.with_maps = with_maps
        self.adv = adv
        self.is_tf = False
        
        if with_maps:
            n = int(len(files)/3)
            self.liste = []
            self.y = []
            self.cursor = 0
            self.proba = []
            if adv:
                self.adv_map = []
            for i in range(n):
                self.liste.append(path+'/'+str(i)+'_datas.npy')
                self.y.append(path+'/'+str(i)+'_labels.npy')      
                if adv:
                    gt = np.load(path+'/'+str(i)+'_labels.npy')
                    self.adv_map.append(path+'/'+str(i)+'_probas.npy')
                    proba_map = np.load(path+'/'+str(i)+'_probas.npy')
                    size = 1
                    for dim in np.shape(gt): size *= dim
                    mean_prob = np.sum(proba_map)/size
                    self.proba.append(mean_prob)
            return
        
        if one_label is not None:
            tags = np.ones(len(files), dtype=np.int32)*one_label
            self.liste = [path+'/' + f for f in files]
            self.y = np.eye(21)[tags]
            return        
        
        for dir_ in dirs:
            p, dirs, files = next(os.walk(path+dir_))
            tags = np.ones(len(files))*int(dir_)
            if self.liste is None:
                self.liste = [p+'/' + f for f in files]
                self.y = tags[:]
            else:
                self.liste = np.concatenate([self.liste, [p+'/'  + f for f in files]])
                self.y = np.concatenate([self.y, tags[:]])
        return

        
    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        a = np.load(self.liste[0])
        return list(np.shape([a]))
    
    def adv_poba(self, classifier, reshape=None, norm=None, batch_size=512):   
        """
        Function to set proba with the following rule
            proba = 1-P where P is the probability of the gt class
        """
        if self.proba is not None:
            return
        for i in range(0, len(self.liste), batch_size):
            if(i+batch_size > len(self.liste)):
                batch_size = len(self.liste)-i
            files = self.liste[i:i+batch_size]
            x = [np.load(f) for f in files]
            if reshape is not None:
                x = np.reshape(x, (reshape))
            if norm is not None:
                x = norm(x)
            y_pred = classifier([x])
            if self.proba is None:
                self.proba = 1-np.sum(self.y[i:i+batch_size]*y_pred, axis=-1)
            else:
                prob = 1-np.sum(self.y[i:i+batch_size]*y_pred, axis=-1)
                self.proba = np.concatenate([self.proba, prob])
            i = i + batch_size
            
    def next_batch(self, size, with_labels=False, with_proba=False, with_map_proba=False):
        """
        Function to get the next_batch of datas

        Parameters
        ----------
        size: int
            size of the batch
        with_labels: bool
            True to have the labels, ignored if with_proba is True
        with_proba: bool
            True to have the probabilty alongside X
        
        Returns
        -------
        np.array
            array of data
        [np.array]
            array of label or proba
        """
        if( (self.cursor + size) > len(self.liste)):
            self.cursor = 0
            self.shuffle()
        
        files = self.liste[self.cursor:self.cursor+size]
        res_x = [np.load(f) for f in files]
        res_x = (np.array(res_x)*2)-1
            
        if(self.with_maps or (with_proba and with_labels)):
            files = self.y[self.cursor:self.cursor+size]
            res_y = [np.load(f) for f in files]
            if self.adv:
                res_probas = np.expand_dims(self.proba[self.cursor:self.cursor+size], -1)
            self.cursor += size
            if(with_map_proba):
                files = self.adv_map[self.cursor:self.cursor+size]
                res_map = np.expand_dims([np.load(f) for f in files], -1)
                return np.array(res_x, dtype=np.float), np.array(res_y, dtype=np.float), np.array(res_probas, dtype=np.float), np.array(res_map, dtype=np.float) 
            if self.adv or with_proba:
                return np.array(res_x, dtype=np.float), np.array(res_y, dtype=np.float), np.array(res_probas, dtype=np.float) 
            else : 
                return np.array(res_x, dtype=np.float), np.array(res_y, dtype=np.float) 
        if(with_proba):
            res_y = self.proba[self.cursor:self.cursor+size]
            self.cursor+= size
            return res_x, res_y
        if(with_labels):
            res_y = self.y[self.cursor:self.cursor+size]
            self.cursor+= size
            return res_x, res_y
        else:
            self.cursor+= size
            return res_x

    def shuffle(self):
        """
        Function to shuffle the dataset
        """
        if(self.proba is None):
            self.liste, self.y = shuffle(self.liste, self.y)
        else:
            self.liste, self.y, self.proba = shuffle(self.liste, self.y, self.proba )

#########################################################
class TF_dataset(Dataset):
    def __init__(self, X, y, augmentation=None, batch_size=64, n_core=4): # 'flip'
        """
        Class to build and use a dataset

        Parameters
        ----------
        X: Array
            any type of data
        y: Array
            OneHot of labels, must have the same len as X
        """
        self.X = np.array(X[:])
        
        print("Number of data : ",len(self.X))
            
        self.max_cursor = len(self.X_a)

        self.y = np.array(y[:])
        self.cursor = 0
        self.augmentation = augmentation
        self.is_tf = True
        self.set = False
        self.bs = batch_size
        self.n_core = n_core
        
        self.proba = None
        self.use_softmax =False
        
        
        self.shape_x = list(np.shape(self.X))
        self.shape_y = list(np.shape(self.y))
        
    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        return self.shape_x, self.shape_y
    
    def adv_poba(self, classifier, reshape=None, norm=None):   
        """
        Function to set proba with the following rule
            proba = 1-P where P is the probability of the gt class
        """
        if self.proba is not None:
            return
        x = self.X[:]
        y_pred = classifier.predict(x)
        self.proba = 1-np.sum(self.y*y_pred, axis=-1)
        
        if self.use_softmax :
            self.proba = utils.softmax(self.proba , self.temp)
    
    def next_batch(self, with_labels=False, with_proba=False):
        """
        Function to get the next_batch of datas

        Parameters
        ----------
        size: int
            size of the batch
        with_labels: bool
            True to have the labels, ignored if with_proba is True
        with_proba: bool
            True to have the probabilty alongside X
        
        Returns
        -------
        np.array
            array of data
        [np.array]
            array of label or proba
        """
        if not self.set:
            self.dataset = tf.data.Dataset.from_generator(self.get_next,
                                             output_types= (tf.float32, tf.float32, tf.float32))
    
            def wrapped_func(a, b, y):
                return self.transform(a, b, y)
                '''
                return tf.py_func(func = self.transform,
                                                inp = [a, b, y],
                                                Tout = (tf.float32, # (H,W,3) img
                                                        tf.float32,
                                                        tf.float32))  # label
                '''
            
            self.dataset = self.dataset.map(wrapped_func,
                                  num_parallel_calls=self.n_core)
            
            self.dataset = self.dataset.batch(self.bs)
            self.iter = self.dataset.make_one_shot_iterator()
            self.set = True
            
        return self.iter.get_next()
        
    def shuffle(self):
        """
        Function to shuffle the dataset
        """
        if(self.proba is None):
            self.X_a, self.y_a= shuffle(self.X_a, self.y_a)
            self.X_b, self.y_b = shuffle(self.X_b, self.y_b)
        else:
            raise Exception('not implemented yet')
            self.X, self.y, self.proba = shuffle(self.X, self.y, self.proba )
            
    def get_next(self, with_labels=False, with_proba=False):
        while 1:  
            for i in range(self.max_cursor):
                res_a = self.X_a[i]
                res_b = self.X_b[i]              
                res_y = self.y_a[i]
                
                yield res_a, res_b, res_y
                
            self.shuffle()
            
    def transform(self, a, b, y):
        a = tf.clip_by_value(a + 0.2*tf.random_normal(tf.shape(a), 0, 0.1), -1.0, 1.0)
        b = tf.clip_by_value(b + 0.2*tf.random_normal(tf.shape(b), 0, 0.1), -1.0, 1.0)
        return a, b, y
    
##############################################
    
class Cycle_Dataset(Dataset):
    def __init__(self, X_a, X_b, y_a=None, y_b=None, augmentation=None): # 'flip'
        """
        Class to build and use a dataset

        Parameters
        ----------
        X: Array
            any type of data
        y: Array
            OneHot of labels, must have the same len as X
        """
        self.X_a= np.array(X_a[:])
        self.X_b= np.array(X_b[:])
        
        print("Number of data : ",len(self.X_a) + len(self.X_b))
        if(y_a is None):
            y_a=np.zeros(len(self.X_a))
        if(y_b is None):
            y_b=np.zeros(len(self.X_b))
            
        self.max_cursor = np.minimum(len(self.X_a), len(self.X_b))

        if y_a is not None:
            self.y_a = np.array(y_a[:])
        if y_b is not None:
            self.y_b = np.array(y_b[:])
            
        self.cursor = 0
        self.proba = None
        self.use_softmax =False
        self.augmentation = augmentation
        self.temp = 20
        self.is_tf = False
        
    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        return list(np.shape(self.X_a)), list(np.shape(self.X_b)), list(np.shape(self.y_a)), list(np.shape(self.y_b))
    
    def adv_poba(self, classifier, reshape=None, norm=None):   
        """
        Function to set proba with the following rule
            proba = 1-P where P is the probability of the gt class
        """
        if self.proba is not None:
            return
        x = self.X_a[:]
        if reshape is not None:
            x = np.reshape(x, (reshape))
        if norm is not None:
            x = norm(x)
        y_pred = classifier(x)
        self.proba = 1-np.sum(self.y_a*y_pred, axis=-1)
        
        if self.use_softmax :
            self.proba = utils.softmax(self.proba , self.temp)
    
    def next_batch(self, size, with_labels=False, with_proba=False):
        """
        Function to get the next_batch of datas

        Parameters
        ----------
        size: int
            size of the batch
        with_labels: bool
            True to have the labels, ignored if with_proba is True
        with_proba: bool
            True to have the probabilty alongside X
        
        Returns
        -------
        np.array
            array of data
        [np.array]
            array of label or proba
        """
        if( (self.cursor + size) > self.max_cursor):
            self.cursor = 0
            self.shuffle()
            
        flip = 0
        
        if self.augmentation == 'flip':
            if random.random() > 0.5:
                flip = 1
            
        if(with_proba and with_labels):
            '''
            res_x = self.X[self.cursor:self.cursor+size]
            res_y = self.y[self.cursor:self.cursor+size]
            res_c = self.proba[self.cursor:self.cursor+size]
            self.cursor+= size
            
            if flip:
                res_x = np.flip(res_x, 2)
            '''
            raise Exception('not implemented yet')
            return 
        
        if(with_proba):
            '''
            res_x = self.X[self.cursor:self.cursor+size]
            res_y = self.proba[self.cursor:self.cursor+size]
            self.cursor+= size
            
            if flip:
                res_x = np.flip(res_x, 2)
            
            '''
            raise Exception('not implemented yet')
            return
        if(with_labels):
            res_a = self.X_a[self.cursor:self.cursor+size]
            res_b = self.X_b[self.cursor:self.cursor+size]
            
            res_y = self.y_a[self.cursor:self.cursor+size]
            
            self.cursor+= size

            return res_a, res_b, res_y
        else:
            res_a = self.X_a[self.cursor:self.cursor+size]
            res_b = self.X_b[self.cursor:self.cursor+size]
            self.cursor+= size

            return res_a, res_b
        
    def shuffle(self):
        """
        Function to shuffle the dataset
        """
        if(self.proba is None):
            self.X_a, self.y_a= shuffle(self.X_a, self.y_a)
            self.X_b, self.y_b = shuffle(self.X_b, self.y_b)
        else:
            raise Exception('not implemented yet')
            self.X, self.y, self.proba = shuffle(self.X, self.y, self.proba )

#########################################################
            
class Cycle_Dataset_tf(Dataset):
    def __init__(self, X_a, X_b, y_a=None, y_b=None, augmentation=None, batch_size=64, n_core=4): # 'flip'
        """
        Class to build and use a dataset

        Parameters
        ----------
        X: Array
            any type of data
        y: Array
            OneHot of labels, must have the same len as X
        """
        self.X_a= np.array(X_a[:])
        self.X_b= np.array(X_b[:])
        
        print("Number of data : ",len(self.X_a) + len(self.X_b))
        if(y_a is None):
            y_a=np.zeros(len(self.X_a))
        if(y_b is None):
            y_b=np.zeros(len(self.X_b))
            
        self.max_cursor = np.minimum(len(self.X_a), len(self.X_b))

        self.y_a = np.array(y_a[:])

        self.y_b = np.array(y_b[:])
            
        self.cursor = 0
        self.augmentation = augmentation
        self.is_tf = True
        self.set = False
        self.bs = batch_size
        self.n_core = n_core
        
        self.proba = None
        
        self.shape_a = list(np.shape(self.X_a))
        self.shape_b = list(np.shape(self.X_b))
        self.shape_ya = list(np.shape(self.y_a))
        self.shape_yb = list(np.shape(self.y_b))
        
    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        return self.shape_a, self.shape_b, self.shape_ya, self.shape_yb
    
    def adv_poba(self, classifier, reshape=None, norm=None):   
        """
        Function to set proba with the following rule
            proba = 1-P where P is the probability of the gt class
        """
        raise Exception('not implemented yet')
    
    def next_batch(self, with_labels=False, with_proba=False):
        """
        Function to get the next_batch of datas

        Parameters
        ----------
        size: int
            size of the batch
        with_labels: bool
            True to have the labels, ignored if with_proba is True
        with_proba: bool
            True to have the probabilty alongside X
        
        Returns
        -------
        np.array
            array of data
        [np.array]
            array of label or proba
        """
        if not self.set:
            self.dataset = tf.data.Dataset.from_generator(self.get_next,
                                             output_types= (tf.float32, tf.float32, tf.float32))
    
            def wrapped_func(a, b, y):
                return self.transform(a, b, y)
                '''
                return tf.py_func(func = self.transform,
                                                inp = [a, b, y],
                                                Tout = (tf.float32, # (H,W,3) img
                                                        tf.float32,
                                                        tf.float32))  # label
                '''
            
            self.dataset = self.dataset.map(wrapped_func,
                                  num_parallel_calls=self.n_core)
            
            self.dataset = self.dataset.batch(self.bs)
            self.iter = self.dataset.make_one_shot_iterator()
            self.set = True
            
        return self.iter.get_next()
        
    def shuffle(self):
        """
        Function to shuffle the dataset
        """
        if(self.proba is None):
            self.X_a, self.y_a= shuffle(self.X_a, self.y_a)
            self.X_b, self.y_b = shuffle(self.X_b, self.y_b)
        else:
            raise Exception('not implemented yet')
            self.X, self.y, self.proba = shuffle(self.X, self.y, self.proba )
            
    def get_next(self, with_labels=False, with_proba=False):
        while 1:  
            for i in range(self.max_cursor):
                res_a = self.X_a[i]
                res_b = self.X_b[i]              
                res_y = self.y_a[i]
                
                yield res_a, res_b, res_y
                
            self.shuffle()
            
    def transform(self, a, b, y):
        a = tf.clip_by_value(a + 0.2*tf.random_normal(tf.shape(a), 0, 0.1), -1.0, 1.0)
        b = tf.clip_by_value(b + 0.2*tf.random_normal(tf.shape(b), 0, 0.1), -1.0, 1.0)
        return a, b, y
    
class Dataset_simple_cycle(Dataset):
    def __init__(self, Xs, ys, Xt, yt):
        
        self.Xs = Xs
        self.ys = ys
        
        self.Xt = Xt
        self.yt = yt

        self.is_tf = False
        
        self.cursor = 0
        self.max_cursor = np.minimum(len(self.Xs), len(self.Xt))

    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        return list(self.x_shape), list(self.y_shape)
    
    def adv_poba(self, classifier, reshape=None, norm=None):   
        return None
    
    def next_batch(self, size):
        
        if( (self.cursor + size) > self.max_cursor):
            self.cursor = 0
            self.shuffle()
            
        Xs = self.Xs[self.cursor:self.cursor+size]
        ys = self.ys[self.cursor:self.cursor+size]
        Xt = self.Xt[self.cursor:self.cursor+size]
        yt = self.yt[self.cursor:self.cursor+size]
        self.cursor+= size
        
        return Xs, ys, Xt, yt
    
    def shuffle(self):
        self.Xs, self.ys= shuffle(self.Xs, self.ys)
        self.Xt, self.yt = shuffle(self.Xt, self.yt)

#########################################################

class Dataset_from_file_cycle(Dataset):
    def __init__(self, Xs, ys, Xt, yt):
        
        self.Xs = Xs
        self.ys = ys
        
        self.Xt = Xt
        self.yt = yt

        self.is_tf = False
        
        self.cursor = 0
        self.max_cursor = np.minimum(len(self.Xs), len(self.Xt))

    def shape(self):
        """
        function that return the shape of the datas
        
        Returns
        -------
        list
            list of the shape of X
        """
        return list(self.x_shape), list(self.y_shape)
    
    def adv_poba(self, classifier, reshape=None, norm=None):   
        return None
    
    def next_batch(self, size):
        
        if( (self.cursor + size) > self.max_cursor):
            self.cursor = 0
            self.shuffle()
            
        Xs = np.array([imageio.imread(x) for x in self.Xs[self.cursor:self.cursor+size]])
        ys = self.ys[self.cursor:self.cursor+size]
        Xt = np.array([imageio.imread(x) for x in self.Xt[self.cursor:self.cursor+size]])
        yt = self.yt[self.cursor:self.cursor+size]
        self.cursor+= size
        
        return Xs, ys, Xt, yt
    
    def shuffle(self):
        self.Xs, self.ys= shuffle(self.Xs, self.ys)
        self.Xt, self.yt = shuffle(self.Xt, self.yt)

#########################################################