#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 08:44:43 2019

@author: vicari
"""

import numpy as np
import tensorflow as tf
from AdvGAN_2 import utils
from sklearn.utils import shuffle


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Dataset:
    def __init__(self):
        return


class DatasetBuilder(Dataset):
    def __init__(self, inputs_data, inputs_labels, batch_size=64, augmentations=None, augmentations_chances=None,
                 n_cores=4):

        super().__init__()
        print("Number of data : ", len(inputs_data))

        self.max_cursor = len(inputs_data)
        self.X = np.array(inputs_data, dtype=np.float32)
        self.y = np.array(inputs_labels, dtype=np.float32)
        self.cursor = 0

        self.augmentations = None
        self.augmentations_chances = None

        if augmentations is not None:

            if not isinstance(augmentations, (list, tuple, np.ndarray)):
                self.augmentations = [augmentations]
            else:
                self.augmentations = augmentations

            if not isinstance(augmentations_chances, (list, tuple, np.ndarray)):
                self.augmentations_chances = [augmentations_chances] * len(augmentations)
            else:
                if len(augmentations) != len(augmentations_chances):
                    raise Exception('augmentations and augmentations_chances must have the same length')
                else:
                    self.augmentations_chances = augmentations_chances

        self.is_tf = True
        self.set = False
        self.bs = batch_size
        self.n_cores = n_cores

        self.proba = np.zeros((len(inputs_labels)), dtype=np.float32)

        self.shape_x = list(np.shape(inputs_data))
        self.shape_y = list(np.shape(inputs_labels))

        self.dataset = None

    def get_number_of_samples(self):
        return self.max_cursor

    def shape(self):
        return self.shape_x, self.shape_y

    def adv_poba(self, classifier, reshape=None, norm=None, use_softmax=False, temp=20):

        y_pred = classifier.predict(self.X)
        self.proba = np.array(1 - np.sum(self.y * y_pred, axis=-1), dtype=np.float32)
        if use_softmax:
            self.proba = np.array(utils.softmax(self.proba, temp), dtype=np.float32)

    def build(self, run_adv_proba=False, classifier=None, use_softmax=False, temp=20):
        # Run adv proba if require
        if run_adv_proba and classifier is not None:
            self.adv_poba(classifier, use_softmax=use_softmax, temp=20)

        data = tf.data.Dataset.from_tensor_slices(self.X)

        if self.augmentations is not None:
            
            for f, p in zip(self.augmentations, self.augmentations_chances):
                data = data.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > p, lambda: f(x), lambda: x),
                                num_parallel_calls=self.n_cores)

        proba = tf.data.Dataset.from_tensor_slices(self.proba)
        labels = tf.data.Dataset.from_tensor_slices(self.y)
        zip_ = tf.data.Dataset.zip((data, proba, labels))
        self.dataset = iter(zip_.repeat().shuffle(len(self.y)).batch(self.bs).prefetch(self.bs))

    def next_batch(self):
        return next(self.dataset)


#########################################################

###########
#
# Must Be Upgraded
#
###########
class CycleDataset(Dataset):
    def __init__(self, X_a, X_b, X, inputs_data, inputs_labels, y_a=None, y_b=None, mask_y_b=None, augmentation=None,
                 batch_size=64, n_cores=4):  # 'flip'
        """
        Class to build and use a dataset

        Parameters
        ----------
        X: Array
            any type of data
        y: Array
            OneHot of labels, must have the same len as X
        """
        super().__init__()
        self.X_a = np.array(X_a[:])
        self.X_b = np.array(X_b[:])

        print("Number of data : ", len(self.X_a) + len(self.X_b))
        if y_a is None:
            y_a = np.zeros(len(self.X_a))
        if y_b is None:
            y_b = np.zeros(len(self.X_b))

        self.max_cursor = np.minimum(len(self.X_a), len(self.X_b))

        self.y_a = np.array(y_a[:])

        self.y_b = np.array(y_b[:])

        self.mask_y_b = np.array(mask_y_b[:])

        self.cursor = 0
        self.augmentation = augmentation
        self.is_tf = True
        self.set = False
        self.bs = batch_size
        self.n_core = n_cores

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
                                                          output_types=(tf.float32, tf.float32, tf.float32,
                                                                        tf.float32, tf.float32))

            def wrapped_func(a, b, y):
                return self.transform(a, b, y)

            # self.dataset = self.dataset.map(wrapped_func, num_parallel_calls=self.n_core)

            self.dataset = self.dataset.batch(self.bs)
            self.iter = self.dataset.make_one_shot_iterator()
            self.set = True

        return self.iter.get_next()

    def shuffle(self):
        """
        Function to shuffle the dataset
        """
        if self.proba is None:
            self.X_a, self.y_a = shuffle(self.X_a, self.y_a)
            self.X_b, self.y_b, self.mask_y_b = shuffle(self.X_b, self.y_b, self.mask_y_b)
        else:
            raise Exception('not implemented yet')

    def get_next(self):
        while 1:
            for i in range(self.max_cursor):
                res_a = self.X_a[i]
                res_b = self.X_b[i]
                res_ya = self.y_a[i]
                res_yb = self.y_b[i]
                res_mask = self.mask_y_b[i]

                yield res_a, res_b, res_ya, res_yb, res_mask

            self.shuffle()

    def transform(self, a, b, y):
        a = tf.clip_by_value(a + 0.002 * tf.random.normal(tf.shape(input=a), 0, 0.1), -1.0, 1.0)
        b = tf.clip_by_value(b + 0.002 * tf.random.normal(tf.shape(input=b), 0, 0.1), -1.0, 1.0)
        return a, b, y

#########################################################
