#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:41:23 2019

@author: vicari
"""


import tensorflow as tf
import tensorflow.contrib as tc 
from tensorflow.python import control_flow_ops
import numpy as np
from scipy import io, misc
import spectral
import os

from IPython.display import clear_output, Image, display, HTML


###########################
#
# Function pour initialiser des poids
#
###########################
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


###########################
#
# Transforme une image de [-1,1] vers [0,255]
#
###########################
def to_img(x):
    return np.array(((x+1)*127.5), dtype=int)


###########################
#
# Transforme une image de [0,255] vers [-1,1]
#
###########################
def to_normalized(x):
    return np.array(((x/127.5)-1), dtype=np.float32)

###########################
#
# Function pour récupérer le scope dans TF
#
###########################
def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

###########################
#
# Function pour afficher tensorboard dans jupyter
#
# githubusercontent.com/Tony607
#
###########################
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, encoding='utf-8')
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


###########################
#
# Function pour le calcul de la norme spectral via power iteration
#
###########################
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

###########################
#
# Functions pour plot le classifier
#
###########################
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.25, x.max() + 0.25
    y_min, y_max = y.min() - 0.25, y.max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# @author: nico
def plot_contours_with_function(ax, f, xx, yy, **params):
    Z = f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy,Z,edgecolors='face', alpha=0.1,**params)
    out = ax.contour(xx, yy, Z,colors=('k',),linewidths=(1,), alpha=0.5)
    return out

# @author: nico
def plot_decision_with_function(ax, f, xx, yy, **params):
    Z = f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contour(xx, yy, Z,colors=('k',),linewidths=(1,), alpha=0.5)
    return out

###########################
#
# Softmax with temperature  
#
###########################   
def softmax(x,w):
    e_x = np.exp(w*(x - np.max(x)))
    return e_x / e_x.sum(axis=0)

###########################
#
# Linerar interp
#
###########################   
def lerp(a, b, cnt):
    return cnt*a + (1-cnt)*b

###########################
#
# Ceiling function
#
###########################
def ceil(x, v):
    return np.minimum(x,v)

def ceiling_with_interp(x, bs, lim=0.5):
    p = len([n for n in x if n > lim])/bs
    v = p*0.5 + (1-p)
    return np.minimum(x,v)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector
    
def unpad(imgs, shape=(-1,28,28)):
    tmp = np.array([x[2:-2,2:-2] for x in imgs])
    return np.reshape(tmp, shape)


import cv2
import numpy as np

def subimage(image, center, theta, width, height):

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image

def load_svhn(path):
    """
    Function to load data from SVHN dataset

    Parameters
    ----------
    path: string
        path to the file .mat
    
    Returns
    -------
    tuple
        X, y -> datas, labels
    """
    train_dict = io.loadmat(path)
    data = np.asarray(train_dict['X'])

    X = []
    for i in range(data.shape[3]):
        X.append(data[:,:,:,i])
    X = np.asarray(X)

    y = train_dict['y']
    for i in range(len(y)):
        if y[i]%10 == 0:
            y[i] = 0
    y = np.reshape(np.eye(10)[y],(-1,10))
    
    return (X,y)
##################################################
##################################################
##################################################   
##                                              ##
##                                              ##
##                                              ##
##                   HSI PART                   ##  
##                                              ##
##                 inspired by:                 ##
## https://gitlab.inria.fr/naudeber/DeepHyperX/ ##
##                                              ##
##                                              ##
##                                              ##
##################################################
##################################################
##################################################   



def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

    
def sample_gt(gt, train_size):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    if train_size > 1:
       train_size = int(train_size)

    train_gt = np.copy(gt)
    test_gt = np.copy(gt)
    for c in np.unique(gt):
        mask = gt == c
        for x in range(gt.shape[0]):
            first_half_count = np.count_nonzero(mask[:x, :])
            second_half_count = np.count_nonzero(mask[x:, :])
            try:
                ratio = first_half_count / second_half_count
                if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                    break
            except ZeroDivisionError:
                continue
        mask[:x, :] = 0
        train_gt[mask] = 0

    test_gt[train_gt > 0] = 0
    return train_gt, test_gt

def make_set(data, gt, c='pixel', path=None):
    X = []
    y = []
    if(c == 'pixel'):
        a, b = np.shape(gt)
        for i in range(a):
            for j in range(b):
                if gt[i][j] > 0:
                    X.append(data[i][j])
                    y.append(gt[i][j])
        return X, y
    elif(c == 'neigbors'):
        a, b = np.shape(gt)
        k = 0
        if(path is None):
            path = './'
        else:
            if not os.path.exists(path):
                os.makedirs(path)
        for i in range(np.shape(np.unique(gt))[0]):
            if not os.path.exists(path+str(i)):
                os.makedirs(path+str(i))
        for i in range(a):
            for j in range(b):
                if gt[i][j] > 0:
                    neigh = np.empty((5,5,48,1))
                    for m in range(-2,3):
                        for n in range(-2,3):
                            if(i+m < 0 or i+m >= a):
                                if(j+n < 0 or j+n >= b):
                                    neigh[m][n] = data[a-1][b-1]
                                else:
                                    neigh[m][n] = data[a-1][j+n]
                            else:
                                if(j+n < 0 or j+n >= b):
                                    neigh[m][n] = data[i+m][b-1]
                                else:
                                    neigh[m][n] = data[i+m][j+n]
                    np.save(path+str(gt[i][j])+'/'+str(k), np.reshape(neigh,(5,5,48)))
                    k+=1
    elif(c == 'window'):
        a, b = np.shape(gt)
        k = 0
        if(path is None):
            path = './'
        else:
            if not os.path.exists(path):
                os.makedirs(path)
        for i in range(np.shape(np.unique(gt))[0]):
            if not os.path.exists(path+str(i)):
                os.makedirs(path+str(i))
        for i in range(a):
            for j in range(b):
                if gt[i][j] > 0:
                    neigh = np.empty((5,5,48,1))
                    for m in range(-2,3):
                        for n in range(-2,3):
                            if(i+m < 0 or i+m >= a):
                                if(j+n < 0 or j+n >= b):
                                    neigh[m][n] = data[a-1][b-1]
                                else:
                                    neigh[m][n] = data[a-1][j+n]
                            else:
                                if(j+n < 0 or j+n >= b):
                                    neigh[m][n] = data[i+m][b-1]
                                else:
                                    neigh[m][n] = data[i+m][j+n]
                    np.save(path+str(gt[i][j])+'/'+str(k), np.reshape(neigh,(5,5,48)))
                    k+=1
    else:
        print(c, 'is not valid')