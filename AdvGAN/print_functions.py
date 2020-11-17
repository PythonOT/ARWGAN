#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:39:38 2019

@author: vicari
"""

# Imports
#%%
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output, display
import numpy as np
from AdvGAN.utils import make_meshgrid, to_img
from AdvGAN import utils
import os

from itertools import product

# Layers
#%%

def plot_contours(ax, f, xx, yy, **params):
    Z = f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, edgecolors='face', alpha=0.1, **params)
    out = ax.contour(xx, yy, Z, colors=('k',), linewidths=(1,), alpha=0.5)
    return out

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def show_it_label(label):
    def f(samples, gan=None, label=None):
        clear_output()
        display(gan.f_prog)
        print("label :", label, "it :", gan.it)
    return lambda samples, gan : f(samples, gan, label)



def show_it_label_loss(label):
    def f(samples, gan=None, label=None):
        clear_output()
        display(gan.f_prog)
        print("label :", label, "it :", gan.it)
        print("c_loss :", gan.critic_losses[-1])
        print("g_loss :", gan.generator_losses[-1])
    return lambda samples, gan : f(samples, gan, label)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def show_loss(label):
    def f(samples, gan=None, label=None):
        clear_output()
        display(gan.f_prog)
        print('loss :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
    return lambda samples, gan : f(samples, gan)


def scatter_samples_and_loss(X_, Y_):
    def f(samples, gan=None, X=None, y=None):
        clear_output()
        display(gan.f_prog)
        colors =  np.array(list(['#377eb8', '#ff7f00', '#8457ff']))
        colors_adv =  np.array(list(['aqua', 'red', 'green']))
        print('samples :')
        fig = plt.figure(figsize=(6,6))
        if(gan.adv):
            xx, yy = make_meshgrid(X[:,0], X[:,1], h=.02)
            ax = fig.add_subplot(1, 1, 1)
            plot_contours(ax, lambda x: np.argmax(gan.predict(x), axis=1), xx, yy)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y])
            y_pred = gan.predict(samples)
            plt.scatter(samples[:, 0], samples[:, 1], color=colors_adv[np.argmax(y_pred, axis=1)])
            plt.xticks([], [])
            plt.yticks([], [])
            plt.legend(loc='best')
            plt.tight_layout()
        else:
            plt.scatter(X[:, 0], X[:, 1], color='blue')
            plt.scatter(samples[:, 0], samples[:, 1], color='red')#, color=colors_adv[labels])
        plt.show()
        plt.clf()
        print('neg loss :')
        plt.plot(gan.neg_losses)
        plt.show()
        plt.clf()
        print('loss :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
    return lambda samples, gan : f(samples, gan, X_, Y_)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def show_samples_and_loss(nb_channels=3):
    def f(samples, gan=None):
        clear_output()
        display(gan.f_prog)
        columns = 4
        rows = 8
        print('samples :')
        if(gan.adv):
            y_pred = np.argmax(gan.classifier.predict(samples), axis=1)
        f, axarr = plt.subplots(columns, rows)
    
        for i in range(columns):
            for j in range(rows):
                if nb_channels > 1:
                    test = to_img(np.reshape(samples[i*columns+j], [32,32,nb_channels]))
                    axarr[i,j].imshow(test) 
                else:
                    test = to_img(np.reshape(samples[i*columns+j], [32,32]))
                    axarr[i,j].imshow(test, cmap='gray') 

                if(gan.adv):
                    axarr[i,j].set_title(str(y_pred[i*columns+j]))
                axarr[i,j].axis('off')
        plt.show()
        plt.clf()
        print('neg loss :')
        plt.plot(gan.neg_losses)
        plt.show()
        plt.clf()
        print('loss critic :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
        plt.close()
    return lambda samples, gan : f(samples, gan)

def save_samples_and_loss(path='./training_save', nb_channels=3):
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    def f(samples, gan=None, renorm=True):
        clear_output()
        display(gan.f_prog)
        columns = 4
        rows = 8
        if(gan.adv):
            if gan.classifier_norm is not None:
                sample_norm = gan.classifier_norm(samples)
                y_pred = np.argmax(gan.predict(sample_norm), axis=1)
            else:
                y_pred = np.argmax(gan.predict(samples), axis=1)
        f, axarr = plt.subplots(columns, rows)
        
        for i in range(columns):
            for j in range(rows):
                if nb_channels > 1:
                    test = to_img(np.reshape(samples[i*columns+j], [32,32,nb_channels]))
                    axarr[i,j].imshow(test) 
                else:
                    test = to_img(np.reshape(samples[i*columns+j], [32,32]))
                    axarr[i,j].imshow(test, cmap='gray') 

                if(gan.adv):
                    axarr[i,j].set_title(str(y_pred[i*columns+j]))
                axarr[i,j].axis('off')
        idx = int(gan.it / gan.frequency_print)
        plt.savefig(os.path.join(path, str(idx).zfill(3) + '.jpg'))
        plt.clf()
        # loss neg
        plt.plot(gan.neg_losses)
        plt.savefig(os.path.join(path, 'neg_loss' + '.jpg'))
        plt.clf()
        # loss critic
        plt.plot(gan.critic_losses)
        plt.savefig(os.path.join(path, 'critic_loss' + '.jpg'))
        plt.clf()
        # loss gen
        plt.plot(gan.generator_losses)
        plt.savefig(os.path.join(path, 'gen_loss' + '.jpg'))
        plt.clf()
        plt.close()
        print("it :", gan.it)
        print("loss :", gan.neg_losses[-1])
    return lambda samples, gan : f(samples, gan)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def show_samples_and_loss_neigh():
    def f(samples, gan=None):
        clear_output()
        display(gan.f_prog)
        a = np.reshape(gan.neigh[0],(25,48))
        b = np.reshape(samples[0],(25,48))
        print('samples :')
        plt.plot(a[12])
        plt.show()
        plt.plot(b[12])
        plt.show()
        plt.show()
        plt.clf()
        print('loss critic :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
    return lambda samples, gan : f(samples, gan)


def plot_samples_mean_std():
    def plot_mean_std(samples, color='blue'):
        mean_spectrum = np.mean(samples, axis=0)
        std_spectrum = np.std(samples, axis=0)
        plt.plot(mean_spectrum - std_spectrum, linestyle='dotted', label='-std')
        plt.plot(mean_spectrum, label='mean')
        plt.plot(mean_spectrum + std_spectrum, linestyle='dotted', label='+std')
        plt.fill_between(range(len(mean_spectrum)), mean_spectrum + std_spectrum, mean_spectrum - std_spectrum, facecolor=color, alpha=0.2)
        
    def f(samples, gan=None):
        clear_output()
        display(gan.f_prog)
        print('samples :')
        samples = np.reshape(samples, (-1, 48))
        plot_mean_std(samples)
        plt.show()
        plt.clf()
        print('loss critic :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
    return lambda samples, gan : f(samples, gan)

def save_samples_mean_std(path='./training_save/plot'):
    if not os.path.exists(path):
        os.makedirs(path)
        
    def plot_mean_std(samples, color='blue'):
        mean_spectrum = np.mean(samples, axis=0)
        std_spectrum = np.std(samples, axis=0)
        plt.plot(mean_spectrum - std_spectrum, linestyle='dotted', label='-std')
        plt.plot(mean_spectrum, label='mean')
        plt.plot(mean_spectrum + std_spectrum, linestyle='dotted', label='+std')
        plt.fill_between(range(len(mean_spectrum)), mean_spectrum + std_spectrum, mean_spectrum - std_spectrum, facecolor=color, alpha=0.2)
        
    def f(samples, gan=None):
        clear_output()
        display(gan.f_prog)
        print('samples :')
        samples = np.reshape(samples, (-1, 48))
        plot_mean_std(samples)
        idx = int(gan.it / gan.frequency_print) 
        plt.savefig(os.path.join(path, str(idx).zfill(3) + '.jpg'))
        plt.clf()
        plt.plot(gan.neg_losses)
        plt.savefig(os.path.join(path, 'neg_loss' + '.jpg'))
        plt.clf()
        print('loss critic :')
        plt.plot(gan.critic_losses)
        plt.savefig(os.path.join(path, 'critic_loss' + '.jpg'))
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.savefig(os.path.join(path, 'generator_loss' + '.jpg'))
        plt.clf()
    return lambda samples, gan : f(samples, gan)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def make_fig_cycle(imgs, channels, total, columns, path=None, bands=None, size=None):

    rows = total // columns 
    rows += total % columns
    
    fig = plt.figure(figsize=(3.2*(columns), 1.1*(rows)))
    outer = gridspec.GridSpec(rows, columns, wspace=0, hspace=0)
    
    if size is None:
        size = [32, 32, 32]
    
    for k in range(total):
        inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                subplot_spec=outer[k], wspace=0, hspace=0)
        
        for j in range(3):
            ax = plt.Subplot(fig, inner[j])
            if channels[j] == 3:
                test = utils.to_img(np.reshape(imgs[j][k], [size[j],size[j],3]))
                ax.imshow(test) 
            elif channels[j] > 3:
                test = utils.to_img(np.reshape(imgs[j][k], [size[j],size[j],channels[j]]))
                ax.imshow(test[:,:,bands])             
            else:
                test = utils.to_img(np.reshape(imgs[j][k], [size[j],size[j]]))
                ax.imshow(test, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    if path is None:
        plt.show()
        plt.clf()
    else:
        plt.savefig(path)
        plt.clf()

def show_cycle_samples_and_loss(nb_channels_s=3, nb_channels_t=3, nb_img=None, bands=None, size_s=None, size_t=None):
    
    a = np.array([1, 2, 4, 8])
    columns = a[(np.abs(a - int(np.sqrt(nb_img)))).argmin()]
    
    def f(Xs, Xt, fake_Xt, fake_Xs, reconstruct_xs, reconstruct_xt, gan=None):
        clear_output()
        display(gan.f_prog)
        
        if nb_img is None:
            total = gan.batch_size
        else :
            total = nb_img
            

        print('samples source to target and back :')

        imgs = [Xs, fake_Xt, reconstruct_xs]
        channels = [nb_channels_s, nb_channels_t, nb_channels_s]
        if (size_s is None) or (size_t is None):
            size = None
        else:
            size = [size_s, size_t, size_s]
        make_fig_cycle(imgs, channels, total, columns, bands=bands, size=size)    

        print('samples target to source and back :')

        imgs = [Xt, fake_Xs, reconstruct_xt]
        channels = [nb_channels_t, nb_channels_s, nb_channels_t]     
        if (size_s is None) or (size_t is None):
            size = None
        else:
            size = [size_t, size_s, size_t]
        make_fig_cycle(imgs, channels, total, columns, bands=bands, size=size)    

        print('neg loss :')
        plt.plot(gan.neg_losses)
        plt.show()
        plt.clf()
        print('loss critic :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
        plt.close()
        if gan.make_preds:
            print('loss Clf source :')
            plt.plot(gan.clf_s_test_losses)
            plt.show()
            plt.clf()
            plt.close()
            print('loss Clf target :')
            plt.plot(gan.clf_t_test_losses)
            plt.show()
            plt.clf()
            plt.close()
            
    return lambda Xs, Xt, fake_Xt, fake_Xs, reconstruct_xs, reconstruct_xt, gan : f(Xs, Xt, fake_Xt, fake_Xs, reconstruct_xs, reconstruct_xt, gan)

def save_cycle_samples_and_loss(nb_channels_s=3, nb_channels_t=3, nb_img=None, path='./training_save', bands=None, size_s=None, size_t=None):
    if not os.path.exists(path):
        os.makedirs(path)
        
    a = np.array([1, 2, 4, 8])
    columns = a[(np.abs(a - int(np.sqrt(nb_img)))).argmin()]
    
    def f(Xs, Xt, fake_Xt, fake_Xs, reconstruct_xs, reconstruct_xt, gan=None):
        clear_output()
        display(gan.f_prog)
        
        if nb_img is None:
            total = gan.batch_size
        else :
            total = nb_img
            
        idx = int(gan.it / gan.frequency_print)
        
        imgs = [Xs, fake_Xt, reconstruct_xs]
        channels = [nb_channels_s,nb_channels_t,nb_channels_s]   
        if (size_s is None) or (size_t is None):
            size = None
        else:
            size = [size_s, size_t, size_s]
        make_fig_cycle(imgs, channels, total, columns, bands=bands, size=size, path=os.path.join(path, 's2t_'+str(idx).zfill(3) + '.jpg'))    
        
        imgs = [Xt, fake_Xs, reconstruct_xt]
        channels = [nb_channels_t,nb_channels_s,nb_channels_t]   
        if (size_s is None) or (size_t is None):
            size = None
        else:
            size = [size_t, size_s, size_t]
        make_fig_cycle(imgs, channels, total, columns, bands=bands, size=size, path=os.path.join(path, 't2s_'+str(idx).zfill(3) + '.jpg'))  

        plt.plot(gan.neg_losses)
        plt.savefig(os.path.join(path, 'neg_loss' + '.jpg'))
        plt.clf()
        plt.plot(gan.critic_losses)
        plt.savefig(os.path.join(path, 'critic_loss' + '.jpg'))
        plt.clf()
        plt.plot(gan.generator_losses)
        plt.savefig(os.path.join(path, 'generator_loss' + '.jpg'))
        plt.clf()
        plt.plot(gan.clf_losses)
        plt.savefig(os.path.join(path, 'clf_loss' + '.jpg'))
        plt.clf()
        plt.close()
    return lambda Xs, Xt, fake_Xt, fake_Xs, reconstruct_xs, reconstruct_xt, gan : f(Xs, Xt, fake_Xt, fake_Xs, reconstruct_xs, reconstruct_xt, gan)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def show_sample_and_loss_v2(nb_channels=3, bands=[0, 1, 2], size=128, total=32):
    
    a = np.array([1, 2, 4, 8])
    columns = a[(np.abs(a - int(np.sqrt(total)))).argmin()]
    
    def f(samples, gan=None):
        clear_output()
        display(gan.f_prog)
    
        rows = total // columns 
        rows += total % columns
        
        print('samples :')
        tPlot, ax = plt.subplots(
        nrows=rows, ncols=columns, sharex=True, sharey=False, 
        )
           
        k = 0
        for a in ax.flatten():
            if nb_channels > 3:
                test = samples[k]
                test = test[:, :, bands]
                test = to_img(np.reshape(test, [size,size,3]))
                a.imshow(test) 
            elif nb_channels > 1:
                test = to_img(np.reshape(samples[k], [size,size,nb_channels]))
                a.imshow(test) 
            else:
                test = to_img(np.reshape(samples[k], [size, size]))
                a.imshow(test, cmap='gray')                
            a.set_xticks([])
            a.set_yticks([])
            k+=1
            
        plt.show()
        plt.clf()
        print('neg loss :')
        plt.plot(gan.neg_losses)
        plt.show()
        plt.clf()
        print('loss critic :')
        plt.plot(gan.critic_losses)
        plt.show()
        plt.clf()
        print('loss generator :')
        plt.plot(gan.generator_losses)
        plt.show()
        plt.clf()
        plt.close()
    return lambda samples, gan : f(samples, gan)

def save_sample_and_loss_v2(nb_channels=3, bands=[0, 1, 2], size=128, total=32, path='./training_save'):
    if not os.path.exists(path):
        os.makedirs(path)
    a = np.array([1, 2, 4, 8])
    columns = a[(np.abs(a - int(np.sqrt(total)))).argmin()]
    
    def f(samples, gan=None):
        clear_output()
        display(gan.f_prog)
    
        rows = total // columns 
        rows += total % columns
        
        tPlot, ax = plt.subplots(
        nrows=rows, ncols=columns, sharex=True, sharey=False, 
        )
           
        k = 0
        for a in ax.flatten():
            if nb_channels > 3:
                test = samples[k]
                test = test[:, :, bands]
                test = to_img(np.reshape(test, [size,size,3]))
                a.imshow(test) 
            elif nb_channels > 1:
                test = to_img(np.reshape(samples[k], [size,size,nb_channels]))
                a.imshow(test) 
            else:
                test = to_img(np.reshape(samples[k], [size, size]))
                a.imshow(test, cmap='gray')                
            a.set_xticks([])
            a.set_yticks([])
            k+=1
            
        idx = int(gan.it / gan.frequency_print)       
        plt.savefig(os.path.join(path, str(idx).zfill(3) + '.jpg'))
        plt.clf()
        plt.plot(gan.neg_losses)
        plt.savefig(os.path.join(path, 'neg_loss' + '.jpg'))
        plt.clf()
        plt.plot(gan.critic_losses)
        plt.savefig(os.path.join(path, 'critic_loss' + '.jpg'))
        plt.clf()
        plt.plot(gan.generator_losses)
        plt.savefig(os.path.join(path, 'generator_loss' + '.jpg'))
        plt.clf()
        plt.close()
    return lambda samples, gan : f(samples, gan)