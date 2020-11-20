import numpy as np
import torch



def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    
def random_perm(A,p):
    for i in range(len(A)):
        if np.random.rand() < p:
            if A[i]==1:
                A[i]=0
            else:
                A[i]=1
    return A


def random_selection(X,y,n_samples):
    idx = np.arange(X.shape[0])
    idx = np.random.permutation(idx)
    idx1 = idx[:n_samples]
    idx2 = idx[n_samples:]

    return X[idx1,:],y[idx1],X[idx2,:],y[idx2]


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, f, xx, yy, **params):
    Z = f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy,Z,edgecolors='face', alpha=0.1, shading='auto', **params)
    #out = ax.contour(xx, yy, Z,colors=('darkred',),linewidths=(1,), alpha=0.5)
    out = ax.contour(xx, yy, Z,colors=('black',),linewidths=(1,), alpha=0.5)
    return out
    
    
def softmax(x,w):
    e_x = np.exp(w*(x - np.max(x)))
    return e_x / e_x.sum(axis=0)

def extract(v):
    return v.data.storage().tolist()