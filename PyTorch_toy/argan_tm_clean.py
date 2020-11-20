import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import copy

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

import seaborn as sns; sns.set(color_codes=True)

from utils import plot_history, random_perm, make_meshgrid, plot_contours, softmax
from utils import extract
from models import Classifier, Generator, Discriminator

    


seed=1978
np.random.seed(seed)


#####################################
#             DATA
#####################################

output_dir='./'
n_samples=4000
data = make_moons(n_samples=n_samples,noise=0.1)

X_full, y_full = data
X_full = torch.FloatTensor(X_full)
y_full = torch.LongTensor(y_full)
X_full = StandardScaler().fit_transform(X_full)



####################################
#        TRAIN CLASSIFIER
####################################

X_full = torch.FloatTensor(X_full).cuda()
y_full = torch.LongTensor(y_full).cuda()
X_data = X_full
y_data = y_full
y_data = torch.reshape(y_data, (len(y_data), 1))
n_epoch = 1
net = Classifier()
net.cuda()

net.train()
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3,  betas=(0.9, 0.999))

for epoch in range(n_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(int(len(X_data)/8)):
        inputs = Variable(X_data[i])
        label = y_data[i]
        
        optimizer.zero_grad()
        outputs = torch.reshape(net(inputs), (1, 2))
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training Classifier')


####################################
#        TRAIN ARWGAN
####################################

## Load and modify dataset

one_class = True  # Only generate adversarial examples for 1 class
if one_class==True:
    print('one class selection')
    index = []
    for i in range(len(y_full)):
        if y_full[i] == 0:
            index.append(i)
    y_data = copy.deepcopy(y_full[index])
    X_data = copy.deepcopy(X_full[index])
    n_data = 2000

else:
    y_data = copy.deepcopy(y_full)
    X_data = copy.deepcopy(X_full)
    n_data = 4000


fig_toy = plt.figure(figsize=(5, 10))
list_w = [20, 0]
gen_data_toy=[]
for i_exp, w in enumerate(list_w):
    d_learning_rate = 0.0001
    g_learning_rate = 0.0001
    n_epoch = 601

    #Define network
    G = Generator().cuda()
    D = Discriminator().cuda()
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)

    #Define Learning
    d_optimizer = optim.RMSprop(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.RMSprop(G.parameters(), lr=g_learning_rate)
    G_loss = []
    D_loss = []


    n_critic = 5
    BATCH_SIZE = 256

    LAMBDA = 10
    output_dir='./'

    for epoch in range(n_epoch):
        running_loss = 0.0
        for i in range(20):
            for _ in range(n_critic):
                D.zero_grad()

                #Compute gradient fake data
                idx = np.random.choice(n_data, size=BATCH_SIZE, replace=False)
                real_data = Variable(X_data[idx]).cuda()
                label = y_data[idx]

                cl = net(real_data)
                cur_proba = torch.FloatTensor(F.softmax(cl.cpu(), dim = 1)).detach().cpu().numpy()
                new_weights = np.ones(BATCH_SIZE) - cur_proba[np.arange(BATCH_SIZE),
                                                              label.cpu().reshape(BATCH_SIZE)]
                new_weights = np.minimum(new_weights, 0.5)
                new_weights = softmax(new_weights, w)
                new_weights = BATCH_SIZE * new_weights
                new_weights = torch.cuda.FloatTensor(new_weights)

                real_err = torch.mean(D(real_data).reshape(BATCH_SIZE).mul(new_weights))

                #compute grad fake data
                z = torch.randn(BATCH_SIZE, 10).cuda()
                d_noise = Variable(z)
                fake_output = G(d_noise).detach()
                fake_err = torch.mean(D(fake_output))

                d_err = -(real_err - fake_err) #minimize the opposite function
                # Weight Clipping's WGAN
                for param in D.parameters():
                    param.data.clamp_(-0.01, 0.01)

                d_err.backward()
                d_optimizer.step()

            G.zero_grad()

            z = torch.randn(BATCH_SIZE, 10).cuda()
            g_noise = Variable(z)
            g_fake_data = G(g_noise)
            g_err = -torch.mean(D(g_fake_data))
            g_err.backward()
            g_optimizer.step()

            G_loss.append(extract(g_err)[0])
            D_loss.append(extract(real_err - fake_err)[0])



        if (epoch) % 100 == 0:
            print("%s: D: %s G: %s " % (epoch + 1, extract(real_err - fake_err)[0], extract(g_err)[0]))

    z = torch.randn(1500, 10)   # fixed noise
    z = Variable(z).cuda()
    gen_data_toy.append(G(z).detach().cpu().numpy())

print('Finished Training ARWGAN and WGAN')
    
####################################
#      DISPLAY ARWGAN RESULTS
####################################
print('Display results')

cmap_light = ListedColormap(['white', 'lightblue'])
cmap_bold = ListedColormap(['orangered', 'gold'])

cmap_all='Blues'

# build the mesh only for display purpose
h = 0.02
xx, yy = make_meshgrid(X_full.cpu()[:,0], X_full.cpu()[:,1], h=.02)

x_min, x_max = X_full.cpu()[:, 0].min() - 0.5, X_full.cpu()[:, 0].max() + 0.5
y_min, y_max = X_full.cpu()[:, 1].min() - 0.5, X_full.cpu()[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(1, 2, 1)
plot_contours(ax, lambda x: torch.max(net(torch.FloatTensor(x).cuda()).cpu(), 1)[1], xx, yy,cmap=cmap_light)
ax = sns.kdeplot(gen_data_toy[1][:,0], gen_data_toy[1][:,1], color='r',
                cmap="Reds", shade=True, shade_lowest=False, legend=True, label='generated data')
plt.scatter(X_full.cpu()[:,0],X_full.cpu()[:,1],c=y_full.cpu(), label='original data', cmap=cmap_all, edgecolors='k',marker='o',s=40, alpha=0.1)
plt.title('WGAN', fontsize=16)
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])
plt.tight_layout()


ax = fig.add_subplot(1, 2, 2)
plot_contours(ax, lambda x: torch.max(net(torch.FloatTensor(x).cuda()).cpu(), 1)[1], xx, yy,cmap=cmap_light)
ax = sns.kdeplot(gen_data_toy[0][:,0], gen_data_toy[0][:,1], color='r',
                cmap="Reds", shade=True, shade_lowest=False, legend=True, label='generated data')
plt.scatter(X_full.cpu()[:,0],X_full.cpu()[:,1],c=y_full.cpu(), label='original data', cmap=cmap_all, edgecolors='k',marker='o',s=40, alpha=0.1)
plt.title('ARWGAN, w=20, c=0.5', fontsize=16)
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

outfile = os.path.join(output_dir, 'test_toy_img.png'.format(epoch))
fig.savefig(outfile, bbox_inches='tight')
