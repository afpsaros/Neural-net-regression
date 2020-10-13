# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:55:49 2020

@author: afpsaros
"""


import sys
sys.path.insert(0,'..')

from reg_classes import DNN
from data_disc import data_getter
from callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt 
from global_search import *
from planes_projections import planes_projections

import pickle 

with open('sm_out.txt', 'rb') as f:
    [budgets, M_snaps, M_preds, M_errors] = pickle.load(f)
    
with open('ca_out.txt', 'rb') as f:
    [CA_snaps, CA_preds, CA_errors] = pickle.load(f)     

c = len(M_snaps[0][0])
reps = len(M_snaps) 

y = [[i] * c for i in range(reps)]    
y = [item for sublist in y for item in sublist]


y = np.arange(0, c)
y = [y for _ in range(reps)]
y = [item for sublist in y for item in sublist]
print(y)

#%%
pj = planes_projections() 
w_vecs = []
y = []
for r in range(reps):
    y.append([r] * c)
    for ci in range(c):
        w_vecs.append(pj.abtovec(M_snaps[r][0][ci], M_snaps[r][1][ci]))
    
    y.append([r + reps] * c)
    for ci in range(c):
        w_vecs.append(pj.abtovec(CA_snaps[r][0][ci], CA_snaps[r][1][ci]))
y = [item for sublist in y for item in sublist]
print(y)
#%%
XX = w_vecs

# M_preds_flat = [item for sublist in XX for item in sublist]
# M_preds_flat = [np.array(item).flatten() for item in M_preds_flat]

# plt.plot(M_preds_flat[0])
# plt.plot(M_preds_flat[2])
# plt.show()
# print(M_preds_flat)
#%%
from sklearn.manifold import TSNE
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", n_colors = 2*c)
palette = sns.color_palette('RdYlGn',2*c)
# jet= plt.get_cmap('jet')
# palette= jet(np.linspace(0,1,2*c))
#%%
tsne = TSNE()
# X_embedded = tsne.fit_transform(M_preds_flat)
X_embedded = tsne.fit_transform(XX)
#%%
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = y, legend='full', palette=palette)


# colors = iter(jet(np.linspace(0,1,2*c)))
colors = list(iter(palette))

for ci in range(2*c):
    color = colors[y[ci * c]]
    # print(color)
    plt.plot(X_embedded[ci * c:(ci+1) * c,0], X_embedded[ci * c:(ci+1) * c,1], color = color)
    plt.scatter(X_embedded[ci * c,0], X_embedded[ci * c,1], marker = 's', s = 100, color = color)
    plt.scatter(X_embedded[(ci + 1) * c - 1,0], X_embedded[(ci + 1) * c - 1,1], marker = 'o', s = 100, color = color)
    
