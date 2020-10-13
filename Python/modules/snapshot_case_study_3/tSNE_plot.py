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

c = len(M_snaps[0][0])
reps = len(M_snaps) 

y = [[i] * c for i in range(reps)]    
y = [item for sublist in y for item in sublist]
print(y)

# pj = planes_projections()   
# M_preds = M_snaps

M_preds_flat = [item for sublist in M_preds for item in sublist]
M_preds_flat = [np.array(item).flatten() for item in M_preds_flat]

# plt.plot(M_preds_flat[0])
# plt.plot(M_preds_flat[2])
# plt.show()
# print(M_preds_flat)
#%%
from sklearn.manifold import TSNE
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", c)
#%%
tsne = TSNE()
X_embedded = tsne.fit_transform(M_preds_flat)
#%%
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = y, legend='full', palette=palette)

jet= plt.get_cmap('jet')
colors = iter(jet(np.linspace(0,1,c)))
for ci in range(2):
    color = next(colors)
    plt.plot(X_embedded[ci * c:(ci+1) * c,0], X_embedded[ci * c:(ci+1) * c,1], color = color)
    plt.scatter(X_embedded[ci * c,0], X_embedded[ci * c,1], marker = 's', s = 100, color = color)
    plt.scatter(X_embedded[(ci + 1) * c - 1,0], X_embedded[(ci + 1) * c - 1,1], marker = 'o', s = 100, color = color)

# plt.plot(X_embedded[c:2*c,0], X_embedded[c:2*c,1])
# plt.scatter(X_embedded[c,0], X_embedded[c,1], marker = 's', s = 100)
# plt.scatter(X_embedded[2*c-1,0], X_embedded[2*c-1,1], marker = 'o', s = 100)




# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = np.arange(0, 2*c))