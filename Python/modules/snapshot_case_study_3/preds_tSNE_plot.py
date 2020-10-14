# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:52:46 2020

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

#%%
snaps_num = 4

preds = []
y = []
for r in range(reps):
    y.append([r] * snaps_num)
    for ci in range(c - snaps_num, c):
        preds.append(M_preds[r][ci].flatten())
    
    y.append([r + reps] * snaps_num)
    for ci in range(c - snaps_num, c):
        preds.append(CA_preds[r][ci].flatten())
        
y = [item for sublist in y for item in sublist]
print(y)

XX = preds

#%%
from sklearn.manifold import TSNE
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette('YlOrRd',2*reps)

#%%
tsne = TSNE()
X_embedded = tsne.fit_transform(XX)
#%%
sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
for i in range(int(2 * reps / 4)):
    
    plt.figure(figsize=(5,4))
    
    ax = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = y, legend='full', \
                        palette=palette)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    plt.xlabel('t-SNE axis 1', fontsize = 15)
    plt.ylabel('t-SNE axis 2', fontsize = 15)
        
    colors = list(iter(palette))
        
    for r in range(i * 4, i * 4 + 4):
        color = colors[y[r * snaps_num]]
        plt.plot(X_embedded[r * snaps_num:(r+1) * snaps_num,0], X_embedded[r * snaps_num:(r+1) * snaps_num,1], color = color)
        plt.scatter(X_embedded[r * snaps_num,0], X_embedded[r * snaps_num,1], marker = 's', s = 100, color = color)
        plt.scatter(X_embedded[(r + 1) * snaps_num - 1,0], X_embedded[(r + 1) * snaps_num - 1,1], marker = 'o', s = 100, color = color)

    plt.tight_layout()
    plt.savefig('preds_tSNE_{}.png'.format(i), dpi = 300)            
    plt.show()
    
sns.reset_orig()