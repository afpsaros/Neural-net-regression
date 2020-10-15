# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:26:09 2020

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

with open('kmn_out.txt', 'rb') as f:
    [no_models, KMN_errors] = pickle.load(f)  

c = len(KMN_errors[0])

# print(np.array(KMN_errors).round(2))
# print(np.transpose(np.array(KMN_errors).round(2)))

KMN_errors = np.flip(np.array(KMN_errors).round(2), 0)

print(KMN_errors)
    
fig, ax = plt.subplots(1, 1, figsize=(4,2)) 
# ax1.set_title('Standard learning rate')
img = ax.imshow(KMN_errors, cmap="autumn")
img.set_visible(False)   

ax.axis('off')
ax.axis('tight')
the_table = plt.table(cellText=KMN_errors, rowLabels = np.flip(no_models), \
                      colLabels = np.arange(1, c + 1),\
                          loc = 'center', cellLoc='center', cellColours=img.to_rgba(KMN_errors))

fig.tight_layout() 
plt.savefig('kmn_plot.png', dpi = 300)    