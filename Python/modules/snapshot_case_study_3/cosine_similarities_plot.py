# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:02:56 2020

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
    

def cosim(x, y):
    return np.inner(x, y) / np.sqrt(np.inner(x, x)) / np.sqrt(np.inner(y, y))

pj = planes_projections()    

c = len(M_snaps[0][0])

M_snaps_vecs = []
CA_snaps_vecs = []
for r in range(len(M_snaps)):
    
    snaps1 = []
    snaps2 = []
    
    for ci in range(c):
        snaps1.append(np.array(pj.abtovec(M_snaps[r][0][ci], M_snaps[r][1][ci])))
        snaps2.append(np.array(pj.abtovec(CA_snaps[r][0][ci], CA_snaps[r][1][ci])))
        
    M_snaps_vecs.append(snaps1)
    CA_snaps_vecs.append(snaps2)  

M_params_cosims = np.zeros([c,c])
CA_params_cosims = np.zeros([c,c])
for ci in range(c):
    
    for j in range(ci, c):
        M_params_cosims[ci, j] = cosim(M_snaps_vecs[0][ci], M_snaps_vecs[0][j]) 

        CA_params_cosims[ci, j] = cosim(CA_snaps_vecs[0][ci], CA_snaps_vecs[0][j]) 
        
M_params_cosims = M_params_cosims + np.tril(np.transpose(M_params_cosims), -1)
CA_params_cosims = CA_params_cosims + np.tril(np.transpose(CA_params_cosims), -1)
# np.set_printoptions(precision=3)
# print(param_cosims)
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,2)) 
ax1.set_title('Standard learning rate')
img = ax1.imshow(M_params_cosims.round(2), cmap="cividis")
img.set_visible(False)   

ax1.axis('off')
ax1.axis('tight')

the_table = ax1.table(cellText=M_params_cosims.round(2), rowLabels = np.array(M_errors[0]).round(1), \
                      colLabels = np.arange(1, c + 1),\
                      loc = 'center', cellLoc='center', cellColours=img.to_rgba(M_params_cosims.round(2)))
    
ax2.set_title('Cosine annealing')
ax2.axis('off')
ax2.axis('tight')
the_table = ax2.table(cellText=CA_params_cosims.round(2), rowLabels = np.array(CA_errors[0]).round(1), \
                      colLabels = np.arange(1, c + 1),\
                          loc = 'center', cellLoc='center', cellColours=img.to_rgba(CA_params_cosims.round(2)))
fig.tight_layout() 
plt.savefig('params_cosine_similarities.png', dpi = 300)    
#%%
M_preds_cosims = np.zeros([c,c])
CA_preds_cosims = np.zeros([c,c])
for ci in range(c):
    
    for j in range(ci, c):
        M_preds_cosims[ci, j] = cosim(np.array(M_preds[0][ci]).flatten(), np.array(M_preds[0][j]).flatten()) 

        CA_preds_cosims[ci, j] = cosim(np.array(CA_preds[0][ci]).flatten(), np.array(CA_preds[0][j]).flatten()) 
        
M_preds_cosims = M_preds_cosims + np.tril(np.transpose(M_preds_cosims), -1)
CA_preds_cosims = CA_preds_cosims + np.tril(np.transpose(CA_preds_cosims), -1)
# np.set_printoptions(precision=3)
# print(param_cosims)
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,2)) 
ax1.set_title('Standard learning rate')
img = ax1.imshow(M_preds_cosims.round(2), cmap="cividis")
img.set_visible(False)   

ax1.axis('off')
ax1.axis('tight')
the_table = ax1.table(cellText=M_preds_cosims.round(2), rowLabels = np.array(M_errors[0]).round(1), \
                      colLabels = np.arange(1, c + 1),\
                      loc = 'center', cellLoc='center', cellColours=img.to_rgba(M_preds_cosims.round(2)))

ax2.set_title('Cosine annealing')
ax2.axis('off')
ax2.axis('tight')
the_table = ax2.table(cellText=CA_preds_cosims.round(2), rowLabels = np.array(CA_errors[0]).round(1), \
                      colLabels = np.arange(1, c + 1),\
                          loc = 'center', cellLoc='center', cellColours=img.to_rgba(CA_preds_cosims.round(2)))
fig.tight_layout() 
plt.savefig('preds_cosine_similarities.png', dpi = 300)    

