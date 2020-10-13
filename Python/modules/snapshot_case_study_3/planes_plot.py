# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:55:45 2020

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
#%%
pj = planes_projections()    

reps = len(M_snaps)
c = len(M_snaps[0][0])
#%%
r = 4

w_vecs = []
for ci in range(-3, 0):
    w_vecs.append(pj.abtovec(M_snaps[r][0][ci], M_snaps[r][1][ci]))
    
sw, lw = pj.shapeslengths(M_snaps[r][0][ci])
sb, lb = pj.shapeslengths(M_snaps[r][1][ci])
wkeys = M_snaps[r][0][ci].keys()
bkeys = M_snaps[r][1][ci].keys()

u_hat_vec, v_hat_vec, u_norm, v_norm, inner = \
pj.abcvectobasis(*w_vecs, wkeys, bkeys, sw, lw, sb, lb)
#%%
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)  

tr_dict = {
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal
} 
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

with open("sm_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    
DNN_dict['number of layers'] = int(line.split(',')[0][2:])

#%%
pars = np.linspace(-5, 10, 30)
error_mat = []

for par_1 in pars:
    error_v = []
    for par_2 in pars:        
        wvec_new = w_vecs[0] + par_1 * u_hat_vec + par_2 * v_hat_vec
        
        weights, biases = pj.cvectodict(wvec_new, wkeys, bkeys, sw, lw, sb, lb)      
        
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:  
            model = DNN.standard(DNN_dict, sess, seed = 1)
            error_v.append(model.score_w(tr_dict, weights, biases)[0])

    error_mat.append(error_v)   
#%%
basis = np.concatenate((u_hat_vec.reshape(-1,1), v_hat_vec.reshape(-1,1)), axis = 1)
ata = np.linalg.inv(np.matmul(basis.transpose(), basis))

init_proj_x = []
init_proj_y = []

# for c in range(num):
#     prx, pry = pj.projtoplane(pj.abtovec(init_weights[c], init_biases[c]), w1vec, u_hat_vec, v_hat_vec)
#     init_proj_x.append(prx)
#     init_proj_y.append(pry)    

xx, yy = np.meshgrid(pars, pars)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training loss contour plots', fontsize = 30)

im1 = ax1.contourf(xx, yy, np.array(error_mat).transpose(), 200, origin='lower', cmap='RdGy')
fig.colorbar(im1, ax=ax1)
ax1.scatter([0, u_norm, inner / u_norm], [0, 0, v_norm], marker = 'x', color = 'k', s = 50, label = 'final')
# ax1.scatter(init_proj_x, init_proj_y, marker = 'x', color = 'm', s = 50, label = 'initial')
ax1.legend();

im2 = ax2.contour(xx, yy, np.array(error_mat).transpose(), 40, origin='lower', cmap='RdGy')
fig.colorbar(im2, ax=ax2)
ax2.scatter([0, u_norm, inner / u_norm], [0, 0, v_norm], color = 'k', marker = 'x', s = 50, label = 'final')
# ax2.scatter(init_proj_x, init_proj_y, marker = 'x', color = 'm', s = 50, label = 'initial')
ax2.legend();       
#%%    
w_vecs = []
for ci in range(-3, 0):
    w_vecs.append(pj.abtovec(CA_snaps[r][0][ci], CA_snaps[r][1][ci]))
    
# sw, lw = pj.shapeslengths(M_snaps[r][0][ci])
# sb, lb = pj.shapeslengths(M_snaps[r][1][ci])
# wkeys = M_snaps[r][0][ci].keys()
# bkeys = M_snaps[r][1][ci].keys()

u_hat_vec, v_hat_vec, u_norm, v_norm, inner = \
pj.abcvectobasis(*w_vecs, wkeys, bkeys, sw, lw, sb, lb)    
#%%
pars = np.linspace(-5, 20, 30)
error_mat = []

for par_1 in pars:
    error_v = []
    for par_2 in pars:        
        wvec_new = w_vecs[0] + par_1 * u_hat_vec + par_2 * v_hat_vec
        
        weights, biases = pj.cvectodict(wvec_new, wkeys, bkeys, sw, lw, sb, lb)      
        
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:  
            model = DNN.standard(DNN_dict, sess, seed = 1)
            error_v.append(model.score_w(tr_dict, weights, biases)[0])

    error_mat.append(error_v)   
#%%
basis = np.concatenate((u_hat_vec.reshape(-1,1), v_hat_vec.reshape(-1,1)), axis = 1)
ata = np.linalg.inv(np.matmul(basis.transpose(), basis))

init_proj_x = []
init_proj_y = []

# for c in range(num):
#     prx, pry = pj.projtoplane(pj.abtovec(init_weights[c], init_biases[c]), w1vec, u_hat_vec, v_hat_vec)
#     init_proj_x.append(prx)
#     init_proj_y.append(pry)    

xx, yy = np.meshgrid(pars, pars)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training loss contour plots', fontsize = 30)

im1 = ax1.contourf(xx, yy, np.array(error_mat).transpose(), 200, origin='lower', cmap='RdGy')
fig.colorbar(im1, ax=ax1)
ax1.scatter([0, u_norm, inner / u_norm], [0, 0, v_norm], marker = 'x', color = 'k', s = 50, label = 'final')
# ax1.scatter(init_proj_x, init_proj_y, marker = 'x', color = 'm', s = 50, label = 'initial')
ax1.legend();

im2 = ax2.contour(xx, yy, np.array(error_mat).transpose(), 40, origin='lower', cmap='RdGy')
fig.colorbar(im2, ax=ax2)
ax2.scatter([0, u_norm, inner / u_norm], [0, 0, v_norm], color = 'k', marker = 'x', s = 50, label = 'final')
# ax2.scatter(init_proj_x, init_proj_y, marker = 'x', color = 'm', s = 50, label = 'initial')
ax2.legend();     
    
    
    
    
    
    
    