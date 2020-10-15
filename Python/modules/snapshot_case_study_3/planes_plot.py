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
    [budgets, M_snaps, M_preds, M_errors, M_inits] = pickle.load(f)
   
with open('ca_out.txt', 'rb') as f:
    [CA_snaps, CA_preds, CA_errors, SN_R_preds] = pickle.load(f)    
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
pj = planes_projections(*M_snaps[0])    

reps = len(M_snaps)
c = len(M_snaps[0][0])
#%%
r = 4

plane_ws, plane_bs = M_snaps[r][0][-3:], M_snaps[r][1][-3:]

pars_1 = np.linspace(-6, 10, 50)

error_mat_1, _for_projection_1, (u_norm_1, v_norm_1, inner_1) = \
    pj.createplane(plane_ws, plane_bs, pars_1, DNN_dict, tr_dict)
#%%
plane_ws, plane_bs = CA_snaps[r][0][-3:], CA_snaps[r][1][-3:]  

pars_2 = np.linspace(-15, 20, 50)   
 
error_mat_2, _for_projection_2, (u_norm_2, v_norm_2, inner_2) = \
    pj.createplane(plane_ws, plane_bs, pars_2, DNN_dict, tr_dict)
#%%    
xx, yy = np.meshgrid(pars_1, pars_1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title('Standard learning rate')
norm = plt.Normalize(0, 3)
im1 = ax1.contourf(xx, yy, np.array(error_mat_1).transpose(), 200, origin='lower', cmap='RdGy', norm = norm)
fig.colorbar(im1, ax=ax1)

projected = pj.projmultoplane([M_inits[r][0]], [M_inits[r][1]], _for_projection_1)
ax1.scatter(*projected, marker = '.', color = 'k', s = 100, label = 'initial')

projected = pj.projmultoplane(M_snaps[r][0][:-3], M_snaps[r][1][:-3], _for_projection_1)
ax1.scatter(*projected, marker = 'x', color = 'm', s = 50, label = 'early snaps')
ax1.scatter([0], [0], marker = 'x', color = 'y', s = 50, label = 'snap -2')
ax1.scatter([u_norm_1], [0], marker = 'x', color = 'b', s = 50, label = 'snap -1')
ax1.scatter([inner_1 / u_norm_1], [v_norm_1], marker = 'x', color = 'k', s = 50, label = 'final')

# ax1.legend(); 

xx, yy = np.meshgrid(pars_2, pars_2)

ax2.set_title('Cosine annealing')
im2 = ax2.contourf(xx, yy, np.array(error_mat_2).transpose(), 200, origin='lower', cmap='RdGy', norm = norm)
fig.colorbar(im2, ax=ax2)

projected = pj.projmultoplane([M_inits[r][0]], [M_inits[r][1]], _for_projection_2)
ax2.scatter(*projected, marker = '.', color = 'k', s = 100, label = 'initial')

projected = pj.projmultoplane(CA_snaps[r][0][:-3], CA_snaps[r][1][:-3], _for_projection_2)
ax2.scatter(*projected, marker = 'x', color = 'm', s = 50, label = 'early snaps')
ax2.scatter([0], [0], marker = 'x', color = 'y', s = 50, label = 'snap -2')
ax2.scatter([u_norm_2], [0], marker = 'x', color = 'b', s = 50, label = 'snap -1')
ax2.scatter([inner_2 / u_norm_2], [v_norm_2], marker = 'x', color = 'k', s = 50, label = 'final')

ax2.legend(bbox_to_anchor=(1.6, 1.0))

plt.tight_layout()
plt.savefig('planes.png', dpi = 300)   

#%%
# r = 4

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

combs = [[-1, i] for i in range(-6, -1)] 

pars = np.linspace(0, 1, 11)

ax1.set_title('Standard learning rate')
for c in range(len(combs)):
    
    _ws = [M_snaps[r][0][combs[c][0]], M_snaps[r][0][combs[c][1]]]
    _bs = [M_snaps[r][1][combs[c][0]], M_snaps[r][1][combs[c][1]]]
    
    error_line, _ = pj.lineloss(_ws, _bs, pars, DNN_dict, tr_dict, None, None)
   
    ax1.plot(pars, error_line, '-o', label = 'snaps %.1d and %.1d' %(combs[c][0] + 1, combs[c][1] + 1))

ax1.set_ylim([-0.01, 0.35])

pars = np.linspace(0, 1, 11)

ax2.set_title('Cosine annealing')
for c in range(len(combs)):
    
    _ws = [CA_snaps[r][0][combs[c][0]], CA_snaps[r][0][combs[c][1]]]
    _bs = [CA_snaps[r][1][combs[c][0]], CA_snaps[r][1][combs[c][1]]]
    
    error_line, _ = pj.lineloss(_ws, _bs, pars, DNN_dict, tr_dict, None, None)
   
    ax2.plot(pars, error_line, '-o', label = 'snaps %.1d and %.1d' %(combs[c][0] + 1, combs[c][1] + 1))

ax2.set_ylim([-0.01, 0.35])
ax2.legend(bbox_to_anchor=(1.4, 1.0))


plt.tight_layout()
plt.savefig('line_plots.png', dpi = 300)    

plt.show()  





    
    
    
    
    