# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:50:32 2020

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
r1, r2 = 4, 1

plane_ws = [[0], M_snaps[r1][0][-1], M_snaps[r2][0][-1]]
plane_bs = [[0], M_snaps[r1][1][-1], M_snaps[r2][1][-1]]

pars_1 = np.linspace(-5, 40, 30)

error_mat_1, _for_projection_1, (u_norm_1, v_norm_1, inner_1) = \
    pj.createplane(plane_ws, plane_bs, pars_1, DNN_dict, tr_dict)
#%%
plane_ws = [[0], CA_snaps[r1][0][-1], CA_snaps[r2][0][-1]]
plane_bs = [[0], CA_snaps[r1][1][-1], CA_snaps[r2][1][-1]]

pars_2 = np.linspace(-5, 50, 30)
 
error_mat_2, _for_projection_2, (u_norm_2, v_norm_2, inner_2) = \
    pj.createplane(plane_ws, plane_bs, pars_2, DNN_dict, tr_dict)
#%%   
xx, yy = np.meshgrid(pars_1, pars_1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title('Standard learning rate')
norm = plt.Normalize(0, 2)
im1 = ax1.contourf(xx, yy, np.array(error_mat_1).transpose(), 200, origin='lower', cmap='RdGy', norm = norm)
fig.colorbar(im1, ax=ax1)

ax1.scatter([0], [0], marker = 's', s = 50, label = 'origin')

projected = pj.projmultoplane([M_inits[r1][0]], [M_inits[r1][1]], _for_projection_1)
ax1.scatter(*projected, marker = '.', color = 'k', s = 100, label = 'initial')

projected = pj.projmultoplane([M_inits[r2][0]], [M_inits[r2][1]], _for_projection_1)
ax1.scatter(*projected, marker = '.', color = 'k', s = 100)

projected = pj.projmultoplane(M_snaps[r1][0][:-1], M_snaps[r1][1][:-1], _for_projection_1)
ax1.scatter(*projected, marker = '.', color = 'm', s = 50, label = 'snaps')

projected = pj.projmultoplane(M_snaps[r2][0][:-1], M_snaps[r2][1][:-1], _for_projection_1)
ax1.scatter(*projected, marker = '.', color = 'm', s = 50)

ax1.scatter([u_norm_1], [0], marker = 'x', color = 'k', s = 50)
ax1.scatter([inner_1 / u_norm_1], [v_norm_1], marker = 'x', color = 'k', s = 50, label = 'final')

# ax1.legend(); 
   
xx, yy = np.meshgrid(pars_2, pars_2)

ax2.set_title('Cosine annealing')
im2 = ax2.contourf(xx, yy, np.array(error_mat_2).transpose(), 200, origin='lower', cmap='RdGy', norm = norm)
fig.colorbar(im2, ax=ax2)

ax2.scatter([0], [0], marker = 's', s = 50, label = 'origin')

projected = pj.projmultoplane([M_inits[r1][0]], [M_inits[r1][1]], _for_projection_2)
ax2.scatter(*projected, marker = '.', color = 'k', s = 100, label = 'initial')

projected = pj.projmultoplane([M_inits[r2][0]], [M_inits[r2][1]], _for_projection_2)
ax2.scatter(*projected, marker = '.', color = 'k', s = 100)

projected = pj.projmultoplane(CA_snaps[r1][0][:-1], CA_snaps[r1][1][:-1], _for_projection_2)
ax2.scatter(*projected, marker = '.', color = 'm', s = 50, label = 'snaps')

projected = pj.projmultoplane(CA_snaps[r2][0][:-1], CA_snaps[r2][1][:-1], _for_projection_2)
ax2.scatter(*projected, marker = '.', color = 'm', s = 50)

ax2.scatter([u_norm_2], [0], marker = 'x', color = 'k', s = 50)
ax2.scatter([inner_2 / u_norm_2], [v_norm_2], marker = 'x', color = 'k', s = 50, label = 'final')

ax2.legend(bbox_to_anchor=(1.5, 1.0))

plt.tight_layout()
plt.savefig('origin_planes.png', dpi = 300)        
    
    
    
    
    