# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:37:30 2020

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

#%%   
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal
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

width = line.split(',')[1]
DNN_dict['layer width'] = int(width.split(')')[0][1:])

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': None,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,    
    'lr': None,
    'decay': None,
}
#%%
snap_nums = np.arange(1, 7, 1)

NC_R_errors = []
for r in range(reps):
    print(r)
    g = tf.Graph()
    sess = tf.Session(graph = g)
    with g.as_default() as g:    

        model = DNN.standard(DNN_dict, sess, seed = 0)
        
        snap_weights, snap_biases = M_snaps[r]
  
        NC_errors = [] 
        for snap_num in snap_nums:   
            snap_range = np.arange(-1 - (snap_num - 1), 0, 1)
            
            ensemble = model.fun_ensemble(snap_weights, snap_biases, snap_range)
                
            pred = model.pred_ens(x_scal, ensemble)
            pred = data.scaler_y.inverse_transform(pred)
            
            NC_errors.append(data.assess_pred(pred)[1])
        
        NC_R_errors.append(NC_errors)
       
NC_R_means = [np.mean(line) for line in list(zip(*NC_R_errors))]
#%%
import pickle 

with open('NC_vary_snaps_out.txt', 'wb') as f:
    pickle.dump([snap_nums, NC_R_errors, NC_R_means], f)  