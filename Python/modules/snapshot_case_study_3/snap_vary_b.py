# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:43:21 2020

@author: afpsaros
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:10:44 2020

@author: afpsaros
"""

import sys
sys.path.insert(0,'..')

from reg_classes import DNN
from data_disc import data_getter
from callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt 
from global_search_snapen import *
import pickle
import math
from itertools import combinations

import pickle
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
with open("snapen_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    

fit_dict['lr'] = float(line.split(',')[2][2:])
# print(lr, lr * 10)

lr_ratio = float(line.split(',')[4][1:-2])
# print(lr_ratio, lr_ratio * 10)
#%%
budgets = 6000 * np.arange(1, 6, 1)

c = 6

if sum([b % c for b in budgets]) != 0:
    raise ValueError('some number of cycles is not integer')    

#%%
reps = 6

B_errors = []
for b in budgets:
    fit_dict['num_epochs'] = b
    
    snap_step = int(b / c)
    
    print(b, c, snap_step)
    fit_dict['decay'] = ['cosine_restarts',snap_step, lr_ratio, 1., 1.]
    
    R_errors = []
    for r in range(reps):
        # print(j)
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:    
        
            callbacks = []
            snap = None if snap_step is None else Snapper(snap_step)   
            if snap is not None: callbacks.append(snap) 
            
            fit_dict['callbacks'] = callbacks
            model = DNN.standard(DNN_dict, sess, seed = r + 1)

            model.fit_from_dict(fit_dict)
            
            snap_weights, snap_biases = snap.get_snaps()
                       
            snap_range = range(c)          
            ensemble = model.fun_ensemble(snap_weights, snap_biases, snap_range)
                    
            pred = model.pred_ens(x_scal, ensemble)
            pred = data.scaler_y.inverse_transform(pred)
            R_errors.append(data.assess_pred(pred)[1])

        
    B_errors.append(np.mean(R_errors))
#%%
import pickle 

with open('vary_b_out.txt', 'wb') as f:
    pickle.dump([budgets, B_errors], f)  

# with open('vary_b_out.txt', 'rb') as f:
#     [budgets, B_errors, ms] = pickle.load(f)    
#%%          
plt.plot(budgets, B_errors, '-o')
plt.show()    





            
            
            
            
            
            
            
            