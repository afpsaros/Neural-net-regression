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

M = 10
ms = np.arange(2, M + 1, 2)
# ms = 2 * np.arange(1, 6, 1)

if sum([sum([b % m for m in ms]) for b in budgets]) != 0:
    raise ValueError('some number of cycles is not integer')    

#%%
rep = 1

B_errors = []
for b in budgets:
    fit_dict['num_epochs'] = b
    
    m_errors = []
    for m in ms:
        
        snap_step = int(b / m)
        
        print(b, m, snap_step)
        fit_dict['decay'] = ['cosine_restarts',snap_step, lr_ratio, 1., 1.]
        
        rep_errors = []
        for j in range(rep):
            # print(j)
            g = tf.Graph()
            sess = tf.Session(graph = g)
            with g.as_default() as g:    
            
                callbacks = []
                snap = None if snap_step is None else Snapper(snap_step)   
                if snap is not None: callbacks.append(snap) 
                
                fit_dict['callbacks'] = callbacks
                model = DNN.standard(DNN_dict, sess, seed = 0)
    
                model.fit_from_dict(fit_dict)
                
                snap_weights, snap_biases = snap.get_snaps()
                
                for i in range(m):
                    # print(i)
                    
                    sw, sb = snap_weights[i], snap_biases[i]
                    
                    if i == 0:
                        ensemble = tf.math.divide(model.fun_test(sw, sb).fp, m)
                    else:
                        ensemble = tf.math.add(ensemble, tf.math.divide(model.fun_test(sw, sb).fp, m))   
                        
                pred = model.pred_ens(x_scal, ensemble)
                pred = data.scaler_y.inverse_transform(pred)
                rep_errors.append(data.assess_pred(pred)[0])
            
        m_errors.append(np.mean(rep_errors))
        
    B_errors.append(m_errors)
#%%
import pickle 

with open('snapen_out.txt', 'wb') as f:
    pickle.dump([budgets, B_errors, ms], f)  

# with open('snapen_out.txt', 'rb') as f:
#     [budgets, B_errors, ms] = pickle.load(f)    
#%%
for mi, m in enumerate(ms):
           
    plt.plot(budgets, list(zip(*B_errors))[mi], '-o', label = '{}'.format(m))

plt.legend()
plt.show()    

plt.plot(ms, B_errors[-1], '-o')




            
            
            
            
            
            
            
            