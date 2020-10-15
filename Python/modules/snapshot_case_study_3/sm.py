# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:53:05 2020

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

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': 30000,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,    
    'lr': None,
    'decay': None,
}

with open("sm_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    
DNN_dict['number of layers'] = int(line.split(',')[0][2:])

width = line.split(',')[1]
DNN_dict['layer width'] = int(width.split(')')[0][1:])

fit_dict['lr'] = float(line.split(',')[2][2:])
#%%
M = 6
M_snaps = []
M_errors = []
M_preds = []
M_inits = []
for m in range(M):
    print(m)
    g = tf.Graph()
    sess = tf.Session(graph = g)
    with g.as_default() as g:    
        
        callbacks = []
        
        initial, final = 1, 0
        inifin = None if initial == 0 and final == 0 else InitialFinal(initial, final)
        if inifin is not None: callbacks.append(inifin)

        snap_step = 5000
        snap = None if snap_step is None else Snapper(snap_step)   
        if snap is not None: callbacks.append(snap) 
        
        fit_dict['callbacks'] = callbacks

        # model = DNN.standard(DNN_dict, sess, seed = 0)
        model = DNN.standard(DNN_dict, sess, seed = m + 1)

        model.fit_from_dict(fit_dict)
        
        _iw, _ib, _, _ = inifin.get_params()
        M_inits.append((_iw, _ib))
        
        snap_weights, snap_biases = snap.get_snaps()
        M_snaps.append((snap_weights, snap_biases))
        
        errors = []
        preds = []
        for i in range(0, len(snap_weights)):
            pred = model.pred_w(x_scal, snap_weights[i], snap_biases[i])
            pred = data.scaler_y.inverse_transform(pred)
            
            preds.append(pred)
            errors.append(data.assess_pred(pred)[1])
        
        M_preds.append(preds)    
        M_errors.append(errors)

budgets = snap_step * np.arange(1, len(snap_weights) + 1, 1)

#%%
file = open("sm_info.txt","w") 
file.write('arch {}, {} \n'.format(DNN_dict['number of layers'], DNN_dict['layer width']))  
file.write('learning rate {} \n'.format(fit_dict['lr']))  
file.write('num epochs {} \n'.format(fit_dict['num_epochs']))  
file.write('M {} \n'.format(M))  
file.write('snap step {} \n'.format(snap_step))  
file.close()  
#%%
import pickle 

with open('sm_out.txt', 'wb') as f:
    pickle.dump([budgets, M_snaps, M_preds, M_errors, M_inits], f)
    
# with open('sm_out.txt', 'rb') as f:
#     [budgets, M_snaps, M_errors] = pickle.load(f)    
    
# M = len(M_snaps)    
    
    
    
    
