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
#%%   
n = 200 
s = 0
val_split = 0.7
n_eval = 200
  
scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
data.plot_tr_data()
data.plot_eval_data(1)

x_scal = data.Xe_scal
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 8,
    'layer width': 80, 
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
    'lr': 5 * 1e-4,
    'decay': None,
}
#%%
M = 10
M_snaps = []
M_errors = []
for m in range(M):
    print(m)
    g = tf.Graph()
    sess = tf.Session(graph = g)
    with g.as_default() as g:    
        
        callbacks = []

        snap_step = 6000
        snap = None if snap_step is None else Snapper(snap_step)   
        if snap is not None: callbacks.append(snap) 
        
        fit_dict['callbacks'] = callbacks

        model = DNN.standard(DNN_dict, sess, seed = 0)

        model.fit_from_dict(fit_dict)
        
        snap_weights, snap_biases = snap.get_snaps()
        M_snaps.append((snap_weights, snap_biases))
        
        errors = []
        for i in range(0, len(snap_weights)):
            pred = model.pred_w(x_scal, snap_weights[i], snap_biases[i])
            pred = data.scaler_y.inverse_transform(pred)
            errors.append(data.assess_pred(pred)[0])
            
        M_errors.append(errors)

budgets = snap_step * np.arange(1, len(snap_weights) + 1, 1)
#%%
x = data.Xe.reshape(-1,1)
x_scal = data.Xe_scal
pred = model.pred(x_scal)
pred = data.scaler_y.inverse_transform(pred)
plt.plot(x, pred, '.')
data.plot_eval_data(1)
     
#%%
for i in range(0, len(budgets)):    
    plt.scatter(budgets[0] * (i + 1) * np.ones(M), list(zip(*M_errors))[i])    

plt.plot(budgets, np.mean(list(zip(*M_errors)), 1))  
plt.show()  
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

# with open('sm_out.txt', 'wb') as f:
#     pickle.dump([budgets, M_snaps, M_errors], f)
    
with open('sm_out.txt', 'rb') as f:
    [budgets, M_snaps, M_errors] = pickle.load(f)    
    
M = len(M_snaps)    
    
    
    
    
