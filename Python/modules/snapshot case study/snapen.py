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
from global_search import *

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
    'number of layers': 6,
    'layer width': 20, 
}

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': None,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,    
    'lr': 1e-3,
    'decay': None,
}
#%%
budgets = 600 * np.arange(1, 6, 1)

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
        fit_dict['decay'] = ['cosine_restarts',snap_step, 0.01, 1., 1.]
        
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




            
            
            
            
            
            
            
            