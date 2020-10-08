# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:53:15 2020

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

def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

with open('sm_out.txt', 'rb') as f:
    [budgets, M_snaps, M_errors] = pickle.load(f)
    
M = 10

ENS_errors = []

ms = np.arange(2, M + 1, 2)
for m in ms:
    
    combs = combinations(np.arange(0, M, 1), m) 
    print(m)
    # print(len(list(perms)))
    errors = []
    # print(m)
    
    for comb in combs:

        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:
            model = DNN.standard(DNN_dict, sess, seed = 1) 
            
            for i in range(m):
                # print(i)
                
                sw, sb = M_snaps[comb[i]][0][-1], M_snaps[comb[i]][1][-1]
            
                # pred = model.pred_w(x_scal, sw, sb)
                # pred = data.scaler_y.inverse_transform(pred)
                
                if i == 0:
                    ensemble = tf.math.divide(model.fun_test(sw, sb).fp, m)
                else:
                    ensemble = tf.math.add(ensemble, tf.math.divide(model.fun_test(sw, sb).fp, m))   
                    
            # print(ensemble)
            # print('1', data.assess_pred(ensemble[0]))
            pred = model.pred_ens(x_scal, ensemble)
            pred = data.scaler_y.inverse_transform(pred)
            errors.append(data.assess_pred(pred)[0])
            
    ENS_errors.append(errors)
            
    
#%%
# for mi, m in enumerate(ms): 
#     plt.scatter(m * np.ones(len(ENS_errors[mi])), ENS_errors[mi])   

# plt.plot(ms, [np.mean(el) for el in ENS_errors])    

# sm_mean = np.mean(list(zip(*M_errors))[-1])

# plt.plot(ms, sm_mean * np.ones(len(ms)))         
# plt.show()
# print('rel', (sm_mean - np.mean(ENS_errors[-1])) / sm_mean * 100) 
        
#%%
import pickle 

with open('ens_out.txt', 'wb') as f:
    pickle.dump([ms, ENS_errors], f)      
    
# with open('ens_out.txt', 'rb') as f:
#     [ms, ENS_errors] = pickle.load(f)
    
