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
    'num_epochs': 3000,
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
    [budgets, M_snaps, M_preds, M_errors] = pickle.load(f)
    
no_models_max = 6

ENS_errors = []

no_models = np.arange(1, no_models_max + 1, 1)
#%%
for m in no_models:
    
    combs = combinations(np.arange(0, no_models_max, 1), m) 
    print(m)
    errors = []
    
    for comb in combs:

        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:
            model = DNN.standard(DNN_dict, sess, seed = 1) 
            
            ens_weights = [M_snaps[comb[i]][0][-1] for i in range(m)]
            ens_biases = [M_snaps[comb[i]][1][-1] for i in range(m)]
                            
            ens_range = range(m)          
            ensemble = model.fun_ensemble(ens_weights, ens_biases, ens_range)
                    
            pred = model.pred_ens(x_scal, ensemble)
            pred = data.scaler_y.inverse_transform(pred)
            errors.append(data.assess_pred(pred)[0])
            
    ENS_errors.append(errors)
            
    
#%%
for mi, m in enumerate(no_models): 
    plt.scatter(m * np.ones(len(ENS_errors[mi])), ENS_errors[mi])   

plt.plot(no_models, [np.mean(el) for el in ENS_errors])    

sm_mean = np.mean(list(zip(*M_errors))[-1])

plt.plot(no_models, sm_mean * np.ones(len(no_models)))         
plt.show()
# print('rel', (sm_mean - np.mean(ENS_errors[-1])) / sm_mean * 100) 
        
#%%
import pickle 

with open('ens_out.txt', 'wb') as f:
    pickle.dump([no_models, ENS_errors], f)      
    
# with open('ens_out.txt', 'rb') as f:
#     [no_models, ENS_errors] = pickle.load(f)
    
