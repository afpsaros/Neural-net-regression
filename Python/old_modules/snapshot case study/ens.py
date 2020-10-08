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
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 8,
    'layer width': 250, 
}

n = 200 
s = 0
val_split = 0.7
n_eval = 200
  
scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
# data.plot_tr_data()
# data.plot_eval_data(1)

x_scal = data.Xe_scal

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
for mi, m in enumerate(ms): 
    plt.scatter(m * np.ones(len(ENS_errors[mi])), ENS_errors[mi])   

plt.plot(ms, [np.mean(el) for el in ENS_errors])    

sm_mean = np.mean(list(zip(*M_errors))[-1])

plt.plot(ms, sm_mean * np.ones(len(ms)))         
plt.show()
print('rel', (sm_mean - np.mean(ENS_errors[-1])) / sm_mean * 100) 
        
#%%
import pickle 

with open('ens_out.txt', 'wb') as f:
    pickle.dump([ms, ENS_errors], f)      
    
# with open('ens_out.txt', 'rb') as f:
#     [ms, ENS_errors] = pickle.load(f)
    
