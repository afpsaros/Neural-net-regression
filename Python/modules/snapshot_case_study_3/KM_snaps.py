# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:01:18 2020

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

with open('ca_out.txt', 'rb') as f:
    [CA_snaps, CA_preds, CA_errors, SN_R_preds] = pickle.load(f)     

c = len(CA_snaps[0][0])
reps = len(CA_snaps)     
#%%   
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal

x = data.Xe.flatten()
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
    'num_epochs': None,
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
    
no_models_max = 6

no_models = np.arange(1, no_models_max + 1, 1)
#%%
KM_errors = []
KM_preds = []
for m in no_models:
    
    combs = combinations(np.arange(0, no_models_max, 1), m) 
    print(m)
    errors = []
    preds = []
    for comb in combs:

        g = tf.Graph()
        sess = tf.Session(graph = g)
        with g.as_default() as g:
            model = DNN.standard(DNN_dict, sess, seed = 1) 
            
            ew = [CA_snaps[comb[i]][0] for i in range(m)]
            ew = [item for sublist in ew for item in sublist]
            
            # print([comb[i] for i in range(m)])
            
            eb = [CA_snaps[comb[i]][1] for i in range(m)]
            eb = [item for sublist in eb for item in sublist]

            ens_range = range(c * m)          
            ensemble = model.fun_ensemble(ew, eb, ens_range)
                    
            pred = model.pred_ens(x_scal, ensemble)
            pred = data.scaler_y.inverse_transform(pred)
            
            preds.append(pred)
            errors.append(data.assess_pred(pred)[1])
    
    KM_preds.append(preds)        
    KM_errors.append(errors)
    
#%%
import pickle 

with open('km_out.txt', 'wb') as f:
    pickle.dump([no_models, KM_errors, KM_preds], f)      
    
# with open('km_out.txt', 'rb') as f:
#     [no_models, KM_errors, KM_preds] = pickle.load(f)    
    
    
    
        
