# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:18:33 2020

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
    [budgets, M_snaps, M_preds, M_errors] = pickle.load(f)
    
no_models_max = 6

ENS_errors = []

no_models = np.arange(1, no_models_max + 1, 1)

#%%
print(len(M_snaps))
print(len(M_snaps[0]))
print(len(M_snaps[0][0]))

for m in no_models:
    
    combs = combinations(np.arange(0, no_models_max, 1), m) 
    print(m)
    
    for comb in combs:
        print(comb)
        
        print([comb[i] for i in range(m)])
        







