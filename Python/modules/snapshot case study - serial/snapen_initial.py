# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:44:24 2020

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
#%%
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

refit = 0
r_callbacks = None  
n_random = 30
random_seed = 3

# ev_arch = {
#         'number of layers': ([8], 'a'),
#         'layer width': ([80], 'a')
#         }
ev_arch = {}
    # 'wd_par': ([0, -5, -3], 'e'),
ev_params = {
    'num_epochs': ([30000], 'b'),
    'lr': ([-4, -2], 'd'),
    'callbacks': ([15000], 'b'),
    'decay': ([-2, -1], 'd')
    }    

#%%
arch_cv = random_global_cv(ev_params, ev_arch, refit, r_callbacks, n_random, random_seed)
scores, best, model = arch_cv.fit(fit_dict, DNN_dict)
#%%
print(arch_cv.best)
#%%
file = open("snapen_best_arch.txt","w") 
file.write('{} \n'.format(arch_cv.best[0]))  
file.close()  
#%%
with open("snapen_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    

lr = float(line.split(',')[2][2:])
print(lr, lr * 10)

lr_ratio = float(line.split(',')[4][1:-2])
print(lr_ratio, lr_ratio * 10)