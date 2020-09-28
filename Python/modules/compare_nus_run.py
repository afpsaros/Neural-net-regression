# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:19:18 2020

@author: afpsaros
"""

import matplotlib.pyplot as plt
from data_burgers import data_getter
from reg_classes import DNN
import time
from global_search import random_global_cv

import numpy as np

n = 200
s = 0
val_split = 0.7
scale = 1

DNN_dict = {
    'input dimension': 2,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

refit = 0
adv_refit = 1    

n_random = 2
random_seed = 3

ev_arch = {
        'number of layers': ([1, 10], 'a'),
        'layer width': ([5, 100], 'a')
        }

ev_params = {
    'num_epochs': ([100], 'b'),
    'wd_par': ([0, -5, -3], 'e'),
    'lr': ([-3, -2], 'd')
    }      

file = open("apo_results.txt","w") 

for nu in np.linspace(0.1, 1, 10):
    
    file.write('nu = {} \n'.format(nu))
    
    data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(scale)
    
    fit_dict = {
        'initialize': 1,
        'wd_par': None,
        'num_epochs': None,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
        'Xv': data.Xv_scal,
        'Yv': data.Yv_scal,
        'lr': None
    }      
    
    arch_cv = random_global_cv(ev_params, ev_arch, refit, adv_refit, n_random, random_seed)
    scores, best, model = arch_cv.fit(fit_dict, DNN_dict)
    
    file.write('{} \n'.format(arch_cv.best[0]))
    
file.close() 

import plot_nus