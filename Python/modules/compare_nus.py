# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:07:08 2020

@author: afpsaros
"""

import matplotlib.pyplot as plt
from data_burgers import data_getter
from reg_classes import DNN

from global_search import random_global_cv

n = 200
s = 0
val_split = 0.7
normalize = 1

DNN_dict = {
    'input dimension': 2,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

refit = 1
adv_refit = 1    
random_sel = 1
n_random = 20
random_seed = 3

ev_arch = {
        'number of layers': ([2, 8], 'a'),
        'layer width': ([10, 60], 'a')
        }

ev_params = {
    'num_epochs': ([5000], 'b'),
    'wd_par': ([0, -5, -3], 'e'),
    'lr': ([-3, -2], 'd')
    }      

for nu in [1]:
    print('nu = ', nu)
    
    data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(normalize)
    data.plot3D_train()
    data.plot3D_eval(1)
    
    fit_dict = {
        'initialize': 1,
        'wd_par': None,
        'num_epochs': None,
        'Xt': data.Xt_norm,
        'Yt': data.Yt_norm,
        'Xv': data.Xv_norm,
        'Yv': data.Yv_norm,
        'lr': None
    }      
    
    arch_cv = random_global_cv(ev_params, ev_arch, refit, adv_refit, n_random, random_seed)
    scores, best, model = arch_cv.fit(fit_dict, DNN_dict)
    
    print(arch_cv.best)
    
    plt.plot(arch_cv.tocs) 
    plt.show()
    
    data.create_eval_data_3D(nt_eval = 5).preproc(normalize)
    
    xlen = data.xs
    
    data.plot2D_eval(0)
    x = data.Xe[:xlen, 0]
    if normalize == 1:
        for i in range(len(data.times)):
            
            pred = data.scaler_y.inverse_transform(model.pred(data.Xe_norm[i * xlen:xlen + i * xlen, 0:2]))
            plt.plot(x.reshape(-1,1), pred, '.')
    else:
        for i in range(len(data.times)):
            pred = model.pred(data.Xe_norm[i * xlen:xlen + i * xlen, 0:2])
            plt.plot(x, pred, '.')
    
    plt.show()    
    