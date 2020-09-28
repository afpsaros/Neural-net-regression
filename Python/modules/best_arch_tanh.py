# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:20:18 2020

@author: afpsaros
"""
# import sys
# sys.path.insert(0, '/modules')

import matplotlib.pyplot as plt
from data_tanh import data_getter
from reg_classes import DNN

from hyper_opt import grid_cv_arch

import numpy as np


n = 100
s = 0
val_split = 0.8
n_eval = 200

depths, widths = [], []

DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

ev_params = {
        'num_epochs': [10000],
        'wd_par': [0],
        'lr': [1e-3]
        }

ev_arch = {
            'number of layers': np.arange(1, 11, 1),
            'layer width': np.arange(5, 105, 10)
            }
               
refit = 0
adv_refit = 1    
random_sel = 0
n_random = 0
#%%
nus = np.arange(5, 105, 5)
for nu in nus:
    
    print(nu)
    
    scale = 1
    
    data = data_getter(n, s, val_split, n_eval).create_data(nu).preproc(scale)
    # data.plot_tr_data()
    # data.plot_eval_data(1)
    
    fit_dict = {
        'initialize': 0,
        'wd_par': None,
        'num_epochs': None,
        'Xt': data.Xt_scal,
        'Yt': data.Yt_scal,
        'Xv': data.Xv_scal,
        'Yv': data.Yv_scal,
        'lr': None
    }

    # =============================================================================
    # n_random = 3
    # 
    # if random_sel == 1:
    #     ev_params = {
    #         'num_epochs': ([1000, 1500], 'b'),
    #         'wd_par': ([0, -5, -3], 'e'),
    #         'lr': ([-3, -2], 'd')
    #         }
    # else: 
    #     ev_params = {
    #             'num_epochs': [10000],
    #             'wd_par': np.linspace(1e-6, 1e-4, 2),
    #             'lr': np.linspace(1e-3, 1e-2, 2)
    #             }
    # =============================================================================
        
    arch_cv = grid_cv_arch(ev_params, refit, adv_refit, ev_arch, random_sel, n_random)
    scores, (best, _), model = arch_cv.fit(fit_dict, DNN_dict, seed = 0)      
    
    depths.append(best[0])
    widths.append(best[1])
#%%        
plt.plot(depths, '-o')
name = 'tanh_depths.png'
plt.savefig(name, dpi = 400)
plt.show()

plt.plot(widths, '-o')
name = 'tanh_widths.png'
plt.savefig(name, dpi = 400)
plt.show()        
#%%
file = open("tanh_archs.txt","w") 
file.write('{} \n'.format(list(nus)))  
file.write('{} \n'.format(depths))  
file.write('{} \n'.format(widths))  
file.close()         