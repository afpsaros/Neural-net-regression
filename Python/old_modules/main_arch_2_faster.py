# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:08:49 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from data_burgers import data_getter
from reg_classes import DNN
from hyper_opt import grid_cv_arch


sess = tf.Session()
for nu in [0.1, 0.5, 1]:
    print('nu = ', nu)
    
    n = 60
    s = 0
    val_split = 0.7
    n_eval = 50
    n_time = 1

    data = data_getter(n, s, val_split, n_eval, nu, n_time).create_data_time()

    Xt, Yt = data.data_tr[0][:, [0]], data.data_tr[0][:, [1]]
    Xv, Yv = data.data_val[0][:, [0]], data.data_val[0][:, [1]]
    Xe = data.data_eval[0][:, [0]]
    
    DNN_dict = {
        'input dimension': 1,
        'output dimension': 1,
        'number of layers': None,
        'layer width': None 
    }
    
    fit_dict = {
        'initialize': 0,
        'wd_par': None,
        'num_epochs': None,
        'Xt': Xt,
        'Yt': Yt,
        'Xv': Xv,
        'Yv': Yv,
        'lr': None
    }
    
    val_dict = {
        'Xe': Xv,
        'Ye': Yv
    }
    
    
    ev_params = {
            'num_epochs': [10000, 15000],
            'wd_par': [0] + list(10**(-np.arange(3, 5, dtype = float))),
            'lr': list(10**(-np.arange(2, 5, dtype = float))),
            }
    
    ev_arch = {
            'number of layers': [2, 4, 6, 8],
            'layer width': [30, 60, 90]
            }

    refit = 1
    adv_refit = 1

    print(ev_params)
    arch_cv = grid_cv_arch(ev_params, refit, adv_refit, ev_arch)
    
    scores, best, model = arch_cv.fit(fit_dict, val_dict, DNN_dict, sess)
    print(scores)
    print('best', best)
        
    
    plt.yscale('log')
    plt.plot(model.tr_error)
    plt.plot(model.val_error)
    plt.show()
    
    plt.plot(model.tr_error)
    plt.plot(model.val_error)
    plt.show()
    
    data.plot_eval_data(0)
    plt.plot(Xe, model.pred(Xe), '.')
    plt.show()