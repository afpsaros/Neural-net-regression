# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:23:38 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from data_burgers import data_getter
from reg_classes import DNN
from hyper_opt import grid_cv_arch

# various parameters
#num_epochs = 5000
#wd_par = None
#lr = None
sess = tf.Session()
for nu in [0.1, 0.5, 1]:
    n = 60
    s = 0
    val_split = 0.7
    n_eval = 50
    n_time = 1

    data = data_getter(n, s, val_split, n_eval, nu, n_time).create_data_time()
    
    # data.plot_tr_data()
    # data.plot_eval_data(1)


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
    
    # ev_params = {
    #         'num_epochs': [10],
    #         'wd_par': [0] + list(10**(-np.arange(2, 3, dtype = float))),
    #         # 'wd_par': [0],
    #         # 'lr': np.linspace(1e-3, 1e-2, 2)
    #         'lr': list(10**(-np.arange(2, 3, dtype = float))),
    #         }
    
    ev_params = {
            'num_epochs': [10000, 15000],
            'wd_par': [0] + list(10**(-np.arange(2, 5, dtype = float))),
            # 'wd_par': [0],
            # 'lr': np.linspace(1e-3, 1e-2, 2)
            'lr': list(10**(-np.arange(2, 5, dtype = float))),
            }
    
    ev_arch = {
            'number of layers': [2, 4, 6, 8],
            'layer width': [10, 30, 50, 70, 90]
            }
    
    # ev_arch = {
    #         'number of layers': [2, 4, 6, 8],
    #         'layer width': [10, 30, 50, 70, 90]
    #         }
    
    refit = 1
    adv_refit = 1
    
    #with tf.Session() as sess:
    
    
    tic = time.perf_counter()
    # model = DNN.standard(DNN_dict, sess)
    # cv = grid_cv(ev_params, refit, adv_refit)
    
    print(ev_params)
    arch_cv = grid_cv_arch(ev_params, refit, adv_refit, ev_arch)
    
    scores, best, model = arch_cv.fit(fit_dict, val_dict, DNN_dict, sess)
    print(scores)
    print('best', best)
    
    
    
    
    # scores, best_params, min_score = cv.fit(model, fit_dict, val_dict)
    
    # toc = time.perf_counter()
    # #    print(scores)
    # print(best_params, min_score)
    # print(toc-tic)
    # print((toc-tic)/60)
        
    # 
        
    
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