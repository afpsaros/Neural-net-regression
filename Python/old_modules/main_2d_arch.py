# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:48:17 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from data_burgers import data_getter
from reg_classes import DNN
from hyper_opt import grid_cv_arch

n = 200
s = 0
val_split = 0.7
nu = .1

data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 2)
data.plot3D_train()
data.plot3D_eval(1)

#%%
Xt, Yt = data.data_tr[:, 0:2], data.data_tr[:, [2]]
Xv, Yv = data.data_val[:, 0:2], data.data_val[:, [2]]
print(Xt.shape, Yt.shape, Xv.shape, Yv.shape)
DNN_dict = {
    'input dimension': 2,
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
        'num_epochs': [10000],
        'wd_par': [0] + list(10**(-np.arange(3, 3, dtype = float))),
        'lr': list(10**(-np.arange(2, 3, dtype = float))),
        }

ev_arch = {
        'number of layers': [2],
        'layer width': [20, 40]
        }

refit = 1
adv_refit = 1


#%%
sess = tf.Session()

arch_cv = grid_cv_arch(ev_params, refit, adv_refit, ev_arch)

scores, best, model = arch_cv.fit(fit_dict, val_dict, DNN_dict, sess)
#%%
xlen = data.xs
data.create_eval_data_3D(nt_eval = 4)
Xe = data.data_eval[:, 0:2]
data.plot2D_eval(0)
for i in range(len(data.times)+1):
    plt.plot(Xe[i * xlen:xlen + i * xlen, 0], model.pred(Xe[i * xlen:xlen + i * xlen, 0:2]), '.')
    
    
    
    
