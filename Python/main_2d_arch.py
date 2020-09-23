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
nt_eval = 3
nu = .1
n_time = 2

data = data_getter(n, s, val_split, nt_eval, nu, n_time).create_data_3D()
data.plot3D_train()
data.plot3D_eval(1)

#%%
Xt, Yt = data.data_tr[:, 0:2], data.data_tr[:, [2]]
Xv, Yv = data.data_val[:, 0:2], data.data_val[:, [2]]
Xe, Ye = data.data_eval[:, 0:2], data.data_eval[:, [2]]

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
        'wd_par': [0] + list(10**(-np.arange(3, 5, dtype = float))),
        'lr': list(10**(-np.arange(2, 4, dtype = float))),
        }

ev_arch = {
        'number of layers': [2, 4, 6],
        'layer width': [20, 40]
        }

refit = 1
adv_refit = 1
#%%
sess = tf.Session()

arch_cv = grid_cv_arch(ev_params, refit, adv_refit, ev_arch)

scores, best, model = arch_cv.fit(fit_dict, val_dict, DNN_dict, sess)

data.plot2D_eval(0)
for i in range(len(data.times)+1):
    plt.plot(Xe[i * data.xs:data.xs + i * data.xs, 0], model.pred(Xe[i * data.xs:data.xs + i * data.xs, 0:2]), '.')
