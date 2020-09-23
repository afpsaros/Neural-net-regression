# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:51:08 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from data_file import data_getter
from reg_classes import DNN
from hyper_opt import grid_cv

n = 30 
s = 0
val_split = 0.8
n_eval = 50
data = data_getter(n, s, val_split, n_eval).create_data()

data.plot_tr_data()
data.plot_eval_data(1)

#%%
Xt, Yt = data.data_tr[:, [0]], data.data_tr[:, [1]]
Xv, Yv = data.data_val[:, [0]], data.data_val[:, [1]]
Xe = data.data_eval[:, [0]]

DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 4,
    'layer width': 50, 
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
        'wd_par': np.linspace(0, 1e-4, 3),
        'lr': np.linspace(1e-3, 1e-2, 3)
        }

refit = 1
adv_refit = 1

sess = tf.Session()
tic = time.perf_counter()
model = DNN.standard(DNN_dict, sess)
if fit_dict['initialize'] == 0:
    model.initialize(fit_dict['Xt'], fit_dict['Yt'])
cv = grid_cv(ev_params, refit, adv_refit)
scores, (best_params, min_score) = cv.fit(model, fit_dict, val_dict)

print(best_params, min_score)
print(time.perf_counter()-tic)

    
# 