# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:00:08 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from data_burgers import data_getter
from reg_classes import DNN
from hyper_opt import param_cv

n = 200
s = 0
val_split = 0.7
nu = .1
normalize = 1
data = data_getter(n, s, val_split, nu).create_tr_data_3D().create_eval_data_3D(nt_eval = 3).preproc(normalize)
data.plot3D_train()
data.plot3D_eval(1)
#%%
DNN_dict = {
    'input dimension': 2,
    'output dimension': 1,
    'number of layers': 4,
    'layer width': 50, 
}

fit_dict = {
    'initialize': 0,
    'wd_par': None,
    'num_epochs': None,
    'Xt': data.Xt_norm,
    'Yt': data.Yt_norm,
    'Xv': data.Xv_norm,
    'Yv': data.Yv_norm,
    'lr': None
}

val_dict = {
    'Xv': data.Xv_norm,
    'Yv': data.Yv_norm
}
#%%
# =============================================================================
# ev_params = {
#         'num_epochs': ([1000, 1500], 'b'),
#         'wd_par': ([-6, -1], 'd'),
#         'lr': ([-3, -2], 'd')
#         }
# =============================================================================

ev_params = {
        'num_epochs': [1000, 1500],
        'wd_par': np.linspace(1e-6, 1e-4, 3),
        'lr': np.linspace(1e-3, 1e-2, 3)
        }

refit = 1
adv_refit = 1
#%%
sess = tf.Session()

model = DNN.standard(DNN_dict, sess)
if fit_dict['initialize'] == 0:
    model.initialize(fit_dict['Xt'], fit_dict['Yt'])
cv = param_cv(ev_params, refit, adv_refit, 0, 18)
scores, (best_params, min_score) = cv.fit(model, fit_dict, val_dict)

print(best_params)
print(min_score)
#%%
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


    
