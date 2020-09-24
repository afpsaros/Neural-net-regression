# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:36:19 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from data_burgers import data_getter
from reg_classes import DNN

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
    'number of layers': 4,
    'layer width': 50 
}

fit_dict = {
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': 10000,
    'Xt': Xt,
    'Yt': Yt,
    'Xv': Xv,
    'Yv': Yv,
    'lr': 0.001
}

val_dict = {
    'Xe': Xv,
    'Ye': Yv
}
#%%
sess = tf.Session()

model = DNN.standard(DNN_dict, sess)

model.fit_from_dict(fit_dict)
#%%
data.plot2D_eval(0)
for i in range(len(data.times)+1):
    plt.plot(Xe[i * data.xs:data.xs + i * data.xs, 0], model.pred(Xe[i * data.xs:data.xs + i * data.xs, 0:2]), '.')
