# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:38:23 2020

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
    'wd_par': 0,
    'num_epochs': 1000,
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

sess = tf.Session()
#%%
tocs = []
model = DNN.standard(DNN_dict, sess)
model.initialize(fit_dict['Xt'], fit_dict['Yt'])

for i in range(5):
    print(i)
    tic = time.perf_counter()
    model.fit_from_dict(fit_dict)
    tocs.append(time.perf_counter()-tic)
  
plt.plot(tocs)
plt.show()