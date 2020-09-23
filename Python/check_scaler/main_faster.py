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

n = 30 
s = 1
val_split = 0.8
n_eval = 50
normalize = 1
data = data_getter(n, s, val_split, n_eval).create_data().preproc(normalize)

data.plot_tr_data()
data.plot_eval_data(1)
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 4,
    'layer width': 50, 
}

fit_dict = {
    'initialize': 0,
    'wd_par': 5* 1e-4,
    'num_epochs': 1000,
    'Xt': data.Xt_norm,
    'Yt': data.Yt_norm,
    'Xv': data.Xv_norm,
    'Yv': data.Yv_norm,
    'lr': 0.001
}

sess = tf.Session()
#%%
model = DNN.standard(DNN_dict, sess)
model.initialize(fit_dict['Xt'], fit_dict['Yt'])
model.fit_from_dict(fit_dict)
#%%

data.plot_eval_data(0)
if normalize == 1:
    pred = data.scaler.inverse_transform(model.pred(data.Xe_norm))
else:
    pred = model.pred(data.Xe_norm)

plt.plot(data.Xe, pred)
plt.show()

print('error = ', np.linalg.norm(pred - data.Ye))


