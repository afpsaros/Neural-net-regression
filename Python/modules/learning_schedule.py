# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:22:22 2020

@author: afpsaros
"""


import tensorflow as tf
import numpy as np
from callbacks import *
from reg_classes import DNN

import matplotlib.pyplot as plt 
from data_file import data_getter

n = 30
s = 0
val_split = 0.8
n_eval = 100

scale = 1

data = data_getter(n, s, val_split, n_eval).create_data().preproc(scale)
data.plot_tr_data()
data.plot_eval_data(1)


DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': 2,
    'layer width': 50, 
}

callbacks = []

snap_step = 99
snap = None if snap_step is None else Snapper(snap_step)   
if snap is not None: callbacks.append(snap) 

loss_hist = 1
loss = None if loss_hist == 0 else Losshistory()   
if loss is not None: callbacks.append(loss) 

lr_sched = 1
lr_inst = None if lr_sched == 0 else LRschedule()   
if lr_inst is not None: callbacks.append(lr_inst) 

fit_dict = {
    'callbacks': callbacks,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': 100,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,
    'lr': 1e-2,
    'decay': None,
}

snap_ratio = 0.001
fit_dict['decay'] = ['cosine_restarts',snap_step, snap_ratio, 1., 1.]

sess = tf.Session()
model = DNN.standard(DNN_dict, sess, seed = 1)
model.initialize(fit_dict['Xt'], fit_dict['Yt'])

model.fit_from_dict(fit_dict)
    
#%%
snap_weights, snap_biases = snap.get_snaps()
snap_epochs = snap.get_snap_epochs()
tr_error, val_error = loss.get_loss_history()
lr_sched = lr_inst.get_lr_schedule()

plt.yscale('log')
plt.plot(tr_error, label = 'training loss')
plt.plot(val_error, label = 'validation error')
plt.legend()
plt.show()


eta_max = fit_dict['lr']
eta_min = snap_ratio * eta_max
Ti = snap_step
lr_th = []
for epoch in range(fit_dict['num_epochs']):
    val = (epoch) % Ti
    lr_th.append(eta_min + 0.5 * (eta_max-eta_min)*(1+np.cos(val / Ti *np.pi)))

plt.yscale('log')
plt.plot(list(range(fit_dict['num_epochs']))[:Ti+1], lr_sched[:Ti+1], 'o')
plt.plot(list(range(fit_dict['num_epochs']))[:Ti+1], lr_th[:Ti+1], '*')
plt.axvline(x=snap_epochs[0])
plt.show()

plt.yscale('log')
plt.plot(list(range(fit_dict['num_epochs'])), lr_sched, 'o')
plt.plot(list(range(fit_dict['num_epochs'])), lr_th, '*')
for se in snap_epochs:
    plt.axvline(x=se)
plt.show()

print(sum([i-j for i,j in zip(lr_sched,lr_th)]))
