# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:22:22 2020

@author: afpsaros
"""


import sys
sys.path.insert(0,'..')

from reg_classes import DNN
from data_disc import data_getter
from callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt 
from global_search_snapen import *
import pickle
import math
from itertools import combinations

import pickle
#%%   
with open('data_instance.txt', 'rb') as f:
    data = pickle.load(f)    

x_scal = data.Xe_scal
#%%
with open('budget_info.txt', 'r') as f:
    l = list(f)
    line = l[0].strip()   
    
total_budget, mini_budget, given_step = [int(line.split(',')[i]) for i in range(3)]
#%%
DNN_dict = {
    'input dimension': 1,
    'output dimension': 1,
    'number of layers': None,
    'layer width': None, 
}

with open("sm_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    
DNN_dict['number of layers'] = int(line.split(',')[0][2:])

width = line.split(',')[1]
DNN_dict['layer width'] = int(width.split(')')[0][1:])

fit_dict = {
    'callbacks': None,
    'initialize': 1,
    'wd_par': 0,
    'num_epochs': None,
    'Xt': data.Xt_scal,
    'Yt': data.Yt_scal,
    'Xv': data.Xv_scal,
    'Yv': data.Yv_scal,    
    'lr': None,
    'decay': None,
}
#%%
with open("snapen_best_arch.txt", "r") as f:

    l = list(f)
    line = l[0].strip()
    

fit_dict['lr'] = float(line.split(',')[2][2:])
# print(lr, lr * 10)

lr_ratio = float(line.split(',')[4][1:-2])
# print(lr_ratio, lr_ratio * 10)
#%%
fit_dict['num_epochs'] = 30000


callbacks = []

snap_step = given_step
snap = None if snap_step is None else Snapper(snap_step)   
if snap is not None: callbacks.append(snap) 

loss_hist = 0
loss = None if loss_hist == 0 else Losshistory()   
if loss is not None: callbacks.append(loss) 

lr_sched = 1
lr_inst = None if lr_sched == 0 else LRschedule()   
if lr_inst is not None: callbacks.append(lr_inst) 

fit_dict['callbacks'] = callbacks

# fit_dict = {
#     'callbacks': callbacks,
#     'initialize': 1,
#     'wd_par': 0,
#     'num_epochs': 1000,
#     'Xt': data.Xt_scal,
#     'Yt': data.Yt_scal,
#     'Xv': data.Xv_scal,
#     'Yv': data.Yv_scal,
#     'lr': 1e-2,
#     'decay': None,
# }

# snap_ratio = 0.001
fit_dict['decay'] = ['cosine_restarts',given_step, lr_ratio, 1., 1.]

sess = tf.Session()
model = DNN.standard(DNN_dict, sess, seed = 1)
model.initialize(fit_dict['Xt'], fit_dict['Yt'])

model.fit_from_dict(fit_dict)
    
#%%
# snap_weights, snap_biases = snap.get_snaps()
snap_epochs = snap.get_snap_epochs()
# tr_error, val_error = loss.get_loss_history()
lr_sched = lr_inst.get_lr_schedule()

# plt.yscale('log')
# plt.plot(tr_error, label = 'training loss')
# plt.plot(val_error, label = 'validation error')
# plt.legend()
# plt.show()


eta_max = fit_dict['lr']
eta_min = lr_ratio * eta_max
Ti = snap_step
lr_th = []
for epoch in range(fit_dict['num_epochs']):
    val = (epoch) % Ti
    lr_th.append(eta_min + 0.5 * (eta_max-eta_min)*(1+np.cos(val / Ti *np.pi)))

plt.figure(figsize=(5,4))
plt.yscale('log')
plt.plot(list(range(fit_dict['num_epochs']))[999::1000], lr_sched[999::1000], 'o', label = 'theoretical')
plt.plot(list(range(fit_dict['num_epochs']))[999::1000], lr_sched[999::1000], '*', label = 'tensorflow')
plt.axvline(x=snap_epochs[0])
plt.axvline(x=snap_epochs[1])
plt.axvline(x=snap_epochs[2], label = 'snapshot')
plt.legend(loc = 'upper right')
plt.xlabel('epoch', fontsize = 15)
plt.ylabel('learning rate', fontsize = 15)
plt.tight_layout()
plt.savefig('learning_schedule.png', dpi = 300)
plt.show()

# plt.yscale('log')
# plt.plot(list(range(fit_dict['num_epochs'])), lr_sched, 'o')
# plt.plot(list(range(fit_dict['num_epochs'])), lr_th, '*')
# for se in snap_epochs:
#     plt.axvline(x=se)
# plt.show()

# print(sum([i-j for i,j in zip(lr_sched,lr_th)]))
